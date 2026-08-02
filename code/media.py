"""
Media resolution and inspection.

Resolves a message's media_id to an actual file path using images.csv or
voice_notes.csv, then inspects the file under dataset/media/ since the CSVs
only provide paths, not content. OCR and ASR are behind small abstract
interfaces (OCREngine / ASREngine) so the extraction backend can be swapped
without touching callers. The shipped implementations are local, free, and
API-key-independent: pytesseract for OCR and faster-whisper for ASR. Both
degrade to an empty string when the underlying engine or binary isn't
available in the environment, so the pipeline always runs end-to-end.

Claude's Messages API does not accept raw audio content blocks (only text,
image, PDF, and file references), so voice-note transcription cannot be
done via the Claude API directly -- faster-whisper (a local, open
speech-to-text model) is the working ASR path, independent of any API key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loaders import Dataset, Message


class OCREngine(ABC):
    """Abstract interface for extracting text from an image file."""

    @abstractmethod
    def extract(self, image_path: Path) -> str:
        """Return the text found in the image at image_path, or '' if none/unavailable."""
        raise NotImplementedError


class ASREngine(ABC):
    """Abstract interface for transcribing an audio file to text."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return the transcript of the audio at audio_path, or '' if none/unavailable."""
        raise NotImplementedError


class PytesseractOCREngine(OCREngine):
    """
    Local OCR fallback using pytesseract + Pillow.

    Requires the `tesseract` binary to be installed on the system in
    addition to the pytesseract/Pillow Python packages. If either is
    missing, or extraction fails for any reason, returns '' rather than
    raising, so a missing local dependency never crashes the pipeline.
    """

    def extract(self, image_path: Path) -> str:
        """Run local OCR on image_path, returning '' if pytesseract/tesseract is unavailable."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return ""

        try:
            with Image.open(image_path) as img:
                return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""


class WhisperASREngine(ASREngine):
    """
    Local ASR using faster-whisper (an optimized local Whisper implementation).

    Runs fully offline once the model weights are cached; no API key or
    network access is required at inference time. Uses the 'tiny' model for
    fast CPU transcription suited to short voice notes. If faster-whisper
    isn't installed, or transcription fails for any reason, returns ''
    rather than raising, so a missing local dependency never crashes the
    pipeline.
    """

    def __init__(self, model_size: str = "tiny") -> None:
        """Lazily create the WhisperModel on first use so import failures don't crash construction."""
        self._model_size = model_size
        self._model = None

    def _get_model(self):
        """Return the cached WhisperModel, loading it on first call."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        """Run local Whisper transcription on audio_path, returning '' if unavailable or it fails."""
        try:
            model = self._get_model()
        except ImportError:
            return ""

        try:
            segments, _info = model.transcribe(str(audio_path))
            return " ".join(segment.text for segment in segments).strip()
        except Exception:
            return ""


class NullASREngine(ASREngine):
    """
    No-op ASR fallback used when no transcription backend should run at all.

    Always returns ''. Useful for tests or environments where even the
    local Whisper dependency is unavailable/unwanted.
    """

    def transcribe(self, audio_path: Path) -> str:
        """Always returns '' -- this engine never transcribes."""
        return ""


@dataclass
class MediaInspection:
    """The result of resolving and inspecting a message's media, if any."""

    media_path: Optional[Path]
    derived_text: str


class MediaResolver:
    """
    Resolves a message's media_id to a file path and extracts its text.

    Looks up image/voice file paths from the loaded Dataset's images and
    voice_notes indices, then dispatches to the configured OCREngine or
    ASREngine to produce derived_text that downstream stages can reason
    over alongside message_text.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_dir: Path,
        ocr_engine: Optional[OCREngine] = None,
        asr_engine: Optional[ASREngine] = None,
    ) -> None:
        """Store the dataset media indices and the OCR/ASR engines to use for extraction."""
        self._dataset = dataset
        self._dataset_dir = dataset_dir
        self._ocr_engine = ocr_engine or PytesseractOCREngine()
        self._asr_engine = asr_engine or WhisperASREngine()

    def resolve_path(self, media_type: Optional[str], media_id: Optional[str]) -> Optional[Path]:
        """Return the absolute Path for a message's media_id, or None if there's no media."""
        if not media_type or not media_id:
            return None
        if media_type == "image":
            relative_path = self._dataset.images.get(media_id)
        elif media_type == "voice":
            relative_path = self._dataset.voice_notes.get(media_id)
        else:
            relative_path = None
        if not relative_path:
            return None
        return self._dataset_dir / relative_path

    def inspect(self, message: Message) -> MediaInspection:
        """Resolve and extract text for a message's media, returning '' derived_text if none/unavailable."""
        media_path = self.resolve_path(message.media_type, message.media_id)
        if media_path is None:
            return MediaInspection(media_path=None, derived_text="")

        if message.media_type == "image":
            derived_text = self._ocr_engine.extract(media_path)
        elif message.media_type == "voice":
            derived_text = self._asr_engine.transcribe(media_path)
        else:
            derived_text = ""

        return MediaInspection(media_path=media_path, derived_text=derived_text)
