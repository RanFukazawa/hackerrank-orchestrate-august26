"""
Media resolution and inspection.

Resolves a message's media_id to an actual file path using images.csv or
voice_notes.csv, then inspects the file under dataset/media/ since the CSVs
only provide paths, not content. Runs OCR on image posters/screenshots and
ASR (transcription) on voice notes to produce derived text that downstream
stages (retrieval, safety, routing) can reason over alongside message_text.
"""
