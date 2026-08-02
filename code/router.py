"""
Routing decision engine.

Combines the joined context (user, group/membership, business/history),
derived media text, retrieved evidence, and the safety guard's findings into
a final decision. Applies the priority cascade: unsafe/scam signals force
mute first; then repeated-ignored/dismissed patterns force mute or digest;
then time-boxed urgency from a trusted/relevant sender triggers notify;
otherwise the message defaults to digest. Also assigns message_type, a
short human-readable reason, a calibrated confidence, and the evidence
message ids for the final output row.

This module is pure rules: every decision is a deterministic function of
the structured signals in FeatureBundle, with no model call anywhere in
action selection. Two seams are marked "MODEL HOOK" below -- reason-text
generation and message_type classification -- where a model could be
layered in later purely as a swap of that one method's body. Nothing else
in this file, and no caller in main.py, would need to change to add it,
because action selection (the safety-critical output) stays rule-driven
either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loaders import Business, Group, GroupMembership, Message, User, UserBusinessHistory
from retrieval import EvidenceMatch
from safety import SafetyResult

# Time-boxed deadline / immediate-action language. Distinguishes "this
# needs a response in the next few minutes/hours" from generic chatter.
# Deliberately does not include words like "urgent" alone, since sender
# trust plus a concrete deadline is what the samples reward, not the
# presence of an alarming-sounding word by itself.
_URGENCY_PATTERNS: tuple[str, ...] = (
    r"\b\d+\s*(min|mins|minute|minutes)\b",
    r"\b\d+\s*(hour|hours|hrs)\b",
    r"\bright now\b",
    r"\bcome online now\b",
    r"\bleaving early\b",
    r"\basap\b",
    r"\bimmediately\b",
    r"\btoday by\b",
    r"\bbefore \d",
)

_GROUP_ADMIN_ROLES = ("admin", "owner")

# Business messages reporting the status of something the user is
# specifically waiting on (an order, a booking, a claim). Distinguishes
# these from generic business communication like feedback requests or
# safety advisories, which use similar polite/formal language but are not
# time-relevant to the user right now. This is a keyword heuristic, not
# real content understanding -- see the MODEL HOOK note below.
_BUSINESS_STATUS_UPDATE_PATTERNS: tuple[str, ...] = (
    r"\byour order\b",
    r"\byour (appointment|booking|reservation)\b",
    r"\bready for (review|pickup|collection)\b",
    r"\bexpected to (reach|arrive|deliver)\b",
    r"\bclaim is (ready|approved|processed)\b",
    r"\bpackage (has been|is) (packed|shipped|out for delivery)\b",
)


def _matches_urgency(text: str) -> bool:
    """Return True if text contains time-boxed deadline / immediate-action language."""
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _URGENCY_PATTERNS)


def _matches_business_status_update(text: str) -> bool:
    """Return True if text reports the status of a specific pending user order/booking/claim."""
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _BUSINESS_STATUS_UPDATE_PATTERNS)


def _parse_dnd_window(window: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse a 'HH:MM-HH:MM' do_not_disturb_window into (start_minutes, end_minutes) since midnight."""
    if not window:
        return None
    try:
        start_str, end_str = window.split("-")
        start_h, start_m = (int(part) for part in start_str.split(":"))
        end_h, end_m = (int(part) for part in end_str.split(":"))
    except (ValueError, AttributeError):
        return None
    return start_h * 60 + start_m, end_h * 60 + end_m


def _is_in_dnd(created_at_minutes: Optional[int], window: Optional[str]) -> bool:
    """Return True if created_at_minutes (minutes since midnight) falls inside an overnight-wrapping DND window."""
    parsed = _parse_dnd_window(window)
    if parsed is None or created_at_minutes is None:
        return False
    start, end = parsed
    if start <= end:
        return start <= created_at_minutes < end
    # Overnight window (e.g. 22:00-07:00) wraps past midnight.
    return created_at_minutes >= start or created_at_minutes < end


@dataclass
class FeatureBundle:
    """
    All joined context and derived signals for one message, assembled before routing.

    This is the single input to RoutingEngine.route(). Every field here is
    either raw joined data (message/user/group/business) or the output of
    an earlier pipeline stage (media.py's derived_text, retrieval.py's
    evidence, safety.py's SafetyResult) -- router.py itself does no
    joining or extraction, only decision-making.
    """

    message: Message
    user: Optional[User]
    group: Optional[Group]
    membership: Optional[GroupMembership]
    business: Optional[Business]
    biz_history: Optional[UserBusinessHistory]
    derived_text: str
    evidence: list[EvidenceMatch]
    safety: SafetyResult


@dataclass
class RoutingDecision:
    """The final per-message output row: action, message_type, reason, confidence, evidence ids."""

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str] = field(default_factory=list)

    def evidence_field(self) -> str:
        """Render evidence_message_ids as the ';'-joined string output.csv expects, or 'none'."""
        return ";".join(self.evidence_message_ids) if self.evidence_message_ids else "none"


class RoutingEngine:
    """
    Decides notify/digest/mute for one message via a fixed priority cascade.

    Order matters and is intentional: safety runs first and can never be
    overridden by sender trust or engagement history (mirrors safety.py's
    own no-instruction-following principle -- a message cannot argue its
    way out of being unsafe). Repetition/fatigue runs next, since a
    message that is individually harmless but part of a pattern the user
    ignores should not interrupt them. Urgency is checked only for
    messages that already passed the first two gates, so time-pressure
    language in a scam or in spam the user has muted before does not
    itself earn a notify. Anything left over defaults to digest: safe,
    not urgent, no negative history.
    """

    def route(self, features: FeatureBundle) -> RoutingDecision:
        """Run the full cascade for one message and return its RoutingDecision."""
        decision = self._check_safety(features)
        if decision is not None:
            return decision

        decision = self._check_repetition(features)
        if decision is not None:
            return decision

        decision = self._check_urgency(features)
        if decision is not None:
            return decision

        return self._default_digest(features)

    def _check_safety(self, features: FeatureBundle) -> Optional[RoutingDecision]:
        """Force mute if the safety guard flagged the message as unsafe (scam/injection/domain mismatch)."""
        if not features.safety.is_unsafe:
            return None

        message_type = features.safety.message_type_hint or "scam"
        reason = self._build_reason(features, rule="safety")
        return RoutingDecision(
            message_id=features.message.message_id,
            action="mute",
            message_type=message_type,
            reason=reason,
            confidence=self._compute_confidence(features, rule="safety"),
            evidence_message_ids=self._evidence_ids(features),
        )

    def _check_repetition(self, features: FeatureBundle) -> Optional[RoutingDecision]:
        """
        Mute or digest if evidence shows the user has a negative reaction pattern to similar messages.

        A negative pattern (dismissed/muted-after/reported outweighing
        opened) on matched evidence outranks a positive/neutral read: the
        samples mute repeated forwards, greetings, and promotions purely on
        this basis, with no safety signal involved. Group-muted membership
        is treated the same way, since it is an explicit user preference
        that content-level signals should not override.
        """
        if features.membership is not None and features.membership.group_muted_by_user:
            reason = self._build_reason(features, rule="group_muted")
            return RoutingDecision(
                message_id=features.message.message_id,
                action="mute",
                message_type=self._classify_message_type(features, rule="repetition"),
                reason=reason,
                confidence=self._compute_confidence(features, rule="group_muted"),
                evidence_message_ids=self._evidence_ids(features),
            )

        negative_matches = [m for m in features.evidence if m.reaction.is_negative_pattern()]
        if not negative_matches:
            return None

        reason = self._build_reason(features, rule="repetition")
        return RoutingDecision(
            message_id=features.message.message_id,
            action="mute",
            message_type=self._classify_message_type(features, rule="repetition"),
            reason=reason,
            confidence=self._compute_confidence(features, rule="repetition"),
            evidence_message_ids=self._evidence_ids(features),
        )

    def _check_urgency(self, features: FeatureBundle) -> Optional[RoutingDecision]:
        """
        Notify if the message has time-boxed urgency language from a trusted/relevant sender,
        or if it's a real transactional update from a verified/matched business.

        Trust is established by: group admin role, a verified business
        with a matching domain and real user_business_history, or positive
        (opened/replied) evidence from this sender/pattern before. Urgency
        alone from an unknown or untrusted sender is not enough -- that
        combination is exactly what safety.py's scam signals target, and
        this check only runs after _check_safety has already cleared the
        message.

        Verified-business transactional updates (order/appointment/booking
        status backed by real user_business_history) qualify even without
        explicit deadline keywords: an order-packed or appointment-ready
        notice is inherently time-relevant to the user despite not using
        words like "today" or "in 20 minutes", per the samples' 0.89-0.91
        confidence business_update notify cases.

        A message that would otherwise notify but arrives inside the
        user's do_not_disturb_window is downgraded to digest rather than
        muted: the content is still worth surfacing, but not by
        interrupting the user during quiet hours, per users.csv's
        do_not_disturb_window field.
        """
        combined_text = f"{features.message.message_text}\n{features.derived_text}"
        has_urgency_language = _matches_urgency(combined_text)
        is_transactional_business_update = (
            features.message.conversation_type == "business"
            and self._has_real_business_relationship(features)
            and _matches_business_status_update(combined_text)
        )
        if not has_urgency_language and not is_transactional_business_update:
            return None

        is_trusted = self._is_trusted_sender(features)
        if not is_trusted:
            return None

        if self._is_in_quiet_hours(features):
            reason = self._build_reason(features, rule="urgency_deferred")
            return RoutingDecision(
                message_id=features.message.message_id,
                action="digest",
                message_type=self._classify_message_type(features, rule="urgency"),
                reason=reason,
                confidence=self._compute_confidence(features, rule="urgency_deferred"),
                evidence_message_ids=self._evidence_ids(features),
            )

        reason = self._build_reason(features, rule="urgency")
        return RoutingDecision(
            message_id=features.message.message_id,
            action="notify",
            message_type=self._classify_message_type(features, rule="urgency"),
            reason=reason,
            confidence=self._compute_confidence(features, rule="urgency"),
            evidence_message_ids=self._evidence_ids(features),
        )

    def _is_in_quiet_hours(self, features: FeatureBundle) -> bool:
        """True if the message's created_at falls inside the user's do_not_disturb_window."""
        if features.user is None or features.message.created_at is None:
            return False
        created_at_minutes = features.message.created_at.hour * 60 + features.message.created_at.minute
        return _is_in_dnd(created_at_minutes, features.user.do_not_disturb_window)

    def _default_digest(self, features: FeatureBundle) -> RoutingDecision:
        """Fall through to digest: safe, not urgent, and no negative history was found."""
        reason = self._build_reason(features, rule="default")
        return RoutingDecision(
            message_id=features.message.message_id,
            action="digest",
            message_type=self._classify_message_type(features, rule="default"),
            reason=reason,
            confidence=self._compute_confidence(features, rule="default"),
            evidence_message_ids=self._evidence_ids(features),
        )

    def _has_real_business_relationship(self, features: FeatureBundle) -> bool:
        """True if the business is verified, its sender domain matches its official domain, and real history exists."""
        if features.business is None:
            return False
        domain_matches = features.business.domain_used_by_sender.lower() == features.business.official_domain.lower()
        has_real_relationship = features.biz_history is not None and (
            features.biz_history.activity_count_180d > 0 or features.biz_history.allows_promotions
        )
        return features.business.verified and domain_matches and has_real_relationship

    def _is_trusted_sender(self, features: FeatureBundle) -> bool:
        """True if the sender is a group admin, a verified/matching business, or has positive evidence history."""
        if features.membership is not None and features.membership.role in _GROUP_ADMIN_ROLES:
            return True

        if self._has_real_business_relationship(features):
            return True

        if any(match.reaction.is_positive_pattern() for match in features.evidence):
            return True

        return False

    # --- MODEL HOOK -----------------------------------------------------
    # message_type is currently assigned by rule-of-thumb per cascade
    # branch. A model given the same FeatureBundle (message text,
    # derived_text, conversation_type, and which rule fired) could make a
    # more nuanced choice among the allowed message_type values -- e.g.
    # distinguishing "event" from "urgent" from "personal" within the
    # urgency branch, which this method currently cannot do since it has
    # no read on the message's actual subject matter. Swapping this
    # method's body for a model call would not require any change to
    # route(), FeatureBundle, or RoutingDecision.
    def _classify_message_type(self, features: FeatureBundle, rule: str) -> str:
        """Assign a message_type consistent with the cascade branch that fired."""
        if rule == "safety":
            return features.safety.message_type_hint or "scam"

        if rule == "repetition":
            if features.message.forwarded_count > 0:
                return "forward"
            if features.message.conversation_type == "business":
                return "promotion"
            return "greeting"

        if rule == "urgency":
            if features.message.conversation_type == "business":
                return "business_update"
            return "urgent"

        # default/digest branch
        if features.message.conversation_type == "business":
            return "business_update"
        if features.message.forwarded_count > 0:
            return "forward"
        return "personal"

    # --- MODEL HOOK -----------------------------------------------------
    # reason is currently a fixed template per cascade branch, filled in
    # with the specific signal names that fired (from SafetyResult.signals
    # or the matched evidence). A model given the same FeatureBundle could
    # write a more natural, message-specific sentence instead of reusing
    # one of a handful of templates, which is what the evaluation
    # criteria's "usefulness and consistency of reason" rewards at the
    # margin. Swapping this method's body for a model call would not
    # require any change to route(), FeatureBundle, or RoutingDecision.
    def _build_reason(self, features: FeatureBundle, rule: str) -> str:
        """Build a short human-readable reason string consistent with the cascade branch that fired."""
        if rule == "safety":
            if "prompt_injection_attempt" in features.safety.signals:
                return "The message tries to instruct the router, but the routing decision is based on the actual content and risk."
            if "business_domain_mismatch" in features.safety.signals:
                return "The sender's domain does not match the business's official domain, a strong scam signal."
            if "credential_request" in features.safety.signals:
                return "The message asks for sensitive verification details under urgency or account-risk pressure."
            return "The message shows signals consistent with a scam or unsafe request."

        if rule == "group_muted":
            return "The user has muted this group, so its messages should not interrupt them."

        if rule == "repetition":
            if any(m.reaction.reported_count > 0 for m in features.evidence):
                return "Similar messages from this sender were previously reported by the user."
            return "The sender has a pattern of repeated messages that the user usually ignores or dismisses."

        if rule == "urgency":
            if features.message.conversation_type == "business":
                return "A verified business is sending a time-sensitive update relevant to the user's recent activity."
            if features.membership is not None and features.membership.role in _GROUP_ADMIN_ROLES:
                return "A trusted group admin sent a time-sensitive update that should interrupt the user."
            return "The sender is trusted and the message contains a direct deadline or action needed soon."

        if rule == "urgency_deferred":
            return "The message is time-sensitive and from a trusted sender, but it arrived during the user's quiet hours."

        # default/digest branch
        if features.message.conversation_type == "business":
            return "A business is sending a non-urgent update that does not need to interrupt the user."
        return "The message is safe but not urgent enough to interrupt the user right now."

    def _compute_confidence(self, features: FeatureBundle, rule: str) -> float:
        """
        Assign a confidence score reflecting how clear-cut the fired rule's evidence is.

        Ranges are informed by sample_messages.csv's observed clustering:
        notify/mute decisions with clear signals score highest (0.85-0.9),
        softer digest calls score lower (0.75-0.85). Safety and repetition
        decisions gain a further boost when directly backed by evidence
        with a negative reaction pattern, since a decision backed by the
        user's own history is more clear-cut than one from content alone.
        """
        if rule == "safety":
            base = 0.85
            if len(features.safety.signals) > 1:
                base += 0.03
            return min(0.95, base)

        if rule == "group_muted":
            return 0.9

        if rule == "repetition":
            base = 0.8
            if any(m.reaction.reported_count > 0 for m in features.evidence):
                base += 0.05
            if any(m.reaction.muted_after_count > 0 for m in features.evidence):
                base += 0.03
            return min(0.9, base)

        if rule == "urgency":
            base = 0.85
            if features.membership is not None and features.membership.role in _GROUP_ADMIN_ROLES:
                base += 0.02
            return min(0.9, base)

        if rule == "urgency_deferred":
            return 0.8

        # default/digest branch
        return 0.78

    def _evidence_ids(self, features: FeatureBundle) -> list[str]:
        """Return the evidence message ids from retrieval, in ranked order, for the output row."""
        return [match.message_id for match in features.evidence]
