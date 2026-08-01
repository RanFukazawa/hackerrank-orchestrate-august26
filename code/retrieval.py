"""
Evidence retrieval from historical messages.

Searches message_history.csv for messages similar to the incoming one
(same sender/group/business and/or similar text content, including OCR/ASR
derived text) and pulls each match's user reaction from message_events.csv
(opened, replied, dismissed, muted_after, reported). Produces the ranked
evidence_message_ids and reaction-pattern signal used to detect repetition,
fatigue, or previously-flagged risk for this specific user.
"""
