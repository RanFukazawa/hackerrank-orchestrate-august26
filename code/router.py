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
"""
