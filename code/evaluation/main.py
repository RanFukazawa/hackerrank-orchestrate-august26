"""
Self-evaluation harness.

Compares a generated output.csv against dataset/sample_messages.csv (the
solved examples with action/message_type/reason/confidence/evidence_message_ids
filled in) to sanity-check the router's predictions before submission.
Reports agreement on action and message_type, and flags rows where
evidence_message_ids or confidence look inconsistent with the samples'
style. Not used to train or hardcode behavior -- reference/style check only.
"""
