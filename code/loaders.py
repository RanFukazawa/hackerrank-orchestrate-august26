"""
Dataset loading and joining.

Reads every participant-facing CSV in dataset/ (messages, users, groups,
group_members, business_accounts, user_business_history, message_history,
message_events, images, voice_notes, daily_notification_summary) and parses
them into typed in-memory structures. Builds the lookup indices the rest of
the pipeline joins against, e.g. user_id -> User, (group_id, user_id) ->
GroupMembership, (user_id, business_id) -> UserBusinessHistory, and
message_id -> list of MessageEvent. Does not read organizer-only files.
"""
