"""
Dataset loading and joining.

Reads every participant-facing CSV in dataset/ (messages, users, groups,
group_members, business_accounts, user_business_history, message_history,
message_events, images, voice_notes) and parses them into typed dataclasses.
Builds the lookup indices the rest of the pipeline joins against, e.g.
user_id -> User, (group_id, user_id) -> GroupMembership, (user_id,
business_id) -> UserBusinessHistory, and message_id -> list of MessageEvent.
Does not read organizer-only files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _parse_datetime(value: str) -> Optional[datetime]:
    """Parse a 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD' timestamp, returning None for blank values."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d")


def _parse_bool(value: str) -> bool:
    """Parse a '0'/'1' CSV flag into a bool."""
    return value.strip() == "1"


def _parse_int(value: str) -> int:
    """Parse an integer CSV field, treating blank as 0."""
    return int(value) if value.strip() else 0


def _none_if_blank(value: str) -> Optional[str]:
    """Return None for an empty CSV field, otherwise the original string."""
    return value if value else None


@dataclass
class Message:
    """One incoming or historical WhatsApp message (messages.csv / message_history.csv row)."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: Optional[datetime]
    message_text: str
    media_type: Optional[str]
    media_id: Optional[str]
    forwarded_count: int


@dataclass
class User:
    """Per-user notification behavior baseline (users.csv row)."""

    user_id: str
    do_not_disturb_window: Optional[str]
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass
class Group:
    """Metadata about a group chat (groups.csv row)."""

    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: Optional[datetime]
    messages_30d: int


@dataclass
class GroupMembership:
    """One user's relationship to one group (group_members.csv row)."""

    group_id: str
    user_id: str
    role: str
    joined_at: Optional[datetime]
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool


@dataclass
class Business:
    """A business sender's identity and trust signals (business_accounts.csv row)."""

    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


@dataclass
class UserBusinessHistory:
    """One user's relationship history with one business (user_business_history.csv row)."""

    user_id: str
    business_id: str
    why_user_knows_account: Optional[str]
    last_activity_at: Optional[datetime]
    allows_promotions: bool
    promotions_opted_out_at: Optional[datetime]
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: Optional[datetime]


@dataclass
class MessageEvent:
    """How a user reacted to one historical message (message_events.csv row)."""

    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: Optional[int]
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


@dataclass
class DailyNotificationSummary:
    """A user's notification load for a single day (daily_notification_summary.csv row)."""

    user_id: str
    date: str
    notifications_sent: int
    notifications_dismissed: int


@dataclass
class Dataset:
    """
    The full set of joined lookup tables the pipeline needs.

    Each field is an index keyed for direct O(1) lookup rather than a raw
    list, since every downstream stage (media, retrieval, safety, router)
    joins against a specific message/user/group/business rather than
    scanning the whole table.
    """

    messages: list[Message] = field(default_factory=list)
    users: dict[str, User] = field(default_factory=dict)
    groups: dict[str, Group] = field(default_factory=dict)
    group_members: dict[tuple[str, str], GroupMembership] = field(default_factory=dict)
    businesses: dict[str, Business] = field(default_factory=dict)
    user_business_history: dict[tuple[str, str], UserBusinessHistory] = field(default_factory=dict)
    message_history: list[Message] = field(default_factory=list)
    message_events_by_message_id: dict[str, list[MessageEvent]] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)
    voice_notes: dict[str, str] = field(default_factory=dict)
    daily_notification_summary: dict[tuple[str, str], DailyNotificationSummary] = field(default_factory=dict)


def load_messages(path: Path) -> list[Message]:
    """Load messages.csv or message_history.csv (same schema) into a list of Message."""
    with path.open(newline="", encoding="utf-8") as f:
        return [
            Message(
                message_id=row["message_id"],
                user_id=row["user_id"],
                conversation_type=row["conversation_type"],
                group_id=_none_if_blank(row["group_id"]),
                business_id=_none_if_blank(row["business_id"]),
                sender_user_id=_none_if_blank(row["sender_user_id"]),
                created_at=_parse_datetime(row["created_at"]),
                message_text=row["message_text"],
                media_type=_none_if_blank(row["media_type"]),
                media_id=_none_if_blank(row["media_id"]),
                forwarded_count=_parse_int(row["forwarded_count"]),
            )
            for row in csv.DictReader(f)
        ]


def load_users(path: Path) -> dict[str, User]:
    """Load users.csv into a dict keyed by user_id."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["user_id"]: User(
                user_id=row["user_id"],
                do_not_disturb_window=_none_if_blank(row["do_not_disturb_window"]),
                messages_opened_30d=_parse_int(row["messages_opened_30d"]),
                messages_replied_30d=_parse_int(row["messages_replied_30d"]),
                notifications_dismissed_30d=_parse_int(row["notifications_dismissed_30d"]),
                messages_reported_30d=_parse_int(row["messages_reported_30d"]),
            )
            for row in csv.DictReader(f)
        }


def load_groups(path: Path) -> dict[str, Group]:
    """Load groups.csv into a dict keyed by group_id."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["group_id"]: Group(
                group_id=row["group_id"],
                group_name=row["group_name"],
                group_type=row["group_type"],
                member_count=_parse_int(row["member_count"]),
                admin_count=_parse_int(row["admin_count"]),
                created_at=_parse_datetime(row["created_at"]),
                messages_30d=_parse_int(row["messages_30d"]),
            )
            for row in csv.DictReader(f)
        }


def load_group_members(path: Path) -> dict[tuple[str, str], GroupMembership]:
    """Load group_members.csv into a dict keyed by (group_id, user_id)."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (row["group_id"], row["user_id"]): GroupMembership(
                group_id=row["group_id"],
                user_id=row["user_id"],
                role=row["role"],
                joined_at=_parse_datetime(row["joined_at"]),
                messages_sent_30d=_parse_int(row["messages_sent_30d"]),
                messages_read_30d=_parse_int(row["messages_read_30d"]),
                replies_sent_30d=_parse_int(row["replies_sent_30d"]),
                notifications_dismissed_30d=_parse_int(row["notifications_dismissed_30d"]),
                group_muted_by_user=_parse_bool(row["group_muted_by_user"]),
            )
            for row in csv.DictReader(f)
        }


def load_business_accounts(path: Path) -> dict[str, Business]:
    """Load business_accounts.csv into a dict keyed by business_id."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["business_id"]: Business(
                business_id=row["business_id"],
                display_name=row["display_name"],
                brand_name=row["brand_name"],
                category=row["category"],
                verified=_parse_bool(row["verified"]),
                official_domain=row["official_domain"],
                domain_used_by_sender=row["domain_used_by_sender"],
                account_age_days=_parse_int(row["account_age_days"]),
                messages_sent_30d=_parse_int(row["messages_sent_30d"]),
                user_reports_30d=_parse_int(row["user_reports_30d"]),
                domain_used_by_sender_age_days=_parse_int(row["domain_used_by_sender_age_days"]),
            )
            for row in csv.DictReader(f)
        }


def load_user_business_history(path: Path) -> dict[tuple[str, str], UserBusinessHistory]:
    """Load user_business_history.csv into a dict keyed by (user_id, business_id)."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (row["user_id"], row["business_id"]): UserBusinessHistory(
                user_id=row["user_id"],
                business_id=row["business_id"],
                why_user_knows_account=_none_if_blank(row["why_user_knows_account"]),
                last_activity_at=_parse_datetime(row["last_activity_at"]),
                allows_promotions=_parse_bool(row["allows_promotions"]),
                promotions_opted_out_at=_parse_datetime(row["promotions_opted_out_at"]),
                activity_count_180d=_parse_int(row["activity_count_180d"]),
                messages_opened_30d=_parse_int(row["messages_opened_30d"]),
                messages_dismissed_30d=_parse_int(row["messages_dismissed_30d"]),
                messages_replied_30d=_parse_int(row["messages_replied_30d"]),
                last_reply_at=_parse_datetime(row["last_reply_at"]),
            )
            for row in csv.DictReader(f)
        }


def load_message_events(path: Path) -> dict[str, list[MessageEvent]]:
    """Load message_events.csv into a dict keyed by message_id, grouping multiple reactions per message."""
    events_by_message_id: dict[str, list[MessageEvent]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            event = MessageEvent(
                user_id=row["user_id"],
                message_id=row["message_id"],
                message_opened=_parse_bool(row["message_opened"]),
                message_replied=_parse_bool(row["message_replied"]),
                reaction_time_minutes=(
                    _parse_int(row["reaction_time_minutes"]) if row["reaction_time_minutes"] else None
                ),
                notification_dismissed=_parse_bool(row["notification_dismissed"]),
                muted_after_message=_parse_bool(row["muted_after_message"]),
                message_reported=_parse_bool(row["message_reported"]),
            )
            events_by_message_id.setdefault(event.message_id, []).append(event)
    return events_by_message_id


def load_images(path: Path) -> dict[str, str]:
    """Load images.csv into a dict keyed by image_id, valued by file_path."""
    with path.open(newline="", encoding="utf-8") as f:
        return {row["image_id"]: row["file_path"] for row in csv.DictReader(f)}


def load_voice_notes(path: Path) -> dict[str, str]:
    """Load voice_notes.csv into a dict keyed by voice_note_id, valued by file_path."""
    with path.open(newline="", encoding="utf-8") as f:
        return {row["voice_note_id"]: row["file_path"] for row in csv.DictReader(f)}


def load_daily_notification_summary(path: Path) -> dict[tuple[str, str], DailyNotificationSummary]:
    """Load daily_notification_summary.csv into a dict keyed by (user_id, date)."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (row["user_id"], row["date"]): DailyNotificationSummary(
                user_id=row["user_id"],
                date=row["date"],
                notifications_sent=_parse_int(row["notifications_sent"]),
                notifications_dismissed=_parse_int(row["notifications_dismissed"]),
            )
            for row in csv.DictReader(f)
        }


def load_dataset(dataset_dir: Path) -> Dataset:
    """Load every dataset/*.csv file under dataset_dir and return a joined Dataset."""
    return Dataset(
        messages=load_messages(dataset_dir / "messages.csv"),
        users=load_users(dataset_dir / "users.csv"),
        groups=load_groups(dataset_dir / "groups.csv"),
        group_members=load_group_members(dataset_dir / "group_members.csv"),
        businesses=load_business_accounts(dataset_dir / "business_accounts.csv"),
        user_business_history=load_user_business_history(dataset_dir / "user_business_history.csv"),
        message_history=load_messages(dataset_dir / "message_history.csv"),
        message_events_by_message_id=load_message_events(dataset_dir / "message_events.csv"),
        images=load_images(dataset_dir / "images.csv"),
        voice_notes=load_voice_notes(dataset_dir / "voice_notes.csv"),
        daily_notification_summary=load_daily_notification_summary(
            dataset_dir / "daily_notification_summary.csv"
        ),
    )
