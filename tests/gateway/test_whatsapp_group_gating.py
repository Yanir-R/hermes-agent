import asyncio
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(require_mention=None, mention_patterns=None, free_response_chats=None,
                  dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None,
                  selective_response_chats=None, sender_authorized=True):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if free_response_chats is not None:
        extra["free_response_chats"] = free_response_chats
    if dm_policy is not None:
        extra["dm_policy"] = dm_policy
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_policy is not None:
        extra["group_policy"] = group_policy
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from
    if selective_response_chats is not None:
        extra["selective_response_chats"] = selective_response_chats

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._authorization_check = lambda *_args: sender_authorized
    adapter._dm_policy = str(extra.get("dm_policy", "pairing")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "pairing")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("group_allow_from"))
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._free_response_chats = adapter._whatsapp_free_response_chats()
    return adapter


def _group_message(body="hello", **overrides):
    data = {
        "isGroup": True,
        "body": body,
        "chatId": "120363001234567890@g.us",
        "mentionedIds": [],
        "botIds": ["15551230000@s.whatsapp.net", "15551230000@lid"],
        "quotedParticipant": "",
    }
    data.update(overrides)
    return data


def _dm_message(body="hello", **overrides):
    data = {
        "isGroup": False,
        "body": body,
        "senderId": "6281234567890@s.whatsapp.net",
        "from": "6281234567890@s.whatsapp.net",
        "botIds": [],
        "mentionedIds": [],
    }
    data.update(overrides)
    return data


# --- Existing tests (unchanged logic, updated helper) ---


def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True, group_policy="open")

    assert adapter._should_process_message(_group_message("hello everyone")) is False
    assert adapter._should_process_message(
        _group_message(
            "hi there",
            mentionedIds=["15551230000@s.whatsapp.net"],
        )
    ) is True
    assert adapter._should_process_message(
        _group_message(
            "replying",
            quotedParticipant="15551230000@lid",
        )
    ) is True
    assert adapter._should_process_message(_group_message("/status")) is True


def test_regex_mention_patterns_allow_custom_wake_words():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("   chompy help")) is True
    assert adapter._should_process_message(_group_message("hey chompy")) is False


def test_invalid_regex_patterns_are_ignored():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"(", r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_free_response_chats_bypass_mention_gating():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["120363001234567890@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_free_response_chats_does_not_bypass_other_groups():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["999999999999@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_selective_response_group_bypasses_mention_gate_with_scoped_metadata():
    target = "120363001234567890@g.us"
    key = {
        "id": "incoming-1",
        "remoteJid": target,
        "participant": "15550001111@s.whatsapp.net",
        "fromMe": False,
    }
    adapter = _make_adapter(
        require_mention=True,
        group_policy="allowlist",
        group_allow_from=[target],
        selective_response_chats=[target],
    )

    event = asyncio.run(
        adapter._build_message_event(
            _group_message(
                "ordinary group chatter",
                senderId="15550001111@s.whatsapp.net",
                readReceiptKey=key,
            )
        )
    )

    assert event is not None
    assert event.metadata["whatsapp_selective_response"] is True
    assert event.metadata["whatsapp_reaction_keys"] == [key]
    assert "output exactly NO_REPLY" in event.channel_prompt


def test_selective_response_does_not_open_other_groups_or_dms():
    target = "120363001234567890@g.us"
    adapter = _make_adapter(
        require_mention=True,
        dm_policy="disabled",
        group_policy="open",
        selective_response_chats=[target],
    )

    other_group = _group_message(
        "ordinary group chatter",
        chatId="999999999999999999@g.us",
        senderId="15550001111@s.whatsapp.net",
    )
    matching_id_but_dm = _dm_message(
        "private message",
        chatId=target,
        senderId=target,
    )

    assert asyncio.run(adapter._build_message_event(other_group)) is None
    assert asyncio.run(adapter._build_message_event(matching_id_but_dm)) is None


def test_selective_response_fails_closed_for_denied_group_or_sender():
    target = "120363001234567890@g.us"
    denied_group = _make_adapter(
        require_mention=True,
        group_policy="allowlist",
        group_allow_from=["999999999999999999@g.us"],
        selective_response_chats=[target],
    )
    denied_sender = _make_adapter(
        require_mention=True,
        group_policy="allowlist",
        group_allow_from=[target],
        selective_response_chats=[target],
        sender_authorized=False,
    )
    message = _group_message(
        "ordinary group chatter",
        senderId="15550001111@s.whatsapp.net",
    )

    assert asyncio.run(denied_group._build_message_event(message)) is None
    assert asyncio.run(denied_sender._build_message_event(message)) is None


def test_selective_silence_reacts_only_on_its_own_successful_turn():
    adapter = _make_adapter()
    adapter._send_reaction_key = AsyncMock(return_value=True)
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="120363001234567890@g.us",
        chat_type="group",
        user_id="15550001111@s.whatsapp.net",
    )
    key = {
        "id": "incoming-1",
        "remoteJid": source.chat_id,
        "participant": source.user_id,
        "fromMe": False,
    }
    selective = MessageEvent(
        text="chatter",
        message_type=MessageType.TEXT,
        source=source,
        metadata={
            "whatsapp_selective_response": True,
            "gateway_intentional_silence": True,
            "whatsapp_reaction_keys": [key],
        },
    )

    asyncio.run(adapter.on_processing_complete(selective, ProcessingOutcome.SUCCESS))
    adapter._send_reaction_key.assert_awaited_once_with(key, "👀")

    adapter._send_reaction_key.reset_mock()
    ordinary = MessageEvent(text="dm", source=source, metadata={})
    asyncio.run(adapter.on_processing_complete(ordinary, ProcessingOutcome.SUCCESS))
    asyncio.run(adapter.on_processing_complete(selective, ProcessingOutcome.FAILURE))
    adapter._send_reaction_key.assert_not_awaited()


def test_selective_group_debounce_keeps_different_senders_separate():
    adapter = _make_adapter()
    adapter.config.extra["group_sessions_per_user"] = False
    common = {
        "platform": Platform.WHATSAPP,
        "chat_id": "120363001234567890@g.us",
        "chat_type": "group",
    }
    first = MessageEvent(
        text="first",
        source=SessionSource(user_id="111@s.whatsapp.net", **common),
        metadata={"whatsapp_selective_response": True},
    )
    second = MessageEvent(
        text="second",
        source=SessionSource(user_id="222@s.whatsapp.net", **common),
        metadata={"whatsapp_selective_response": True},
    )

    assert adapter._text_batch_key(first) != adapter._text_batch_key(second)


def test_mention_stripping_removes_bot_phone_from_body():
    adapter = _make_adapter(require_mention=True)

    data = _group_message("@15551230000 what is the weather?")
    cleaned = adapter._clean_bot_mention_text(data["body"], data)
    assert "15551230000" not in cleaned
    assert "weather" in cleaned


# --- New dm_policy tests ---


def test_dm_policy_disabled_still_allows_groups():
    adapter = _make_adapter(
        dm_policy="disabled",
        require_mention=False,
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello")) is True


# --- New group_policy tests ---


# --- Config bridging tests ---

def test_config_bridges_whatsapp_dm_and_group_policy(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  dm_policy: disabled\n"
        "  group_policy: allowlist\n"
        "  group_allow_from:\n"
        "    - \"120363001234567890@g.us\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_DM_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert config.platforms[Platform.WHATSAPP].extra["dm_policy"] == "disabled"
    assert config.platforms[Platform.WHATSAPP].extra["group_policy"] == "allowlist"
    assert config.platforms[Platform.WHATSAPP].extra["group_allow_from"] == ["120363001234567890@g.us"]
    assert __import__("os").environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert __import__("os").environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert __import__("os").environ["WHATSAPP_GROUP_ALLOWED_USERS"] == "120363001234567890@g.us"


def test_config_bridges_selective_response_chats_to_adapter_extra(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    target = "120363001234567890@g.us"
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  selective_response_chats:\n"
        f"    - \"{target}\"\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")

    config = load_gateway_config()

    assert config.platforms[Platform.WHATSAPP].extra["selective_response_chats"] == [target]


# --- Broadcast / status / newsletter pseudo-chats are always dropped ---


def test_status_broadcast_chats_are_always_dropped():
    """Felipe's gateway.log showed the agent replying to status@broadcast
    (a contact's WhatsApp Story update). These pseudo-chats aren't real
    conversations and the adapter must drop them regardless of dm_policy.
    """

    # Even on the most permissive config — open DMs, no allowlist — Stories
    # and Channel posts must not reach the agent.
    adapter = _make_adapter(dm_policy="open")

    # Classic Story update — what Felipe was seeing in production.
    status_msg = _dm_message(
        body="[video received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(status_msg) is False

    # Channel / Newsletter broadcast posts.
    newsletter_msg = _dm_message(
        body="check out our latest post",
        chatId="120363999999999999@newsletter",
        senderId="120363999999999999@newsletter",
    )
    assert adapter._should_process_message(newsletter_msg) is False


def test_broadcast_filter_runs_before_allowlist():
    """A status@broadcast message from an allowlisted sender still drops —
    we never want to reply to Stories, even from authorized contacts.
    """
    adapter = _make_adapter(
        dm_policy="allowlist",
        allow_from=["34612345678@s.whatsapp.net"],
    )

    msg = _dm_message(
        body="[image received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(msg) is False
