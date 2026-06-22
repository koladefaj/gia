"""Tests for the TTS provider and voice streaming utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.providers.tts import is_emotional, strip_audio_tags
from backend.app.voice.streaming import split_sentences, stream_sentences


class TestStripAudioTags:
    """Audio tags are delivery cues for ElevenLabs — never read aloud by Kokoro."""

    def test_removes_tags_and_tidies_spacing(self) -> None:
        assert strip_audio_tags("[warmly] Hey there [pause] friend") == "Hey there friend"

    def test_plain_text_unchanged(self) -> None:
        assert strip_audio_tags("Here's Free Mind by Tems.") == "Here's Free Mind by Tems."

    def test_all_tags_becomes_empty(self) -> None:
        assert strip_audio_tags("[laughs] [excited]") == ""


class TestIsEmotional:
    def test_audio_tag_is_emotional(self) -> None:
        assert is_emotional("[warmly] Hey, long week?")

    def test_question_is_emotional(self) -> None:
        assert is_emotional("Everything okay?")

    def test_logistics_not_emotional(self) -> None:
        assert not is_emotional("Adding to your library.")

    def test_empty_string_not_emotional(self) -> None:
        assert not is_emotional("")

    def test_laughs_tag_emotional(self) -> None:
        assert is_emotional("[laughs] That tweet is wild.")

    def test_plain_statement_not_emotional(self) -> None:
        assert not is_emotional("Here are four tracks for you.")


class TestSplitSentences:
    def test_splits_on_period(self) -> None:
        parts = split_sentences("Hello. How are you. I'm well.")
        assert len(parts) >= 2

    def test_splits_on_question_mark(self) -> None:
        parts = split_sentences("Everything okay? Just checking.")
        assert len(parts) == 2

    def test_preserves_audio_tags(self) -> None:
        parts = split_sentences("[warmly] Hey. Long week?")
        assert any("[warmly]" in p for p in parts)

    def test_empty_string(self) -> None:
        assert split_sentences("") == []

    def test_single_sentence_no_split(self) -> None:
        parts = split_sentences("Here's Free Mind by Tems")
        assert len(parts) == 1

    def test_filters_empty_parts(self) -> None:
        parts = split_sentences("Hello.   ")
        assert all(p.strip() for p in parts)


class TestStreamSentences:
    """Incremental reassembly of a token stream into whole sentences."""

    async def _collect(self, deltas: list[str]) -> list[str]:
        async def _src():
            for d in deltas:
                yield d
        return [s async for s in stream_sentences(_src())]

    @pytest.mark.asyncio
    async def test_emits_sentence_on_boundary(self) -> None:
        # Tokens split mid-word; a sentence flushes only once "." + space arrives.
        out = await self._collect(["Hey", " the", "re.", " How", " are you?", " Good."])
        assert out == ["Hey there.", "How are you?", "Good."]

    @pytest.mark.asyncio
    async def test_flushes_tail_without_terminal_punctuation(self) -> None:
        out = await self._collect(["Here's ", "Free Mind by Tems"])
        assert out == ["Here's Free Mind by Tems"]

    @pytest.mark.asyncio
    async def test_does_not_split_decimal_midstream(self) -> None:
        # "3.5" has no trailing space after the dot, so it must not flush early.
        out = await self._collect(["It's ", "3.5", " stars.", " Nice."])
        assert out == ["It's 3.5 stars.", "Nice."]

    @pytest.mark.asyncio
    async def test_pause_tag_is_a_boundary(self) -> None:
        out = await self._collect(["Hold on", "[pause]", "okay go."])
        assert out[0].endswith("[pause]")
        assert "okay go." in out[-1]

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self) -> None:
        assert await self._collect([]) == []


@pytest.mark.asyncio
async def test_synthesize_kokoro_returns_empty_when_not_installed() -> None:
    """Kokoro returns ``b""`` gracefully when not installed."""
    from backend.app.providers.tts import synthesize

    with patch("backend.app.providers.tts._get_kokoro_pipeline", return_value=None):
        result = await synthesize("Hello there.", provider="kokoro")

    assert result == b""


@pytest.mark.asyncio
async def test_synthesize_elevenlabs_returns_empty_without_key() -> None:
    """ElevenLabs returns ``b""`` when no API key is configured."""
    from backend.app.providers.tts import synthesize

    result = await synthesize("Hello there.", provider="elevenlabs", api_key="", voice_id="")
    assert result == b""


@pytest.mark.asyncio
async def test_synthesize_elevenlabs_calls_api() -> None:
    """ElevenLabs hits the API (via the pooled client) when a key is provided."""
    from backend.app.providers import tts

    mock_resp = MagicMock()
    mock_resp.content = b"fake-mp3-bytes"
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("backend.app.providers.tts._get_http_client", return_value=mock_client):
        result = await tts.synthesize(
            "Hello.", provider="elevenlabs", api_key="test-key", voice_id="voice-123"
        )

    assert result == b"fake-mp3-bytes"
    # The whole-file endpoint is used (no ``/stream`` suffix).
    assert mock_client.post.call_args.args[0].endswith("/voice-123")


@pytest.mark.asyncio
async def test_synthesize_stream_yields_chunks() -> None:
    """``synthesize_stream`` forwards the ElevenLabs ``/stream`` byte chunks."""
    from backend.app.providers import tts

    class _FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            for c in (b"mp3-part-1", b"mp3-part-2"):
                yield c

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=_FakeStream())

    with patch("backend.app.providers.tts._get_http_client", return_value=mock_client):
        chunks = [
            c async for c in tts.synthesize_stream(
                "Everything okay?", provider="elevenlabs", api_key="k", voice_id="v"
            )
        ]

    assert chunks == [b"mp3-part-1", b"mp3-part-2"]
    # Streaming hits the ``/stream`` endpoint.
    assert mock_client.stream.call_args.args[1].endswith("/v/stream")


@pytest.mark.asyncio
async def test_synthesize_stream_no_key_yields_nothing() -> None:
    """No API key → no audio chunks (graceful dev no-op)."""
    from backend.app.providers.tts import synthesize_stream

    chunks = [
        c async for c in synthesize_stream("Hello.", provider="elevenlabs", api_key="", voice_id="")
    ]
    assert chunks == []


@pytest.mark.asyncio
async def test_stream_tts_chunks_yields_nothing_for_empty_text() -> None:
    """Empty input produces no audio chunks."""
    from backend.app.voice.streaming import stream_tts_chunks

    chunks = [chunk async for chunk in stream_tts_chunks("")]
    assert chunks == []


@pytest.mark.asyncio
async def test_stream_tts_chunks_skips_empty_synthesis() -> None:
    """Sentences that synthesize to empty bytes are not yielded."""
    from backend.app.voice.streaming import stream_tts_chunks

    with patch("backend.app.voice.streaming.synthesize", new=AsyncMock(return_value=b"")):
        chunks = [chunk async for chunk in stream_tts_chunks("Hello. World.")]

    assert chunks == []


@pytest.mark.asyncio
async def test_stream_tts_chunks_yields_per_sentence() -> None:
    """One chunk per sentence when synthesis returns data."""
    from backend.app.voice.streaming import stream_tts_chunks

    call_count = 0

    async def fake_synthesize(text, **_kwargs):
        nonlocal call_count
        call_count += 1
        return b"audio-" + text.encode()[:4]

    with patch("backend.app.voice.streaming.synthesize", new=fake_synthesize):
        chunks = [chunk async for chunk in stream_tts_chunks("Hello. How are you.")]

    assert len(chunks) >= 1
    assert all(c.startswith(b"audio-") for c in chunks)
