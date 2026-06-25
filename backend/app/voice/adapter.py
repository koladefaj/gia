"""Voice adapter — abstract tone → provider-specific delivery tags.

The router speaks in abstract tones (``surprised``, ``warm``); only this layer
knows how a given TTS provider expresses them.  That separation means we can
swap ElevenLabs / OpenAI Voice / Cartesia / Kokoro without touching router or
agent logic — the router must NEVER emit ``[light laugh]`` itself.

Tags are prepended to a line, e.g. ``apply("surprised", "Wait, for real?")`` →
``"[surprised] Wait, for real?"``.  Kokoro strips them (see ``tts.strip_audio_tags``);
ElevenLabs v3 interprets them as delivery cues.
"""

from __future__ import annotations

from backend.app.schemas.router import Tone

# Tone → ElevenLabs-style tag. Reuses the persona's existing tag vocabulary
# where one fits, so emotional-routing (is_emotional) stays consistent.
_TONE_TAGS: dict[str, str] = {
    Tone.CURIOUS.value: "[curious]",
    Tone.SURPRISED.value: "[surprised]",
    Tone.WARM.value: "[warmly]",
    Tone.PLAYFUL.value: "[light laugh]",
    Tone.THOUGHTFUL.value: "[thoughtful]",
    Tone.EXCITED.value: "[excited]",
    Tone.EMPATHETIC.value: "[gentle]",
    Tone.CONFIDENT.value: "[confident]",
}


class VoiceAdapter:
    """Maps abstract tones to provider voice tags behind a stable interface."""

    def convert_tone_to_tags(self, tone: str) -> str:
        """Return the provider tag for *tone*, or ``""`` if it has no mapping."""
        return _TONE_TAGS.get(tone, "")

    def apply(self, tone: str, text: str) -> str:
        """Prepend the tone's tag to *text* (no-op when the tone has no tag)."""
        tag = self.convert_tone_to_tags(tone)
        return f"{tag} {text}".strip() if tag else text
