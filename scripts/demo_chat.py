"""Standalone smoke test for the planner → weather → DJ/Artist → TTS pipeline.

This runs a few conversation turns **without** the database stack: it uses an
in-script fake Spotify client and skips long-term memory, so you can verify that
the whole spine works together before standing up the full app —

  * the prompt registry loads,
  * the planner classifies intent and detects the weather signal,
  * the weather tool returns real conditions (or degrades cleanly offline),
  * the DJ / Artist agents call the LLM (gemma3:4b via Ollama by default), and
  * the TTS provider (Kokoro locally) actually produces audio bytes.

Run (uses your .env / environment)::

    python scripts/demo_chat.py

To exercise the headline weather path you only need Ollama running:

    ollama pull gemma3:4b
    LLM_PROVIDER=ollama python scripts/demo_chat.py

The "I'm going for a run" turn is the demo line: it should show
``signals=['weather']``, a weather note, and an energetic recommendation.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make ``backend`` importable when run as `python scripts/demo_chat.py` from the
# repo root (the script's own dir is on sys.path, not the project root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.agents.artist import ArtistService  # noqa: E402
from backend.app.agents.dj import DJService
from backend.app.agents.planner import build_plan
from backend.app.config import settings
from backend.app.observability.logging import get_logger, setup_logging
from backend.app.providers.tts import synthesize
from backend.app.tools.brave import BraveSearchClient
from backend.app.tools.weather import MockWeatherClient, WeatherClient
from backend.app.voice.streaming import split_sentences

log = get_logger(__name__)

# A handful of Afrobeats tracks with audio features, mirroring the seed pool, so
# the DJ has real candidates to sequence without a live Spotify connection.
_DEMO_TRACKS = [
    {"uri": "spotify:track:002", "name": "Last Last", "artist": "Burna Boy",
     "energy": 0.78, "valence": 0.68, "tempo": 118.0, "danceability": 0.80, "key": 7, "mode": 1},
    {"uri": "spotify:track:004", "name": "Infinity", "artist": "Odumodublvck",
     "energy": 0.85, "valence": 0.55, "tempo": 142.0, "danceability": 0.72, "key": 9, "mode": 0},
    {"uri": "spotify:track:001", "name": "Free Mind", "artist": "Tems",
     "energy": 0.38, "valence": 0.71, "tempo": 92.0, "danceability": 0.62, "key": 5, "mode": 0},
    {"uri": "spotify:track:006", "name": "Calm Down", "artist": "Rema",
     "energy": 0.52, "valence": 0.82, "tempo": 105.0, "danceability": 0.78, "key": 6, "mode": 1},
]


class _DemoSpotify:
    """Minimal in-script ``SpotifyClientProtocol`` for the demo (no network)."""

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        return _DEMO_TRACKS[:limit]

    async def get_audio_features(self, uris: list[str]) -> list[dict]:
        by_uri = {t["uri"]: t for t in _DEMO_TRACKS}
        return [by_uri[u] for u in uris if u in by_uri]

    async def start_playback(self, uri: str, device_id: str | None = None) -> dict:
        return {"status": "playing", "uri": uri}


def _weather_note(current: dict | None) -> str | None:
    if not current:
        return None
    return (
        f"**Weather:** It's {current['temperature_c']:.0f}°C and "
        f"{current['condition']} in {settings.weather_default_label} right now."
    )


async def _run_turn(message: str, spotify: _DemoSpotify, weather, brave: BraveSearchClient) -> None:
    print("\n" + "=" * 70)
    print(f"YOU: {message}")

    plan = await build_plan(message, settings)
    print(f"PLAN: intent={plan.intent.value}  steps={plan.steps}  signals={plan.signals}")

    context = ""
    if "weather" in plan.signals:
        current = await weather.get_current(
            settings.weather_default_lat, settings.weather_default_lon
        )
        note = _weather_note(current)
        if note:
            context = note
            print(f"SIGNAL: {note}")

    reply = ""
    if "dj" in plan.steps:
        res = await DJService(spotify=spotify, cfg=settings).recommend(
            query=message, user_context_text=context, n=3
        )
        reply = res.recommendation
        queue = ", ".join(f"{t.name} ({t.camelot_key})" for t in res.queue.tracks)
        print(f"GIA (DJ): {reply}")
        print(f"QUEUE: {queue or 'none'}")
    if "artist" in plan.steps:
        res = await ArtistService(spotify=spotify, brave=brave, cfg=settings).get_info(message)
        reply = res.response
        print(f"GIA (Artist): {reply}")

    # Prove TTS produces audio for the first sentence of the reply.
    if reply:
        first = next(iter(split_sentences(reply)), reply)
        audio = await synthesize(
            first,
            provider=settings.tts_provider,
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
        )
        status = f"{len(audio)} bytes" if audio else "no audio (provider unavailable)"
        print(f"TTS ({settings.tts_provider}): {status}")


async def main() -> None:
    setup_logging(settings.log_level)
    print("Gia demo smoke test")
    print(f"  LLM      : {settings.llm_provider} / "
          f"{settings.ollama_model if settings.llm_provider == 'ollama' else '(provider default)'}")
    print(f"  TTS      : {settings.tts_provider}")
    print(f"  Weather  : {'live (Open-Meteo)' if settings.weather_enabled else 'mock'}")

    spotify = _DemoSpotify()
    weather = WeatherClient() if settings.weather_enabled else MockWeatherClient()
    brave = BraveSearchClient(api_key=settings.brave_api_key)

    turns = [
        "I'm going for a run, play me something",   # → weather-aware DJ (headline)
        "find me something chill for tonight",       # → plain DJ
        "tell me about Burna Boy",                   # → artist
    ]
    for message in turns:
        try:
            await _run_turn(message, spotify, weather, brave)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! turn failed: {exc}")

    print("\n" + "=" * 70)
    print("Done. If each turn printed a plan, a reply, and TTS bytes, the spine works.")


if __name__ == "__main__":
    asyncio.run(main())
