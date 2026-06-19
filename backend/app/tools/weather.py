"""Weather client — current conditions via the Open-Meteo public API.

Open-Meteo is free and requires no API key, which makes it an ideal real-world
signal for a portfolio project: zero credential friction, generous limits.  The
planner uses it to make music recommendations context-aware — energetic picks
for a hot afternoon run, a longer queue for a rainy commute.

Every call goes through :func:`resilient_call` (timeout + retry + a shared
circuit breaker) so a slow or down weather endpoint fails fast and degrades to
"no weather context" rather than stalling the conversation turn.
"""

from __future__ import annotations

import httpx

from backend.app.observability.logging import get_logger
from backend.app.tools.resilience import CircuitBreaker, resilient_call

logger = get_logger(__name__)

_OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# Shared breaker — WeatherClient is cheap to construct, so the breaker lives at
# module level to track endpoint health across instances/requests.
_WEATHER_BREAKER = CircuitBreaker("weather", threshold=5, cooldown=30.0)

# WMO weather interpretation codes → short human conditions.
# https://open-meteo.com/en/docs (WMO Weather interpretation codes)
_WMO_CONDITIONS: dict[int, str] = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def condition_for_code(code: int) -> str:
    """Map a WMO weather code to a short human-readable condition string."""
    return _WMO_CONDITIONS.get(code, "unsettled")


class WeatherClient:
    """Async client for current weather from Open-Meteo (no API key needed)."""

    async def get_current(self, latitude: float, longitude: float) -> dict | None:
        """Return current weather at *(latitude, longitude)*.

        Args:
            latitude:  Decimal degrees.
            longitude: Decimal degrees.

        Returns:
            ``{"temperature_c", "condition", "wind_kph"}`` or ``None`` if the
            lookup failed (network error, breaker open, or unexpected payload).
        """

        async def _do() -> dict:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    _OPEN_METEO_BASE,
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                        "wind_speed_unit": "kmh",
                    },
                )
                resp.raise_for_status()
                payload: dict = resp.json()
                return payload

        try:
            data = await resilient_call(
                _do, name="weather.get_current", timeout_s=10.0, retries=1,
                breaker=_WEATHER_BREAKER,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("weather_lookup_failed", error=str(exc))
            return None

        current = data.get("current") or {}
        if "temperature_2m" not in current:
            logger.warning("weather_payload_unexpected", keys=list(current))
            return None

        result = {
            "temperature_c": float(current["temperature_2m"]),
            "condition": condition_for_code(int(current.get("weather_code", -1))),
            "wind_kph": float(current.get("wind_speed_10m", 0.0)),
        }
        logger.info("weather_lookup_done", **result)
        return result


class MockWeatherClient:
    """Deterministic ``WeatherClientProtocol`` implementation for tests/offline.

    Returns a fixed warm-and-clear reading so weather-aware behaviour can be
    exercised without any network access.
    """

    def __init__(self, temperature_c: float = 27.0, condition: str = "clear") -> None:
        self._temperature_c = temperature_c
        self._condition = condition

    async def get_current(self, latitude: float, longitude: float) -> dict | None:
        """Return the fixed fixture reading regardless of coordinates."""
        return {
            "temperature_c": self._temperature_c,
            "condition": self._condition,
            "wind_kph": 8.0,
        }
