"""Tests for the Weather tool (Open-Meteo client + mock)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.app.interfaces import WeatherClientProtocol
from backend.app.tools.weather import (
    MockWeatherClient,
    WeatherClient,
    condition_for_code,
)


def test_mock_satisfies_protocol() -> None:
    assert isinstance(MockWeatherClient(), WeatherClientProtocol)
    assert isinstance(WeatherClient(), WeatherClientProtocol)


def test_condition_mapping() -> None:
    assert condition_for_code(0) == "clear"
    assert condition_for_code(95) == "thunderstorm"
    assert condition_for_code(999) == "unsettled"  # unknown code → fallback


@pytest.mark.asyncio
async def test_mock_returns_fixture() -> None:
    out = await MockWeatherClient(temperature_c=31.0, condition="clear").get_current(0, 0)
    assert out == {"temperature_c": 31.0, "condition": "clear", "wind_kph": 8.0}


@pytest.mark.asyncio
async def test_weather_client_parses_open_meteo_payload() -> None:
    payload = {
        "current": {
            "temperature_2m": 31.4,
            "weather_code": 2,
            "wind_speed_10m": 12.0,
        }
    }
    fake_resp = MagicMock()
    fake_resp.json.return_value = payload
    fake_resp.raise_for_status = MagicMock()

    fake_http = AsyncMock()
    fake_http.get = AsyncMock(return_value=fake_resp)
    fake_http.__aenter__ = AsyncMock(return_value=fake_http)
    fake_http.__aexit__ = AsyncMock(return_value=None)

    with patch.object(httpx, "AsyncClient", return_value=fake_http):
        out = await WeatherClient().get_current(6.5, 3.3)

    assert out == {"temperature_c": 31.4, "condition": "partly cloudy", "wind_kph": 12.0}


@pytest.mark.asyncio
async def test_weather_client_returns_none_on_error() -> None:
    fake_http = AsyncMock()
    fake_http.get = AsyncMock(side_effect=httpx.ConnectError("down"))
    fake_http.__aenter__ = AsyncMock(return_value=fake_http)
    fake_http.__aexit__ = AsyncMock(return_value=None)

    with patch.object(httpx, "AsyncClient", return_value=fake_http):
        out = await WeatherClient().get_current(6.5, 3.3)

    assert out is None


@pytest.mark.asyncio
async def test_weather_client_returns_none_on_bad_payload() -> None:
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"current": {}}  # missing temperature_2m
    fake_resp.raise_for_status = MagicMock()

    fake_http = AsyncMock()
    fake_http.get = AsyncMock(return_value=fake_resp)
    fake_http.__aenter__ = AsyncMock(return_value=fake_http)
    fake_http.__aexit__ = AsyncMock(return_value=None)

    with patch.object(httpx, "AsyncClient", return_value=fake_http):
        out = await WeatherClient().get_current(6.5, 3.3)

    assert out is None
