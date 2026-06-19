"""Tests for the DI dependency providers in ``backend.app.dependencies``.

Confirms that each provider returns the correct type and that ``get_settings``
returns the singleton, not a fresh object each call.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.app.config import settings as _global_settings
from backend.app.dependencies import get_settings
from backend.app.interfaces import SpotifyClientProtocol


def test_get_settings_returns_singleton() -> None:
    """``get_settings()`` always returns the same ``Settings`` object."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert s1 is _global_settings


def test_get_redis_reads_app_state() -> None:
    """``get_redis`` returns whatever is stored on ``request.app.state.redis``."""
    from backend.app.dependencies import get_redis

    mock_redis = MagicMock()
    mock_request = MagicMock()
    mock_request.app.state.redis = mock_redis

    result = get_redis(mock_request)
    assert result is mock_redis


def test_get_spotify_client_reads_app_state() -> None:
    """``get_spotify_client`` returns whatever is stored on ``request.app.state.spotify``."""
    from backend.app.dependencies import get_spotify_client
    from tests.conftest import FakeSpotifyClient

    fake = FakeSpotifyClient()
    mock_request = MagicMock()
    mock_request.app.state.spotify = fake

    result = get_spotify_client(mock_request)
    assert result is fake
    assert isinstance(result, SpotifyClientProtocol)


def test_get_spotify_raises_if_state_missing() -> None:
    """``get_spotify_client`` raises ``AttributeError`` before startup sets state."""
    from backend.app.dependencies import get_spotify_client

    mock_request = MagicMock(spec=[])  # no attributes
    with pytest.raises(AttributeError):
        get_spotify_client(mock_request)
