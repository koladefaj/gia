from __future__ import annotations

from pydantic import BaseModel, Field


class CreatePlaylistRequest(BaseModel):
    """Body for ``POST /playlist``."""

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=300)
    track_uris: list[str] = Field(default_factory=list)
