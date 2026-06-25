"""Drop deprecated Spotify audio feature columns from listening_events.

Spotify removed audio features (energy, valence, tempo, danceability, key, mode)
from the Web API for new apps. Mood is now LLM-labeled from track/artist names.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("listening_events", "energy")
    op.drop_column("listening_events", "valence")
    op.drop_column("listening_events", "tempo")
    op.drop_column("listening_events", "danceability")
    op.drop_column("listening_events", "key")
    op.drop_column("listening_events", "mode")


def downgrade() -> None:
    op.add_column("listening_events", sa.Column("mode", sa.Integer(), nullable=True))
    op.add_column("listening_events", sa.Column("key", sa.Integer(), nullable=True))
    op.add_column("listening_events", sa.Column("danceability", sa.Float(), nullable=True))
    op.add_column("listening_events", sa.Column("tempo", sa.Float(), nullable=True))
    op.add_column("listening_events", sa.Column("valence", sa.Float(), nullable=True))
    op.add_column("listening_events", sa.Column("energy", sa.Float(), nullable=True))
