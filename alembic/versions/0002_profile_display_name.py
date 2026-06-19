"""Add profiles.display_name — the name Gia calls the user.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-20
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("profiles", sa.Column("display_name", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("profiles", "display_name")
