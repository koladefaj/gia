"""OAuth callback diagnostic — isolates the /auth/spotify/callback failure.

Runs entirely from the host against the published ports (api :8000, redis :6379),
so it bypasses the browser (and its cookies / long URLs).

What it checks:
  1. Which ``redirect_uri`` ``/login`` actually sends to Spotify — confirms we
     point at :8000/auth/spotify/callback and not the MCP server's :8888.
  2. The callback handler end-to-end: it seeds a valid PKCE state in the SAME
     Redis the API reads, then calls the callback with a fake code. Spotify will
     reject the fake code at token-exchange — which is fine: we only want to see
     the handler *run* and log its path.

Run:
    python scripts/test_oauth_callback.py

Interpreting the result:
  * status 400 "Token exchange with Spotify failed" → the handler works; the
    browser failure is the request itself (oversized headers/cookies on
    127.0.0.1). Fix: clear site data for 127.0.0.1 or use an incognito window.
  * status 500 → the handler is broken before token exchange (or the running
    container still has old code). Check `docker compose logs api`.
  * connection refused → the api/redis ports aren't published as expected.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

import httpx
import redis.asyncio as aioredis

API = "http://127.0.0.1:8000"
REDIS_URL = "redis://localhost:6379/0"
STATE = "diag-callback-test"


async def check_login_redirect() -> None:
    """Print the redirect_uri /login hands to Spotify."""
    async with httpx.AsyncClient(follow_redirects=False) as c:
        resp = await c.get(f"{API}/auth/spotify/login")
    loc = resp.headers.get("location", "")
    print(f"[1] /login → {resp.status_code}")
    if not loc:
        print("    no Location header — /login did not redirect. Body:")
        print("   ", resp.text[:300])
        return
    qs = parse_qs(urlparse(loc).query)
    print(f"    authorize host: {urlparse(loc).netloc}")
    print(f"    redirect_uri  : {qs.get('redirect_uri', ['<missing>'])[0]}")
    print(f"    client_id     : {qs.get('client_id', ['<missing>'])[0]}")
    print(f"    scopes        : {qs.get('scope', ['<missing>'])[0][:60]}…")


async def check_callback() -> None:
    """Seed a PKCE state, then call the callback with a fake code."""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    await r.setex(
        f"pkce:state:{STATE}",
        600,
        json.dumps({"verifier": "diag-verifier", "user_id": None}),
    )
    await r.aclose()
    print(f"[2] seeded redis state {STATE!r}")

    async with httpx.AsyncClient(follow_redirects=False) as c:
        resp = await c.get(
            f"{API}/auth/spotify/callback",
            params={"state": STATE, "code": "fake-code-for-diagnostics"},
        )
    print(f"    callback → {resp.status_code}")
    print(f"    location: {resp.headers.get('location', '<none>')}")
    print(f"    body    : {resp.text[:400]}")


async def main() -> None:
    try:
        await check_login_redirect()
    except Exception as exc:  # noqa: BLE001
        print(f"[1] FAILED: {type(exc).__name__}: {exc}")
    print()
    try:
        await check_callback()
    except Exception as exc:  # noqa: BLE001
        print(f"[2] FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
