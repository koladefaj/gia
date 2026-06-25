# Patches

## `spotify-mcp-server.patch`

Two fixes for `marcelmarais/spotify-mcp-server` needed for Gia:

1. **`getTopArtists` crash** (`src/read.ts`) — `Cannot read properties of
   undefined (reading 'length')`. Spotify's **February 2026 API changes** stopped
   returning `genres`/`popularity` on top-artist items, but the server assumed
   `artist.genres` always exists. The patch guards against `undefined`. Required
   for the onboarding profiler (`POST /memory/{user_id}/bootstrap`).

2. **stdout corruption on token refresh** (`src/utils.ts`) — the server logged
   "Access token refreshing…" to **stdout**, but for a stdio MCP server stdout
   *is* the JSON-RPC channel, so that log corrupted the protocol stream and broke
   the client mid-session (~hourly, on token refresh). The patch routes those
   logs to **stderr**. (The app's bridge also auto-reconnects as defense in depth.)

### Apply

```bash
git clone https://github.com/marcelmarais/spotify-mcp-server.git
cd spotify-mcp-server
git apply /path/to/vexis/patches/spotify-mcp-server.patch
npm install && npm run build
```

Then point the app at it via `SPOTIFY_MCP_SERVER_PATH=.../build/index.js`.

> Note: Spotify's Feb-2026 changes also **403 on playlist creation / track
> management** (`createPlaylist`, `addTracksToPlaylist`) — upstream issues #35
> and #62. Those are server-side Spotify restrictions (Development-Mode apps),
> not fixable here.
