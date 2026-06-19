# Patches

## `spotify-mcp-server-getTopArtists.patch`

Fixes a crash in `marcelmarais/spotify-mcp-server`'s `getTopArtists` tool:
`Cannot read properties of undefined (reading 'length')`.

Spotify's **February 2026 API changes** stopped returning `genres` (and
`popularity`) on top-artist items, but the server assumed `artist.genres` always
exists. The patch guards `genres`/`popularity` against `undefined`.

The onboarding profiler (`POST /memory/{user_id}/bootstrap`) needs top artists,
so this fix is required for that feature to use real data.

### Apply

```bash
git clone https://github.com/marcelmarais/spotify-mcp-server.git
cd spotify-mcp-server
git apply /path/to/vexis/patches/spotify-mcp-server-getTopArtists.patch
npm install && npm run build
```

Then point the app at it via `SPOTIFY_MCP_SERVER_PATH=.../build/index.js`.

> Note: Spotify's Feb-2026 changes also **403 on playlist creation / track
> management** (`createPlaylist`, `addTracksToPlaylist`) — see upstream issues
> #35 and #62. Those are server-side Spotify restrictions, not fixable here.
