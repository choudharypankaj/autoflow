# TiDB MCP Server (pytidb)

This repository includes wiring to run the TiDB MCP server (from `pytidb`) for local development and integration with MCP-compatible hosts (e.g., Claude Desktop, Cursor).

---

## Prometheus MCP Server

A Prometheus MCP server (similar to [AWS Labs prometheus-mcp-server](https://github.com/awslabs/mcp/tree/main/src/prometheus-mcp-server)) is included in the backend. It exposes Prometheus HTTP API as MCP tools over stdio.

### Tools

- **prometheus_query** – Instant query (Prometheus `/api/v1/query`). Parameters: `query` (PromQL), optional `time_sec` (Unix timestamp).
- **prometheus_query_range** – Range query (Prometheus `/api/v1/query_range`). Parameters: `query`, `start_sec`, `end_sec`, `step_sec` (default 60).

### Prerequisites

- Prometheus server reachable at a URL (e.g. `http://localhost:9090`)
- Optional: `PROMETHEUS_BEARER_TOKEN` for Bearer auth

### Run (stdio)

```bash
export PROMETHEUS_URL=http://localhost:9090
# optional: export PROMETHEUS_BEARER_TOKEN=your_token

make -C backend dev_prometheus_mcp_server
```

Or directly:

```bash
cd backend
export PROMETHEUS_URL=http://localhost:9090
uv run python -m app.mcp_server_prometheus
```

### Claude Desktop (example)

```json
{
  "mcpServers": {
    "prometheus": {
      "command": "uv",
      "args": ["run", "-m", "app.mcp_server_prometheus"],
      "env": {
        "PROMETHEUS_URL": "http://localhost:9090"
      }
    }
  }
}
```

### Integration with autoflow

The autoflow backend uses the Prometheus **HTTP API** directly (no MCP server required) for slow-query metrics. The Prometheus MCP server is provided for use with MCP hosts (e.g. Claude Desktop, Cursor) that want to query Prometheus via tools.


---

## TiDB MCP Server (pytidb)

### Prerequisites

- Python 3.11+ with `uv` installed (the backend image already uses `uv`)
- A TiDB cluster (host, port, username, password, database)

## Install

Install the MCP extra for `pytidb` in your environment (recommended to use `uv`):

```bash
uv add pytidb --extra mcp
```

Alternatively with `pip`:

```bash
pip install "pytidb[mcp]"
```

## Run (stdio)

Export DB credentials and start the MCP server:

```bash
export TIDB_HOST=127.0.0.1
export TIDB_PORT=4000
export TIDB_USERNAME=root
export TIDB_PASSWORD=...
export TIDB_DATABASE=your_db

make -C backend dev_mcp_server
```

This runs `pytidb.ext.mcp` in stdio mode (suitable for MCP hosts that launch the process).

## Configure MCP Host (UI)

In the Admin UI: Site Settings → Integrations → MCP

- Set multiple DB agents in `MCP Hosts`, e.g.:

```json
[
  { "text": "tidb-prod", "href": "wss://prod-mcp.example.com/ws" },
  { "text": "tidb-staging", "href": "wss://staging-mcp.example.com/ws" },
  { "text": "local", "href": "ws://localhost:8080/ws" }
]
```

From the chat UI, select the agent in the dropdown, then ask slow-query questions with a UTC window.

## Claude Desktop (example)

Open Settings → Developer → Edit Config and add:

```json
{
  "mcpServers": {
    "tidb": {
      "command": "uv",
      "args": ["run", "-m", "pytidb.ext.mcp"],
      "env": {
        "TIDB_HOST": "127.0.0.1",
        "TIDB_PORT": "4000",
        "TIDB_USERNAME": "root",
        "TIDB_PASSWORD": "....",
        "TIDB_DATABASE": "your_db"
      }
    }
  }
}
```

## Notes

- The backend supports selecting agents by name via the chat UI; the selected agent is passed to the server for MCP calls.
- For WebSocket-hosted MCP servers, ensure your bridge exposes a `ws://` or `wss://` URL that the backend can reach.

