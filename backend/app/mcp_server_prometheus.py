"""
Prometheus MCP Server (stdio).

Exposes Prometheus HTTP API as MCP tools, similar to AWS Labs prometheus-mcp-server.
Configure via environment:
  PROMETHEUS_URL       - Prometheus server base URL (required)
  PROMETHEUS_BEARER_TOKEN - Optional Bearer token for auth

Run:
  export PROMETHEUS_URL=http://localhost:9090
  python -m app.mcp_server_prometheus

Or with uv from backend dir:
  uv run python -m app.mcp_server_prometheus
"""
from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

# Build FastMCP server and register tools
mcp = FastMCP(
    name="prometheus-mcp-server",
    instructions="Tools to query Prometheus. Use prometheus_query for instant queries and prometheus_query_range for time-range queries.",
)


def _get_config() -> tuple[str, str | None]:
    url = os.environ.get("PROMETHEUS_URL", "").strip().rstrip("/")
    token = os.environ.get("PROMETHEUS_BEARER_TOKEN", "").strip() or None
    return url, token


@mcp.tool(
    name="prometheus_query",
    description="Run a Prometheus instant query (single point in time). Returns JSON from Prometheus /api/v1/query.",
)
def prometheus_query(query: str, time_sec: float | None = None) -> dict:
    """
    Execute a Prometheus instant query.

    Args:
        query: PromQL expression.
        time_sec: Optional Unix timestamp in seconds; if omitted, uses current time.
    """
    from app.prometheus.client import prometheus_query as run_query

    base_url, bearer_token = _get_config()
    if not base_url or not base_url.startswith("http"):
        return {"error": "PROMETHEUS_URL is not set or invalid. Set it to your Prometheus base URL (e.g. http://localhost:9090)."}
    try:
        result = run_query(
            base_url,
            query,
            time_sec=time_sec,
            bearer_token=bearer_token,
        )
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool(
    name="prometheus_query_range",
    description="Run a Prometheus range query over a time window. Returns JSON from Prometheus /api/v1/query_range.",
)
def prometheus_query_range(
    query: str,
    start_sec: float,
    end_sec: float,
    step_sec: int = 60,
) -> dict:
    """
    Execute a Prometheus range query.

    Args:
        query: PromQL expression.
        start_sec: Start time as Unix timestamp in seconds.
        end_sec: End time as Unix timestamp in seconds.
        step_sec: Query resolution step width in seconds (default 60).
    """
    from app.prometheus.client import prometheus_query_range as run_query_range

    base_url, bearer_token = _get_config()
    if not base_url or not base_url.startswith("http"):
        return {"error": "PROMETHEUS_URL is not set or invalid. Set it to your Prometheus base URL (e.g. http://localhost:9090)."}
    if step_sec < 1:
        step_sec = 60
    try:
        result = run_query_range(
            base_url,
            query,
            start_sec=start_sec,
            end_sec=end_sec,
            step_sec=step_sec,
            bearer_token=bearer_token,
        )
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


def main() -> None:
    base_url, _ = _get_config()
    if not base_url or not base_url.startswith("http"):
        print("Error: PROMETHEUS_URL must be set to a valid Prometheus base URL (e.g. http://localhost:9090)", file=sys.stderr)
        sys.exit(1)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
