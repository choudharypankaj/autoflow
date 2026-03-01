"""
Prometheus HTTP API client for flexible query execution.
Uses /api/v1/query and /api/v1/query_range; no MCP server required.
"""
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _headers(bearer_token: str | None) -> dict[str, str]:
    if not bearer_token or not bearer_token.strip():
        return {}
    return {"Authorization": f"Bearer {bearer_token.strip()}"}


def prometheus_query(
    base_url: str,
    query: str,
    *,
    time_sec: float | None = None,
    timeout: int = 30,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    """
    Run a Prometheus instant query (GET /api/v1/query).
    """
    base = base_url.strip().rstrip("/")
    url = f"{base}/api/v1/query"
    params: dict[str, Any] = {"query": query}
    if time_sec is not None:
        params["time"] = time_sec
    try:
        resp = requests.get(
            url,
            params=params,
            headers=_headers(bearer_token),
            timeout=timeout,
        )
    except Exception as e:
        logger.exception("Prometheus query failed: url=%s query=%s error=%s", url, query[:100], e)
        raise RuntimeError(f"Prometheus query failed: {e}") from e
    if resp.status_code >= 400:
        logger.error("Prometheus query error: status=%s body=%s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Prometheus API error: {resp.status_code} {resp.text}")
    return resp.json()


def prometheus_query_range(
    base_url: str,
    query: str,
    *,
    start_sec: float,
    end_sec: float,
    step_sec: int | None = None,
    timeout: int = 30,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    """
    Run a Prometheus range query (GET /api/v1/query_range).
    """
    base = base_url.strip().rstrip("/")
    url = f"{base}/api/v1/query_range"
    step = step_sec if step_sec is not None and step_sec >= 1 else 60
    params: dict[str, Any] = {
        "query": query,
        "start": start_sec,
        "end": end_sec,
        "step": step,
    }
    try:
        resp = requests.get(
            url,
            params=params,
            headers=_headers(bearer_token),
            timeout=timeout,
        )
    except Exception as e:
        logger.exception(
            "Prometheus query_range failed: url=%s query=%s start=%s end=%s error=%s",
            url,
            query[:100],
            start_sec,
            end_sec,
            e,
        )
        raise RuntimeError(f"Prometheus query_range failed: {e}") from e
    if resp.status_code >= 400:
        logger.error(
            "Prometheus query_range error: status=%s body=%s",
            resp.status_code,
            resp.text[:500],
        )
        raise RuntimeError(f"Prometheus API error: {resp.status_code} {resp.text}")
    return resp.json()
