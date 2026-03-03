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


def prometheus_metric_names(
    base_url: str,
    *,
    timeout: int = 30,
    bearer_token: str | None = None,
) -> list[str]:
    """Fetch all metric names from Prometheus (GET /api/v1/label/__name__/values)."""
    base = base_url.strip().rstrip("/")
    url = f"{base}/api/v1/label/__name__/values"
    try:
        resp = requests.get(url, headers=_headers(bearer_token), timeout=timeout)
    except Exception as e:
        logger.exception("Prometheus metric names failed: url=%s error=%s", url, e)
        raise RuntimeError(f"Prometheus metric names failed: {e}") from e
    if resp.status_code >= 400:
        logger.error("Prometheus metric names error: status=%s body=%s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Prometheus API error: {resp.status_code} {resp.text}")
    data = resp.json()
    vals = data.get("data") if isinstance(data, dict) else None
    if not isinstance(vals, list):
        return []
    return [str(v).strip() for v in vals if v and str(v).strip()]


def prometheus_labels(
    base_url: str,
    *,
    timeout: int = 30,
    bearer_token: str | None = None,
) -> list[str]:
    """Fetch all label names from Prometheus (GET /api/v1/labels)."""
    base = base_url.strip().rstrip("/")
    url = f"{base}/api/v1/labels"
    try:
        resp = requests.get(url, headers=_headers(bearer_token), timeout=timeout)
    except Exception as e:
        logger.exception("Prometheus labels failed: url=%s error=%s", url, e)
        raise RuntimeError(f"Prometheus labels failed: {e}") from e
    if resp.status_code >= 400:
        logger.error("Prometheus labels error: status=%s body=%s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Prometheus API error: {resp.status_code} {resp.text}")
    data = resp.json()
    vals = data.get("data") if isinstance(data, dict) else None
    if not isinstance(vals, list):
        return []
    return [str(v).strip() for v in vals if v and str(v).strip()]


def prometheus_label_values(
    base_url: str,
    label_name: str,
    *,
    timeout: int = 30,
    bearer_token: str | None = None,
) -> list[str]:
    """Fetch all values for a label (GET /api/v1/label/<name>/values)."""
    base = base_url.strip().rstrip("/")
    name = label_name.strip()
    url = f"{base}/api/v1/label/{name}/values"
    try:
        resp = requests.get(url, headers=_headers(bearer_token), timeout=timeout)
    except Exception as e:
        logger.exception("Prometheus label values failed: url=%s label=%s error=%s", url, name, e)
        raise RuntimeError(f"Prometheus label values failed: {e}") from e
    if resp.status_code >= 400:
        logger.error("Prometheus label values error: status=%s body=%s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Prometheus API error: {resp.status_code} {resp.text}")
    data = resp.json()
    vals = data.get("data") if isinstance(data, dict) else None
    if not isinstance(vals, list):
        return []
    return [str(v).strip() for v in vals if v is not None and str(v).strip()]


def prometheus_metadata(
    base_url: str,
    *,
    timeout: int = 30,
    bearer_token: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Fetch metadata (type, help) for all metrics (GET /api/v1/metadata).
    Returns dict: metric_name -> { "type": "counter"|"gauge"|"histogram"|"summary", "help": "..." }.
    """
    base = base_url.strip().rstrip("/")
    url = f"{base}/api/v1/metadata"
    try:
        resp = requests.get(
            url,
            headers=_headers(bearer_token),
            timeout=timeout,
        )
    except Exception as e:
        logger.exception("Prometheus metadata failed: url=%s error=%s", url, e)
        raise RuntimeError(f"Prometheus metadata failed: {e}") from e
    if resp.status_code >= 400:
        logger.error("Prometheus metadata error: status=%s body=%s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Prometheus API error: {resp.status_code} {resp.text}")
    data = resp.json()
    if not isinstance(data, dict):
        return {}
    raw = data.get("data")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, entries in raw.items():
        if not isinstance(entries, list) or not entries:
            continue
        first = entries[0] if isinstance(entries[0], dict) else {}
        out[str(name)] = {
            "type": str(first.get("type") or "unknown").lower(),
            "help": str(first.get("help") or "").strip(),
        }
    return out


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
