"""
Shared helpers for Prometheus-shaped series data (result from /api/v1/query or query_range).
Used by slow_query_prometheus for summarization.
"""
import logging


def extract_series_values(series: list) -> list[float]:
    values = []
    for s in series:
        if isinstance(s, dict):
            data = s.get("data") or s
            result_items = None
            if isinstance(data, dict):
                if isinstance(data.get("result"), list):
                    result_items = data.get("result")
                elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("result"), list):
                    result_items = data["data"].get("result")
            if isinstance(result_items, list):
                for item in result_items:
                    if not isinstance(item, dict):
                        continue
                    vals = item.get("values")
                    if isinstance(vals, list):
                        for v in vals:
                            if isinstance(v, (list, tuple)) and len(v) >= 2:
                                try:
                                    values.append(float(v[1]))
                                except Exception:
                                    continue
                    else:
                        single = item.get("value")
                        if isinstance(single, (list, tuple)) and len(single) >= 2:
                            try:
                                values.append(float(single[1]))
                            except Exception:
                                continue
    return values


def extract_entry_values_by_label(entry: dict, label_key: str) -> dict[str, list[float]]:
    values_by_label: dict[str, list[float]] = {}
    if not isinstance(entry, dict):
        return values_by_label
    data = entry.get("data") or entry
    result_items = None
    if isinstance(data, dict):
        if isinstance(data.get("result"), list):
            result_items = data.get("result")
        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("result"), list):
            result_items = data["data"].get("result")
    if not isinstance(result_items, list):
        return values_by_label
    for item in result_items:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric") if isinstance(item.get("metric"), dict) else {}
        label = str(metric.get(label_key) or "").strip() if isinstance(metric, dict) else ""
        if not label:
            label = "unknown"
        bucket = values_by_label.setdefault(label, [])
        vals = item.get("values")
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        bucket.append(float(v[1]))
                    except Exception:
                        continue
        else:
            single = item.get("value")
            if isinstance(single, (list, tuple)) and len(single) >= 2:
                try:
                    bucket.append(float(single[1]))
                except Exception:
                    continue
    return values_by_label


def extract_entry_values_by_best_label(
    entry: dict,
    candidate_keys: list[str],
    logger: logging.Logger,
    context: str,
) -> tuple[str | None, dict[str, list[float]]]:
    best_key = None
    best_values: dict[str, list[float]] = {}
    for key in candidate_keys:
        values = extract_entry_values_by_label(entry, key)
        if len(values) > len(best_values):
            best_values = values
            best_key = key
    logger.info(
        "%s metrics: best_label_key=%s distinct=%s",
        context,
        best_key,
        len(best_values),
    )
    return best_key, best_values


def extract_series_values_by_label(series: list, label_key: str) -> dict[str, list[float]]:
    values_by_label: dict[str, list[float]] = {}
    for s in series:
        if not isinstance(s, dict):
            continue
        data = s.get("data") or s
        result_items = None
        if isinstance(data, dict):
            if isinstance(data.get("result"), list):
                result_items = data.get("result")
            elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("result"), list):
                result_items = data["data"].get("result")
        if not isinstance(result_items, list):
            continue
        for item in result_items:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric") if isinstance(item.get("metric"), dict) else {}
            label = str(metric.get(label_key) or "").strip() if isinstance(metric, dict) else ""
            if not label:
                label = "unknown"
            bucket = values_by_label.setdefault(label, [])
            vals = item.get("values")
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        try:
                            bucket.append(float(v[1]))
                        except Exception:
                            continue
            else:
                single = item.get("value")
                if isinstance(single, (list, tuple)) and len(single) >= 2:
                    try:
                        bucket.append(float(single[1]))
                    except Exception:
                        continue
    return values_by_label


def summarize_panel_series(series: list, *, per_instance: bool = False) -> str:
    if per_instance:
        values_by_instance = extract_series_values_by_label(series, "instance")
        if not values_by_instance:
            values_by_instance = extract_series_values_by_label(series, "tidb_instance")
        if not values_by_instance:
            values = extract_series_values(series)
            if not values:
                return "- No data points found."
            avg = sum(values) / len(values)
            max_v = max(values)
            return f"- avg: {avg:.6f}\n- max: {max_v:.6f}"
        lines = []
        for instance, values in sorted(values_by_instance.items()):
            if not values:
                lines.append(f"- {instance}: no data")
                continue
            avg = sum(values) / len(values)
            max_v = max(values)
            lines.append(f"- {instance}: avg {avg:.6f}, max {max_v:.6f}")
        return "\n".join(lines)

    values = extract_series_values(series)
    if not values:
        return "- No data points found."
    avg = sum(values) / len(values)
    max_v = max(values)
    return f"- avg: {avg:.6f}\n- max: {max_v:.6f}"


def summarize_cpu_series(series: list, targets: list | None, logger: logging.Logger) -> str:
    if not series:
        return "- No data points found."
    if not isinstance(targets, list) or not targets:
        logger.info("CPU metrics: targets missing; falling back to per-instance raw stats")
        return summarize_panel_series(series, per_instance=True)

    quota_idx = None
    actual_idx = None
    for idx, target in enumerate(targets):
        if not isinstance(target, dict):
            continue
        legend = str(target.get("legendFormat") or target.get("legend") or "").strip().lower()
        expr = str(target.get("expr") or target.get("query") or "").strip().lower()
        logger.info("CPU metrics: target[%s] legend=%s expr=%s", idx, legend, expr)
        if (
            "quota-" in legend
            and ("{{instance}}" in legend or "${instance}" in legend or "$instance" in legend)
        ) or ("quota" in legend) or ("maxprocs" in expr):
            quota_idx = idx
        elif actual_idx is None:
            actual_idx = idx
    logger.info(
        "CPU metrics: targets=%s actual_idx=%s quota_idx=%s",
        len(targets),
        actual_idx,
        quota_idx,
    )

    if actual_idx is None:
        actual_idx = 0
    if actual_idx >= len(series):
        logger.info(
            "CPU metrics: actual_idx out of range actual_idx=%s series_len=%s",
            actual_idx,
            len(series),
        )
        return "- No data points found."

    _, actual_by_instance = extract_entry_values_by_best_label(
        series[actual_idx],
        ["instance", "tidb_instance", "pod", "pod_name", "instance_addr"],
        logger,
        "CPU actual",
    )
    logger.info("CPU metrics: actual_instances=%s", len(actual_by_instance))

    quota_by_instance: dict[str, list[float]] = {}
    if quota_idx is not None and quota_idx < len(series):
        _, quota_by_instance = extract_entry_values_by_best_label(
            series[quota_idx],
            ["instance", "tidb_instance", "pod", "pod_name", "instance_addr"],
            logger,
            "CPU quota",
        )
    logger.info("CPU metrics: quota_instances=%s", len(quota_by_instance))

    if not actual_by_instance:
        return "- No data points found."

    lines = []
    table_rows = []
    all_pcts: list[float] = []
    for instance, values in sorted(actual_by_instance.items()):
        if not values:
            table_rows.append((instance, "-", "-", "-", "-"))
            continue
        avg = sum(values) / len(values)
        max_v = max(values)
        quota_vals = quota_by_instance.get(instance, [])
        if quota_vals:
            quota_max = max(quota_vals)
            if quota_max > 0:
                pct_values = [(v / quota_max * 100.0) for v in values]
                avg_pct = sum(pct_values) / len(pct_values)
                max_pct = max(pct_values)
                logger.info(
                    "CPU metrics: instance=%s avg=%s max=%s quota_max=%s avg_pct=%.2f max_pct=%.2f",
                    instance,
                    avg,
                    max_v,
                    quota_max,
                    avg_pct,
                    max_pct,
                )
                all_pcts.extend(pct_values)
                table_rows.append(
                    (instance, f"{avg_pct:.2f}%", f"{max_pct:.2f}%", f"{avg:.6f}", f"{quota_max:.2f}", pct_values)
                )
                continue
        logger.info(
            "CPU metrics: instance=%s avg=%s max=%s quota_missing_or_zero=%s",
            instance,
            avg,
            max_v,
            not quota_vals or (max(quota_vals) if quota_vals else 0) <= 0,
        )
        table_rows.append((instance, "-", "-", f"{avg:.6f}", "-", []))

    if not table_rows:
        return "- No data points found."
    if all_pcts:
        overall_avg = sum(all_pcts) / len(all_pcts)
        threshold = overall_avg + 5.0
        filtered = []
        for r in table_rows:
            if r[1] == "-":
                continue
            pct_values = r[5] if len(r) > 5 else []
            if pct_values:
                rolling_avgs = [
                    (pct_values[i] + pct_values[i + 1]) / 2.0
                    for i in range(len(pct_values) - 1)
                ]
                if rolling_avgs and max(rolling_avgs) > threshold:
                    filtered.append(r)
        table_rows = filtered
        if not table_rows:
            return "- No instances above avg + 5%."
    header = "| instance | avg_pct | max_pct |"
    sep = "|---|---:|---:|"
    body = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in table_rows)
    return "\n".join([header, sep, body])
