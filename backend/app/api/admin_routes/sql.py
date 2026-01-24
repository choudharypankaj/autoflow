import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.api.deps import SessionDep, CurrentSuperuserDep


router = APIRouter()

_FORBIDDEN_KEYWORDS = [
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "replace",
    "merge",
    "call",
    "execute",
    "into",
    "begin",
    "commit",
    "rollback",
]


class AdhocQueryRequest(BaseModel):
    sql: str = Field(..., description="Read-only SQL query (SELECT/CTE only)")
    max_rows: int = Field(100, ge=1, le=1000)


class AdhocQueryResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    truncated: bool


def _validate_readonly_sql(sql: str) -> None:
    if len(sql) > 20000:
        raise HTTPException(status_code=400, detail="SQL too long")

    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty SQL")

    lowered = cleaned.lower()
    # Allow SELECT or CTE (WITH ... SELECT ...)
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise HTTPException(status_code=400, detail="Only SELECT/CTE queries are allowed")

    # Disallow obvious DML/DDL keywords via word-boundary search
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\\b{kw}\\b", lowered):
            raise HTTPException(status_code=400, detail=f"Keyword '{kw}' is not allowed")

    # Disallow multiple statements (extra semicolons)
    if sql.count(";") > 1:
        raise HTTPException(status_code=400, detail="Multiple statements are not allowed")


@router.post("/admin/sql/run", response_model=AdhocQueryResponse)
def run_sql(session: SessionDep, user: CurrentSuperuserDep, request: AdhocQueryRequest):
    _validate_readonly_sql(request.sql)

    result = session.exec(text(request.sql))  # type: ignore[arg-type]
    columns = list(result.keys())
    rows: List[Dict[str, Any]] = []
    truncated = False

    for idx, row in enumerate(result):
        if idx >= request.max_rows:
            truncated = True
            break
        # Convert SQLAlchemy Row to dict
        if hasattr(row, "_mapping"):
            rows.append(dict(row._mapping))
        elif isinstance(row, (list, tuple)):
            rows.append({columns[i] if i < len(columns) else str(i): row[i] for i in range(len(row))})
        else:
            # Fallback: single scalar
            key = columns[0] if columns else "value"
            rows.append({key: row})

    return AdhocQueryResponse(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
    )

