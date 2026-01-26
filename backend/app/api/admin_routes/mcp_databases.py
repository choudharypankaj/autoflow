from typing import Dict
from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page, Params
from pydantic import BaseModel, Field

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models.mcp_database import MCPDatabase, AdminMCPDatabase, MCPDatabaseUpdate
from app.repositories.mcp_database import mcp_database_repo

router = APIRouter()


class MCPDatabaseTestResult(BaseModel):
    success: bool
    error: str = ""


def _verify_db_conn(credentials: Dict[str, str]) -> None:
    try:
        import pymysql  # type: ignore
    except Exception:
        raise HTTPException(status_code=400, detail="pymysql is not installed on server")
    try:
        conn = pymysql.connect(  # type: ignore
            host=str(credentials.get("tidb_host", "")),
            port=int(str(credentials.get("tidb_port", "0")) or "0"),
            user=str(credentials.get("tidb_username", "")),
            password=str(credentials.get("tidb_password", "")),
            database=str(credentials.get("tidb_database", "")),
            connect_timeout=6,
            read_timeout=6,
            write_timeout=6,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.Cursor,  # type: ignore[attr-defined]
        )
        try:
            with conn.cursor() as cur:  # type: ignore
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect to database: {e}")


@router.get("/admin/mcp/databases")
def list_mcp_databases(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[AdminMCPDatabase]:
    return mcp_database_repo.paginate(db_session, params)


@router.post("/admin/mcp/databases")
def create_mcp_database(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    db: MCPDatabase,
) -> AdminMCPDatabase:
    credentials = db.credentials or {}
    required = ["tidb_host", "tidb_port", "tidb_username", "tidb_password", "tidb_database"]
    if not all(str(credentials.get(k, "")).strip() for k in required):
        raise HTTPException(status_code=400, detail="credentials must include tidb_host, tidb_port, tidb_username, tidb_password, tidb_database")
    _verify_db_conn(credentials)
    return mcp_database_repo.create(db_session, db)


@router.get("/admin/mcp/databases/{db_id}")
def get_mcp_database(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    db_id: int,
) -> AdminMCPDatabase:
    db = mcp_database_repo.get(db_session, db_id)
    if not db:
        raise HTTPException(status_code=404, detail="Database not found")
    return db  # type: ignore[return-value]


@router.put("/admin/mcp/databases/{db_id}")
def update_mcp_database(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    db_id: int,
    update: MCPDatabaseUpdate,
) -> AdminMCPDatabase:
    db = mcp_database_repo.get(db_session, db_id)
    if not db:
        raise HTTPException(status_code=404, detail="Database not found")
    # If credentials are updated, verify again
    if update.credentials:
        _verify_db_conn(update.credentials)  # type: ignore[arg-type]
    return mcp_database_repo.update(db_session, db, update)


@router.delete("/admin/mcp/databases/{db_id}")
def delete_mcp_database(
    db_session: SessionDep,
    user: CurrentSuperuserDep,
    db_id: int,
) -> None:
    db = mcp_database_repo.get(db_session, db_id)
    if not db:
        raise HTTPException(status_code=404, detail="Database not found")
    mcp_database_repo.delete(db_session, db)
    return None

