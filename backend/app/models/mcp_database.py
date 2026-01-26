from typing import Optional, Any
from sqlmodel import Field, Column, String
from pydantic import BaseModel

from .base import UpdatableBaseModel, AESEncryptedColumn


class BaseMCPDatabase(UpdatableBaseModel):
    name: str = Field(max_length=64, unique=True, index=True)
    description: Optional[str] = Field(default=None, max_length=256)


class MCPDatabase(BaseMCPDatabase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # Encrypted JSON: { tidb_host, tidb_port, tidb_username, tidb_password, tidb_database }
    credentials: Any = Field(sa_column=Column(AESEncryptedColumn, nullable=False))

    __tablename__ = "mcp_databases"


class AdminMCPDatabase(BaseMCPDatabase):
    id: int


class MCPDatabaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    credentials: Optional[str | dict] = None

