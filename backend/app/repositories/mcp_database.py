from typing import Optional

from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import update
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import select, Session

from app.models.mcp_database import MCPDatabase, MCPDatabaseUpdate
from app.repositories.base_repo import BaseRepo


class MCPDatabaseRepo(BaseRepo):
    model_cls = MCPDatabase

    def paginate(self, session: Session, params: Params = Params()) -> Page[MCPDatabase]:
        query = select(MCPDatabase).order_by(MCPDatabase.created_at.desc())
        return paginate(session, query, params)

    def get(self, session: Session, id: int) -> Optional[MCPDatabase]:
        return session.get(MCPDatabase, id)

    def get_by_name(self, session: Session, name: str) -> Optional[MCPDatabase]:
        stmt = select(MCPDatabase).where(MCPDatabase.name == name)
        return session.exec(stmt).first()

    def create(self, session: Session, obj: MCPDatabase) -> MCPDatabase:
        obj.id = None
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj

    def update(self, session: Session, db_obj: MCPDatabase, update_obj: MCPDatabaseUpdate) -> MCPDatabase:
        for field, value in update_obj.model_dump(exclude_unset=True).items():
            setattr(db_obj, field, value)
            flag_modified(db_obj, field)
        session.commit()
        session.refresh(db_obj)
        return db_obj

    def delete(self, session: Session, db_obj: MCPDatabase) -> None:
        session.delete(db_obj)
        session.commit()


mcp_database_repo = MCPDatabaseRepo()

