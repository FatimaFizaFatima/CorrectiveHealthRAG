from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, Integer, String
from datetime import datetime, timezone
from app.db.database import Base


class CorrectiveRAGModel(Base):

    __tablename__ = "corrective_rags"
    id : Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename : Mapped[str] = mapped_column(String, nullable=False)
    question : Mapped[str] = mapped_column(String, nullable=False)
    answer : Mapped[str] = mapped_column(String, nullable=False)
    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda:datetime.now(timezone.utc))
 