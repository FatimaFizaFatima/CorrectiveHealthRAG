from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from app.settings import setting

engine = create_engine(
    setting.DATABASE_URL, echo=True
)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass


if __name__ == "__main__":

    from app.model.corrective_health_model import CorrectiveRAGModel

    Base.metadata.create_all(bind=engine)
    print("Tables created successfully")
