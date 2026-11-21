"""Database (SQLAlchemy). SQLite by default; PostgreSQL by setting DATABASE_URL."""
from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from . import config

_kw = {"connect_args": {"check_same_thread": False}} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, **_kw)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Farmer account. Farms are linked to a user_id."""
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(200))
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    mobile: Mapped[str | None] = mapped_column(String(20), nullable=True)
    village: Mapped[str | None] = mapped_column(String(120), nullable=True)
    district: Mapped[str | None] = mapped_column(String(120), nullable=True)
    state: Mapped[str | None] = mapped_column(String(80), nullable=True, default="Maharashtra")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    farms: Mapped[list["Farm"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Farm(Base):
    __tablename__ = "farms"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    farmer_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    village: Mapped[str | None] = mapped_column(String(120), nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_acres: Mapped[float] = mapped_column(Float, default=1.0)
    variety: Mapped[str | None] = mapped_column(String(120), nullable=True)
    transplant_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User | None"] = relationship(back_populates="farms")
    suitability: Mapped[list["SuitabilityResult"]] = relationship(back_populates="farm", cascade="all, delete-orphan")
    scans: Mapped[list["DiseaseScan"]] = relationship(back_populates="farm", cascade="all, delete-orphan")
    growth: Mapped[list["GrowthRecord"]] = relationship(back_populates="farm", cascade="all, delete-orphan")


class SuitabilityResult(Base):
    __tablename__ = "suitability_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    farm_id: Mapped[int | None] = mapped_column(ForeignKey("farms.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    suitable: Mapped[bool] = mapped_column()
    probability: Mapped[float] = mapped_column(Float)
    result: Mapped[dict] = mapped_column(JSON)
    farm: Mapped[Farm | None] = relationship(back_populates="suitability")


class DiseaseScan(Base):
    __tablename__ = "disease_scans"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    farm_id: Mapped[int | None] = mapped_column(ForeignKey("farms.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    original_filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(500))
    gradcam_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    disease_id: Mapped[str] = mapped_column(String(80))
    confidence: Mapped[float] = mapped_column(Float)
    predictions: Mapped[dict] = mapped_column(JSON)
    advice: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage: Mapped[str | None] = mapped_column(String(60), nullable=True)
    farm: Mapped[Farm | None] = relationship(back_populates="scans")


class GrowthRecord(Base):
    __tablename__ = "growth_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    farm_id: Mapped[int | None] = mapped_column(ForeignKey("farms.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    original_filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(500))
    days_after_transplant: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stage_id: Mapped[str] = mapped_column(String(60))
    method: Mapped[str] = mapped_column(String(120))
    details: Mapped[dict] = mapped_column(JSON)
    farm: Mapped[Farm | None] = relationship(back_populates="growth")


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
