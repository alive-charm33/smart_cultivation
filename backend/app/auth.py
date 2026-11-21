"""Simple JWT-based auth (signup / login / current user).

Uses bcrypt for passwords and python-jose for JWT tokens.
All stored in the same SQLite database (User table).

Install (already added to requirements.txt):
    pip install python-jose[cryptography] passlib[bcrypt]
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .db import User, get_session

# ------------------------------------------------------------------ config
SECRET_KEY = os.getenv("JWT_SECRET", "smart_cultivation_secret_key_change_in_production_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ------------------------------------------------------------------ helpers
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_token(user_id: int, username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": str(user_id), "username": username, "exp": expire},
                      SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ------------------------------------------------------------------ dependency
def get_current_user(token: str = Depends(oauth2_scheme),
                     s: Session = Depends(get_session)) -> Optional["User"]:
    """Returns the User if the token is valid, else None (auth is optional)."""
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    user = s.get(User, int(payload["sub"]))
    return user


def require_user(user: Optional["User"] = Depends(get_current_user)):
    """Use this dependency on endpoints that NEED a logged-in user."""
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


# ------------------------------------------------------------------ schemas
class SignupIn(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    mobile: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = "Maharashtra"


class UserOut(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    mobile: Optional[str]
    village: Optional[str]
    district: Optional[str]
    state: Optional[str]
    created_at: str


def _user_out(u: "User") -> dict:
    return {"id": u.id, "username": u.username, "full_name": u.full_name,
            "mobile": u.mobile, "village": u.village, "district": u.district,
            "state": u.state, "created_at": u.created_at.isoformat()}
