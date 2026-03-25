import uuid
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

from backend.config import get_settings

router = APIRouter()
bearer_scheme = HTTPBearer()

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 1

# In production, store client credentials in DB/secrets manager
CLIENTS = {
    "admin-client": {
        "secret": "admin-secret-change-in-prod",
        "scopes": ["query", "ingest", "admin"],
    },
    "query-client": {
        "secret": "query-secret-change-in-prod",
        "scopes": ["query"],
    },
}


class TokenRequest(BaseModel):
    client_id: str
    client_secret: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


def create_access_token(client_id: str, scopes: list[str]) -> str:
    settings = get_settings()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": client_id,
        "scopes": scopes,
        "exp": expire,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.api_secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.api_secret_key, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
) -> dict:
    return decode_token(credentials.credentials)


def require_scope(scope: str):
    async def _check(user: dict = Depends(get_current_user)):
        if scope not in user.get("scopes", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{scope}' required",
            )
        return user
    return _check


@router.post("/auth/token", response_model=TokenResponse)
async def get_token(body: TokenRequest):
    client = CLIENTS.get(body.client_id)
    if not client or client["secret"] != body.client_secret:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(body.client_id, client["scopes"])
    return TokenResponse(access_token=token, expires_in=ACCESS_TOKEN_EXPIRE_HOURS * 3600)
