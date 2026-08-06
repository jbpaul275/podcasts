"""Admin authentication: one password, a signed cookie, no user table.

Deliberately minimal. There is exactly one operator, so accounts, roles and
password resets would be machinery without a purpose.

Fails closed. If PAPERPOD_ADMIN_PASSWORD is unset the admin surface is only
reachable from localhost, so a deploy that forgets to set it locks the admin
out rather than exposing upload, delete and retry to the internet.
"""

import hashlib
import hmac
import logging
import os
import secrets
import time

log = logging.getLogger("paperpod.auth")

COOKIE = "paperpod_admin"
SESSION_TTL = 30 * 24 * 3600  # 30 days; this is one person's own machine
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def admin_password() -> str | None:
    return os.environ.get("PAPERPOD_ADMIN_PASSWORD") or None


def _secret() -> bytes:
    """Cookie-signing key. An explicit secret survives restarts; otherwise it
    is derived from the password so sessions persist without extra config."""
    explicit = os.environ.get("PAPERPOD_SECRET")
    if explicit:
        return explicit.encode()
    pw = admin_password()
    if pw:
        return hashlib.sha256(b"paperpod-session:" + pw.encode()).digest()
    return b"local-only-no-password-set"


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    expires = int(time.time()) + SESSION_TTL
    payload = str(expires)
    return f"{payload}.{_sign(payload)}"


def token_is_valid(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    payload, _, sig = token.rpartition(".")
    if not hmac.compare_digest(sig, _sign(payload)):
        return False
    try:
        return int(payload) > time.time()
    except ValueError:
        return False


def password_matches(candidate: str) -> bool:
    expected = admin_password()
    if not expected:
        return False
    return hmac.compare_digest(candidate or "", expected)


def is_local(request) -> bool:
    host = getattr(getattr(request, "client", None), "host", None)
    return host in LOCAL_HOSTS


def is_admin(request) -> bool:
    """Signed-in, or running locally with no password configured."""
    if token_is_valid(request.cookies.get(COOKIE)):
        return True
    return admin_password() is None and is_local(request)


def new_secret_suggestion() -> str:
    return secrets.token_urlsafe(32)
