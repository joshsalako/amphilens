"""User-scoped Modal credentials, kept outside project data."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..state import default_user_state_path


def default_credentials_path() -> Path:
    return default_user_state_path().with_name("credentials.json")


def _mask(value: str, visible: int = 4) -> str:
    return value[:visible] + "..." if len(value) > visible else "..."


@dataclass(frozen=True, slots=True, repr=False)
class CloudCredentials:
    token_id: str = field(repr=False)
    token_secret: str = field(repr=False)
    source: str = "credentials-file"

    def __repr__(self) -> str:
        return (
            "CloudCredentials(token_id='"
            + _mask(self.token_id)
            + "', token_secret='[redacted]', source='"
            + self.source
            + "')"
        )

    def describe(self) -> dict[str, str]:
        return {
            "source": self.source,
            "token_id": _mask(self.token_id),
            "token_secret": "[configured]",
        }


class CloudCredentialsStore:
    """Resolve explicit credentials, then Modal environment variables, then a 0600 file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser().resolve() if path else default_credentials_path()

    def save(self, token_id: str, token_secret: str) -> None:
        token_id, token_secret = token_id.strip(), token_secret.strip()
        if not token_id or not token_secret:
            raise ValueError("Both Modal token ID and token secret are required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".credentials-", dir=self.path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"token_id": token_id, "token_secret": token_secret},
                    handle,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def resolve(
        self,
        *,
        token_id: str | None = None,
        token_secret: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> CloudCredentials | None:
        if (token_id is None) != (token_secret is None):
            raise ValueError("Both explicit Modal token fields must be supplied together")
        if token_id is not None and token_secret is not None:
            return self._credentials(token_id, token_secret, "explicit")

        environment = os.environ if environ is None else environ
        env_id = environment.get("MODAL_TOKEN_ID")
        env_secret = environment.get("MODAL_TOKEN_SECRET")
        if env_id or env_secret:
            if not env_id or not env_secret:
                raise ValueError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set together")
            return self._credentials(env_id, env_secret, "environment")

        if environ is None:
            dotenv_credentials = self._dotenv_credentials()
            if dotenv_credentials is not None:
                return dotenv_credentials

        if not self.path.is_file():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read Modal credentials file: {self.path}") from exc
        if not isinstance(data, dict):
            raise ValueError("Modal credentials file must contain an object")
        return self._credentials(data.get("token_id"), data.get("token_secret"), "credentials-file")

    @classmethod
    def _dotenv_credentials(cls) -> CloudCredentials | None:
        try:
            from dotenv import dotenv_values, find_dotenv
        except ImportError:
            return None
        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path:
            return None
        values = dotenv_values(dotenv_path)
        token_id = values.get("MODAL_TOKEN_ID")
        token_secret = values.get("MODAL_TOKEN_SECRET")
        if not token_id and not token_secret:
            return None
        if not token_id or not token_secret:
            raise ValueError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set together")
        return cls._credentials(token_id, token_secret, "dotenv")

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _credentials(token_id, token_secret, source: str) -> CloudCredentials:
        if not isinstance(token_id, str) or not token_id.strip():
            raise ValueError("Modal token ID is missing")
        if not isinstance(token_secret, str) or not token_secret.strip():
            raise ValueError("Modal token secret is missing")
        return CloudCredentials(token_id.strip(), token_secret.strip(), source)
