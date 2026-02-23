from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

PROFILE_NAMES = ("demo", "dev", "uat", "prod")
DEFAULT_JSON_CONFIG_CANDIDATES = ("movex.json", "db_migrate.json", "db_profiles.json")
DEFAULT_ENV_CONFIG = ".env"


@dataclass
class DbConfig:
    dsn: str = ""
    host: str = ""
    port: int = 5432
    dbname: str = ""
    user: str = ""
    password: str = ""
    sslmode: str = "require"
    connect_timeout: int = 30

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DbConfig":
        if not data:
            return cls()

        def _int_or_default(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        dsn = str(
            data.get("dsn")
            or data.get("connection_string")
            or data.get("url")
            or data.get("uri")
            or ""
        ).strip()

        return cls(
            dsn=dsn,
            host=str(data.get("host", "")).strip(),
            port=_int_or_default(data.get("port", 5432), 5432),
            dbname=str(data.get("dbname") or data.get("database") or "").strip(),
            user=str(data.get("user") or data.get("username") or "").strip(),
            password=str(data.get("password", "")),
            sslmode=str(data.get("sslmode", "require")).strip() or "require",
            connect_timeout=_int_or_default(data.get("connect_timeout", 30), 30),
        )

    def to_conn_kwargs(self) -> dict[str, Any]:
        if self.dsn:
            kwargs: dict[str, Any] = {"dsn": self.dsn}
            lowered_dsn = self.dsn.lower()
            if self.sslmode and "sslmode=" not in lowered_dsn:
                kwargs["sslmode"] = self.sslmode
            if self.connect_timeout > 0 and "connect_timeout=" not in lowered_dsn:
                kwargs["connect_timeout"] = self.connect_timeout
            return kwargs

        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "sslmode": self.sslmode,
            "connect_timeout": self.connect_timeout,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dsn": self.dsn,
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "sslmode": self.sslmode,
            "connect_timeout": self.connect_timeout,
        }

    def is_complete(self) -> bool:
        return bool(self.dsn or (self.host and self.dbname and self.user and self.password))

    def has_any_values(self) -> bool:
        return bool(
            self.dsn
            or self.host
            or self.dbname
            or self.user
            or self.password
            or (self.sslmode and self.sslmode != "require")
            or self.port != 5432
            or self.connect_timeout != 30
        )


@dataclass
class AppConfig:
    profiles: dict[str, DbConfig]
    source: DbConfig
    target: DbConfig
    config_path: Path
    env_path: Path


def resolve_default_json_config(cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    for candidate in DEFAULT_JSON_CONFIG_CANDIDATES:
        candidate_path = base / candidate
        if candidate_path.exists():
            return candidate_path
    return base / DEFAULT_JSON_CONFIG_CANDIDATES[0]


def resolve_default_env_file(cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    return base / DEFAULT_ENV_CONFIG


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def default_profile_payload() -> dict[str, dict[str, Any]]:
    return {name: DbConfig().to_dict() for name in PROFILE_NAMES}


def json_template_payload() -> dict[str, Any]:
    return {
        "source": {
            "dsn": "postgresql://user:password@localhost:5432/source_db?sslmode=require",
        },
        "target": {
            "dsn": "postgresql://user:password@localhost:5432/target_db?sslmode=require",
        },
        "profiles": default_profile_payload(),
    }


def env_template_lines() -> str:
    return "\n".join(
        [
            "# Source connection",
            "MOVEX_SOURCE_DSN=postgresql://user:password@localhost:5432/source_db?sslmode=require",
            "",
            "# Target connection",
            "MOVEX_TARGET_DSN=postgresql://user:password@localhost:5432/target_db?sslmode=require",
            "",
            "# Optional named profiles",
            "MOVEX_PROFILE_DEMO_DSN=postgresql://user:password@localhost:5432/demo_db?sslmode=require",
            "MOVEX_PROFILE_DEV_DSN=postgresql://user:password@localhost:5432/dev_db?sslmode=require",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _extract_env_mapping(env: Mapping[str, str], prefix: str) -> dict[str, str]:
    aliases = {
        "DSN": "dsn",
        "CONNECTION_STRING": "dsn",
        "URL": "dsn",
        "URI": "dsn",
        "HOST": "host",
        "PORT": "port",
        "DBNAME": "dbname",
        "DATABASE": "dbname",
        "USER": "user",
        "USERNAME": "user",
        "PASSWORD": "password",
        "SSLMODE": "sslmode",
        "CONNECT_TIMEOUT": "connect_timeout",
    }

    result: dict[str, str] = {}
    for env_suffix, field_name in aliases.items():
        env_key = f"{prefix}{env_suffix}"
        if env_key in env and env[env_key] != "":
            result[field_name] = env[env_key]
    return result


def _apply_profile_env_overrides(
    profiles_raw: dict[str, dict[str, Any]],
    env_values: Mapping[str, str],
) -> None:
    field_aliases = {
        "DSN": "dsn",
        "CONNECTION_STRING": "dsn",
        "URL": "dsn",
        "URI": "dsn",
        "HOST": "host",
        "PORT": "port",
        "DBNAME": "dbname",
        "DATABASE": "dbname",
        "USER": "user",
        "USERNAME": "user",
        "PASSWORD": "password",
        "SSLMODE": "sslmode",
        "CONNECT_TIMEOUT": "connect_timeout",
    }
    patterns = (
        re.compile(r"^DBM_PROFILE_([A-Z0-9_]+)_([A-Z_]+)$"),
        re.compile(r"^MOVEX_PROFILE_([A-Z0-9_]+)_([A-Z_]+)$"),
    )

    for key, value in env_values.items():
        match = None
        for pattern in patterns:
            match = pattern.match(key)
            if match:
                break
        if not match:
            continue

        profile_name = match.group(1).lower()
        field_raw = match.group(2)
        field_name = field_aliases.get(field_raw)
        if not field_name:
            continue
        if profile_name not in profiles_raw:
            profiles_raw[profile_name] = {}
        profiles_raw[profile_name][field_name] = value


def load_app_config(
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> AppConfig:
    config_path = config_path or resolve_default_json_config()
    env_path = env_path or resolve_default_env_file()

    raw_json = _read_json(config_path)

    profiles_raw: dict[str, dict[str, Any]] = {
        name: {} for name in PROFILE_NAMES
    }
    profiles_section = raw_json.get("profiles")
    if isinstance(profiles_section, dict):
        for name, value in profiles_section.items():
            if isinstance(value, dict):
                profiles_raw[str(name).lower()] = dict(value)
    else:
        for name in PROFILE_NAMES:
            value = raw_json.get(name)
            if isinstance(value, dict):
                profiles_raw[name] = dict(value)

    source_raw: dict[str, Any] = {}
    target_raw: dict[str, Any] = {}
    if isinstance(raw_json.get("source"), dict):
        source_raw.update(raw_json["source"])
    if isinstance(raw_json.get("target"), dict):
        target_raw.update(raw_json["target"])

    connections_section = raw_json.get("connections")
    if isinstance(connections_section, dict):
        if isinstance(connections_section.get("source"), dict):
            source_raw.update(connections_section["source"])
        if isinstance(connections_section.get("target"), dict):
            target_raw.update(connections_section["target"])

    env_values = parse_env_file(env_path)
    env_values.update(os.environ)

    source_raw.update(_extract_env_mapping(env_values, "DBM_SOURCE_"))
    target_raw.update(_extract_env_mapping(env_values, "DBM_TARGET_"))
    source_raw.update(_extract_env_mapping(env_values, "MOVEX_SOURCE_"))
    target_raw.update(_extract_env_mapping(env_values, "MOVEX_TARGET_"))
    _apply_profile_env_overrides(profiles_raw, env_values)

    profiles = {name: DbConfig.from_mapping(data) for name, data in profiles_raw.items()}
    source = DbConfig.from_mapping(source_raw)
    target = DbConfig.from_mapping(target_raw)

    return AppConfig(
        profiles=profiles,
        source=source,
        target=target,
        config_path=config_path,
        env_path=env_path,
    )


def save_json_config(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
