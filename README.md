# MoveX

MoveX is a fast PostgreSQL schema transfer CLI + Python library.

## Features
- PostgreSQL DSN/connection-string support
- Single config file (`movex.json`) or `.env`
- Schema listing command
- Multi-select schemas and tables in interactive mode
- Parallel table migration workers (`--workers`)
- Rich transfer UI with progress bars for table copy, constraints, and indexes

## Install

One-line Windows install (PowerShell):
```powershell
irm https://raw.githubusercontent.com/MonishGosar/movex/main/install.ps1 | iex
```

Or manual install:

```powershell
git clone https://github.com/MonishGosar/movex.git
cd .\movex
python -m pip install .
```

Run:
```powershell
movex --help
```

## Quick Start

1. Generate a one-file config template:
```powershell
movex init-config --format json
```

2. Edit `movex.json`:
```json
{
  "source": {
    "dsn": "postgresql://user:password@host:5432/source_db?sslmode=require"
  },
  "target": {
    "dsn": "postgresql://user:password@host:5432/target_db?sslmode=require"
  },
  "profiles": {
    "demo": { "dsn": "" },
    "dev": { "dsn": "" },
    "uat": { "dsn": "" },
    "prod": { "dsn": "" }
  }
}
```

3. List source schemas:
```powershell
movex list-schemas
```

4. Run migration:
```powershell
movex migrate
```

## `.env` Alternative

```dotenv
MOVEX_SOURCE_DSN=postgresql://user:password@host:5432/source_db?sslmode=require
MOVEX_TARGET_DSN=postgresql://user:password@host:5432/target_db?sslmode=require
```

Optional profile variables:
```dotenv
MOVEX_PROFILE_DEMO_DSN=postgresql://user:password@host:5432/demo_db?sslmode=require
MOVEX_PROFILE_DEV_DSN=postgresql://user:password@host:5432/dev_db?sslmode=require
```

Legacy `DBM_*` env vars are still supported for backward compatibility.

## Commands

List schemas:
```powershell
movex list-schemas --source-dsn "postgresql://user:password@host:5432/db?sslmode=require"
```

Interactive migrate:
```powershell
movex migrate
```

Fast non-interactive migrate (all schemas/tables, 8 workers):
```powershell
movex migrate --all-schemas --all-tables --workers 8 -y --non-interactive
```

Specific schemas:
```powershell
movex migrate --schemas "public,analytics"
```

Use named profiles:
```powershell
movex migrate --source-profile dev --target-profile uat
```

Use sequential mode:
```powershell
movex migrate --workers 1
```

## Python API

```python
from movex import load_app_config, connect, list_user_schemas

cfg = load_app_config()
source = connect(cfg.source)
print(list_user_schemas(source))
source.close()
```
