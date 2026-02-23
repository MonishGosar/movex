from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import (
    AppConfig,
    DbConfig,
    env_template_lines,
    json_template_payload,
    load_app_config,
    resolve_default_env_file,
    resolve_default_json_config,
    save_json_config,
)
from .core import (
    MigrationSummary,
    build_plan_for_schemas,
    connect,
    list_tables,
    list_user_schemas,
    migrate,
    migrate_parallel,
)

console = Console()


@dataclass
class EndpointResolution:
    config: DbConfig
    origin: str
    profile_name: str | None = None


def _must_answer(value: str | None) -> str:
    if value is None:
        raise KeyboardInterrupt
    return value.strip()


def ask_text(prompt: str, default: str | None = None) -> str:
    return _must_answer(questionary.text(prompt, default=default or "").ask())


def ask_secret(prompt: str) -> str:
    return _must_answer(questionary.password(prompt).ask())


def ask_confirm(prompt: str, default: bool = True) -> bool:
    value = questionary.confirm(prompt, default=default).ask()
    if value is None:
        raise KeyboardInterrupt
    return bool(value)


def render_banner() -> None:
    console.print(
        Panel.fit(
            "[bold bright_white]MoveX[/bold bright_white]\n"
            "[cyan]PostgreSQL schema transfer CLI + library[/cyan]",
            border_style="bright_blue",
            box=box.ROUNDED,
        )
    )


def _int_or_default(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def prompt_db_config(label: str, defaults: DbConfig | None = None) -> DbConfig:
    default_cfg = defaults or DbConfig()
    console.print(f"\n[bold]{label} connection[/bold]")

    use_dsn = ask_confirm(
        f"Use PostgreSQL connection string (DSN) for {label}?",
        default=bool(default_cfg.dsn),
    )
    timeout_default = str(default_cfg.connect_timeout or 30)
    sslmode_default = default_cfg.sslmode or "require"

    if use_dsn:
        dsn = ask_text(
            f"{label} DSN",
            default=default_cfg.dsn
            or "postgresql://user:password@host:5432/dbname?sslmode=require",
        )
        sslmode = ask_text(f"{label} sslmode", default=sslmode_default)
        timeout = _int_or_default(
            ask_text(f"{label} connect timeout (seconds)", default=timeout_default),
            30,
        )
        return DbConfig(
            dsn=dsn,
            sslmode=sslmode,
            connect_timeout=timeout,
        )

    host = ask_text(f"{label} host", default=default_cfg.host or "")
    port = _int_or_default(ask_text(f"{label} port", default=str(default_cfg.port or 5432)), 5432)
    dbname = ask_text(f"{label} database name", default=default_cfg.dbname or "")
    user = ask_text(f"{label} username", default=default_cfg.user or "")

    password = default_cfg.password
    if default_cfg.password:
        keep_existing_password = ask_confirm(
            f"{label}: keep existing password?",
            default=True,
        )
        if not keep_existing_password:
            password = ask_secret(f"{label} password")
    else:
        password = ask_secret(f"{label} password")

    sslmode = ask_text(f"{label} sslmode", default=sslmode_default)
    timeout = _int_or_default(
        ask_text(f"{label} connect timeout (seconds)", default=timeout_default),
        30,
    )

    return DbConfig(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
        connect_timeout=timeout,
    )


def _endpoint_cli_mapping(args: argparse.Namespace, endpoint: str) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    fields = {
        "dsn": f"{endpoint}_dsn",
        "host": f"{endpoint}_host",
        "port": f"{endpoint}_port",
        "dbname": f"{endpoint}_dbname",
        "user": f"{endpoint}_user",
        "password": f"{endpoint}_password",
        "sslmode": f"{endpoint}_sslmode",
        "connect_timeout": f"{endpoint}_connect_timeout",
    }
    for key, attr in fields.items():
        value = getattr(args, attr, None)
        if value is not None:
            mapping[key] = value
    return mapping


def pick_profile(
    label: str,
    config: AppConfig,
    blocked_profile: str | None = None,
) -> str | None:
    options = sorted(
        name
        for name, db_cfg in config.profiles.items()
        if db_cfg.is_complete() and name != blocked_profile
    )
    if not options:
        return None

    choice = questionary.select(
        f"Select {label} profile (or custom)",
        choices=options + ["custom"],
    ).ask()
    if choice is None:
        raise KeyboardInterrupt
    if choice == "custom":
        return None
    return str(choice)


def _missing_fields(cfg: DbConfig) -> List[str]:
    if cfg.dsn:
        return []
    missing: List[str] = []
    if not cfg.host:
        missing.append("host")
    if not cfg.dbname:
        missing.append("dbname")
    if not cfg.user:
        missing.append("user")
    if not cfg.password:
        missing.append("password")
    return missing


def resolve_endpoint(
    endpoint: str,
    label: str,
    args: argparse.Namespace,
    app_config: AppConfig,
    interactive: bool,
    blocked_profile: str | None = None,
) -> EndpointResolution:
    profile_name_raw = getattr(args, f"{endpoint}_profile", None)
    profile_name = str(profile_name_raw).lower() if profile_name_raw else None
    overrides = _endpoint_cli_mapping(args, endpoint)

    base: Dict[str, Any] = {}
    origin = "manual"

    if profile_name:
        selected = app_config.profiles.get(profile_name)
        if selected is None:
            known = ", ".join(sorted(app_config.profiles.keys()))
            raise ValueError(f"Unknown {label} profile '{profile_name}'. Known profiles: {known}")
        base.update(selected.to_dict())
        origin = f"profile:{profile_name}"
    else:
        file_default = app_config.source if endpoint == "source" else app_config.target
        if file_default.has_any_values():
            base.update(file_default.to_dict())
            origin = "config"

    base.update(overrides)
    resolved = DbConfig.from_mapping(base)
    if resolved.is_complete():
        if overrides:
            origin = f"{origin}+cli"
        return EndpointResolution(config=resolved, origin=origin, profile_name=profile_name)

    if not interactive:
        missing = ", ".join(_missing_fields(resolved))
        raise ValueError(
            f"Incomplete {label} configuration. Missing: {missing or 'DSN/details'}. "
            f"Use --{endpoint}-dsn, --{endpoint}-profile, or config/.env."
        )

    picked_profile = pick_profile(label, app_config, blocked_profile=blocked_profile)
    if picked_profile:
        cfg = app_config.profiles[picked_profile]
        if cfg.is_complete():
            return EndpointResolution(
                config=cfg,
                origin=f"profile:{picked_profile}",
                profile_name=picked_profile,
            )

    return EndpointResolution(
        config=prompt_db_config(label, defaults=resolved if resolved.has_any_values() else None),
        origin="prompt",
        profile_name=None,
    )


def connect_with_retry(
    label: str,
    initial: EndpointResolution,
    interactive: bool,
) -> tuple[DbConfig, Any]:
    config = initial.config
    while True:
        try:
            conn = connect(config, autocommit=False)
            console.print(f"[green]{label} connected[/green] ({initial.origin})")
            return config, conn
        except Exception as exc:
            console.print(f"[red]{label} connection failed:[/red] {exc}")
            if not interactive:
                raise
            if not ask_confirm("Edit credentials and retry?", default=True):
                raise
            config = prompt_db_config(label, defaults=config)


def parse_schema_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[,\s]+", raw) if part.strip()]


def pick_schema_table_plan(
    source_conn: Any,
    schemas: list[str],
    interactive: bool,
    all_tables: bool,
) -> dict[str, list[str]]:
    if all_tables or not interactive:
        return build_plan_for_schemas(source_conn, schemas)

    plan: dict[str, list[str]] = {}
    for schema in schemas:
        tables = list_tables(source_conn, schema)
        if not tables:
            console.print(f"[yellow]Schema '{schema}' has no base tables. Skipping.[/yellow]")
            continue

        migrate_all = ask_confirm(
            f"Migrate all {len(tables)} tables from '{schema}'?",
            default=True,
        )
        if migrate_all:
            plan[schema] = tables
            continue

        selected_tables = questionary.checkbox(
            f"Select tables from '{schema}'",
            choices=tables,
        ).ask()
        if selected_tables is None:
            raise KeyboardInterrupt
        if selected_tables:
            plan[schema] = list(selected_tables)
    return plan


def print_schema_list(schemas: list[str]) -> None:
    table = Table(title="Source Schemas", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Schema", style="green")
    for idx, name in enumerate(schemas, start=1):
        table.add_row(str(idx), name)
    console.print(table)


def print_plan(plan: dict[str, list[str]]) -> None:
    table = Table(title="Migration Plan", box=box.SIMPLE_HEAVY)
    table.add_column("Schema", style="cyan", no_wrap=True)
    table.add_column("Tables", style="green")
    table.add_column("Count", justify="right", style="magenta")

    total = 0
    for schema, tables in plan.items():
        total += len(tables)
        table.add_row(schema, ", ".join(tables), str(len(tables)))
    table.add_row("[bold]TOTAL[/bold]", "", f"[bold]{total}[/bold]")
    console.print(table)


def print_summary(summary: MigrationSummary) -> int:
    summary_table = Table(title="MoveX Transfer Summary", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")
    summary_table.add_row("Tables planned", str(summary.total_tables))
    summary_table.add_row("Tables migrated", str(summary.migrated_tables))
    summary_table.add_row(
        "Constraints applied",
        f"{summary.applied_constraints}/{summary.total_constraints}",
    )
    summary_table.add_row("Indexes applied", f"{summary.applied_indexes}/{summary.total_indexes}")
    summary_table.add_row("Failures", str(len(summary.failures)))
    console.print(summary_table)

    if summary.failures:
        fail_table = Table(title="Failures", box=box.SIMPLE_HEAVY)
        fail_table.add_column("Object", style="cyan")
        fail_table.add_column("Step", style="yellow")
        fail_table.add_column("Error", style="red")
        for item in summary.failures:
            fail_table.add_row(item.object_name, item.step, item.error[:2000])
        console.print(fail_table)
        return 2

    console.print("[green]Migration completed successfully.[/green]")
    return 0


def init_config_command(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    if args.format == "json":
        target = output or resolve_default_json_config()
        if target.exists() and not args.force:
            console.print(f"[red]{target} already exists. Use --force to overwrite.[/red]")
            return 1
        save_json_config(target, json_template_payload())
        console.print(f"[green]Created JSON config template:[/green] {target}")
        return 0

    target = output or resolve_default_env_file()
    if target.exists() and not args.force:
        console.print(f"[red]{target} already exists. Use --force to overwrite.[/red]")
        return 1
    target.write_text(env_template_lines(), encoding="utf-8")
    console.print(f"[green]Created .env template:[/green] {target}")
    return 0


def list_schemas_command(args: argparse.Namespace) -> int:
    app_config = load_app_config(
        config_path=Path(args.config) if args.config else None,
        env_path=Path(args.env_file) if args.env_file else None,
    )
    interactive = not args.non_interactive
    source_resolution = resolve_endpoint(
        endpoint="source",
        label="Source",
        args=args,
        app_config=app_config,
        interactive=interactive,
    )

    source_conn = None
    try:
        _, source_conn = connect_with_retry("Source", source_resolution, interactive=interactive)
        schemas = list_user_schemas(source_conn)
        if args.json_output:
            console.print(json.dumps(schemas, indent=2))
            return 0
        print_schema_list(schemas)
        return 0
    finally:
        if source_conn is not None:
            source_conn.close()


def migrate_command(args: argparse.Namespace) -> int:
    render_banner()
    console.print(
        "[bold yellow]Warning:[/bold yellow] selected target tables are dropped/recreated before copy."
    )

    app_config = load_app_config(
        config_path=Path(args.config) if args.config else None,
        env_path=Path(args.env_file) if args.env_file else None,
    )
    interactive = not args.non_interactive

    source_resolution = resolve_endpoint(
        endpoint="source",
        label="Source",
        args=args,
        app_config=app_config,
        interactive=interactive,
    )
    target_resolution = resolve_endpoint(
        endpoint="target",
        label="Target",
        args=args,
        app_config=app_config,
        interactive=interactive,
        blocked_profile=source_resolution.profile_name,
    )

    source_conn = None
    target_conn = None
    source_runtime_config = source_resolution.config
    target_runtime_config = target_resolution.config
    try:
        source_runtime_config, source_conn = connect_with_retry(
            "Source", source_resolution, interactive=interactive
        )
        target_runtime_config, target_conn = connect_with_retry(
            "Target", target_resolution, interactive=interactive
        )

        if args.workers < 1:
            console.print("[red]--workers must be >= 1[/red]")
            return 1

        available_schemas = list_user_schemas(source_conn)
        if not available_schemas:
            console.print("[yellow]No user schemas found in source DB.[/yellow]")
            return 0

        requested_schemas = parse_schema_csv(args.schemas)
        if args.all_schemas:
            selected_schemas = available_schemas
        elif requested_schemas:
            missing = [name for name in requested_schemas if name not in available_schemas]
            if missing:
                console.print(
                    f"[red]Unknown schema(s): {', '.join(missing)}. "
                    f"Run `list-schemas` first.[/red]"
                )
                return 1
            selected_schemas = requested_schemas
        elif interactive:
            selected_schemas = questionary.checkbox(
                "Select schemas to migrate",
                choices=available_schemas,
            ).ask()
            if selected_schemas is None:
                raise KeyboardInterrupt
            if not selected_schemas:
                console.print("[yellow]No schemas selected.[/yellow]")
                return 0
        else:
            console.print(
                "[red]No schemas selected. Use --schemas, --all-schemas, or interactive mode.[/red]"
            )
            return 1

        plan = pick_schema_table_plan(
            source_conn=source_conn,
            schemas=list(selected_schemas),
            interactive=interactive,
            all_tables=args.all_tables,
        )
        if not plan:
            console.print("[yellow]Nothing selected. Exiting.[/yellow]")
            return 0

        print_plan(plan)
        if not args.yes and interactive:
            proceed = ask_confirm(
                "Proceed with migration?",
                default=False,
            )
            if not proceed:
                console.print("[yellow]Migration cancelled.[/yellow]")
                return 0

        if args.workers > 1:
            source_conn.close()
            source_conn = None
            target_conn.close()
            target_conn = None
            summary = migrate_parallel(
                source_config=source_runtime_config,
                target_config=target_runtime_config,
                plan=plan,
                include_foreign_keys=args.include_foreign_keys,
                console=console,
                workers=args.workers,
            )
        else:
            summary = migrate(
                source_conn=source_conn,
                target_conn=target_conn,
                plan=plan,
                include_foreign_keys=args.include_foreign_keys,
                console=console,
            )
        return print_summary(summary)
    finally:
        if source_conn is not None:
            source_conn.close()
        if target_conn is not None:
            target_conn.close()


def add_common_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="Path to one JSON config file with source/target/profiles.",
    )
    parser.add_argument(
        "--env-file",
        help="Path to .env file with MOVEX_* or DBM_* variables.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail instead of prompting for missing values.",
    )


def add_endpoint_flags(
    parser: argparse.ArgumentParser,
    endpoint: str,
    include_profile: bool = True,
) -> None:
    title = endpoint.capitalize()
    if include_profile:
        parser.add_argument(
            f"--{endpoint}-profile",
            help=f"Use named profile for {title}.",
        )
    parser.add_argument(f"--{endpoint}-dsn", help=f"{title} PostgreSQL DSN/URL.")
    parser.add_argument(f"--{endpoint}-host", help=f"{title} host.")
    parser.add_argument(f"--{endpoint}-port", type=int, help=f"{title} port.")
    parser.add_argument(f"--{endpoint}-dbname", help=f"{title} database name.")
    parser.add_argument(f"--{endpoint}-user", help=f"{title} database user.")
    parser.add_argument(f"--{endpoint}-password", help=f"{title} password.")
    parser.add_argument(
        f"--{endpoint}-sslmode",
        help=f"{title} sslmode (default: require).",
    )
    parser.add_argument(
        f"--{endpoint}-connect-timeout",
        type=int,
        help=f"{title} connection timeout in seconds.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movex",
        description="MoveX: fast PostgreSQL schema transfer tool and library.",
    )
    subparsers = parser.add_subparsers(dest="command")

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Transfer selected schemas/tables from source to target.",
    )
    add_common_config_flags(migrate_parser)
    add_endpoint_flags(migrate_parser, "source")
    add_endpoint_flags(migrate_parser, "target")
    migrate_parser.add_argument(
        "--schemas",
        help="Comma/space-separated schema names to transfer.",
    )
    migrate_parser.add_argument(
        "--all-schemas",
        action="store_true",
        help="Transfer all user schemas from source.",
    )
    migrate_parser.add_argument(
        "--all-tables",
        action="store_true",
        help="Transfer all tables from selected schemas without per-schema prompts.",
    )
    migrate_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel table migration workers (use 1 for sequential mode).",
    )
    migrate_parser.add_argument(
        "--skip-foreign-keys",
        action="store_false",
        dest="include_foreign_keys",
        help="Do not recreate foreign key constraints.",
    )
    migrate_parser.set_defaults(include_foreign_keys=True)
    migrate_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )

    list_parser = subparsers.add_parser(
        "list-schemas",
        help="List user schemas from source DB.",
    )
    add_common_config_flags(list_parser)
    add_endpoint_flags(list_parser, "source")
    list_parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print schemas as JSON.",
    )

    init_parser = subparsers.add_parser(
        "init-config",
        help="Generate one-file config template (.json or .env).",
    )
    init_parser.add_argument(
        "--format",
        choices=["json", "env"],
        default="json",
        help="Template format to create.",
    )
    init_parser.add_argument(
        "--output",
        help="Output path. Defaults to movex.json or .env.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(argv if argv is not None else sys.argv[1:])

    known_commands = {"migrate", "list-schemas", "init-config"}
    if raw_args and raw_args[0] in {"-h", "--help"}:
        args = parser.parse_args(raw_args)
    elif not raw_args or raw_args[0] not in known_commands:
        args = parser.parse_args(["migrate", *raw_args])
    else:
        args = parser.parse_args(raw_args)

    try:
        if args.command == "init-config":
            return init_config_command(args)
        if args.command == "list-schemas":
            return list_schemas_command(args)
        return migrate_command(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        return 130
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
