from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import tempfile
from typing import Dict, List, Sequence

import psycopg2
from psycopg2 import sql
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .config import DbConfig


@dataclass
class MigrationFailure:
    object_name: str
    step: str
    error: str


@dataclass
class MigrationSummary:
    total_tables: int
    migrated_tables: int
    total_constraints: int
    applied_constraints: int
    total_indexes: int
    applied_indexes: int
    failures: List[MigrationFailure]


@dataclass
class TableTransferResult:
    schema: str
    table: str
    constraints: List[str]
    indexes: List[str]
    failure: MigrationFailure | None = None


def connect(config: DbConfig, autocommit: bool = False) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(**config.to_conn_kwargs())
    conn.autocommit = autocommit
    return conn


def list_user_schemas(conn: psycopg2.extensions.connection) -> List[str]:
    query = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT LIKE 'pg\\_%'
          AND schema_name <> 'information_schema'
        ORDER BY schema_name;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[0] for row in cur.fetchall()]


def list_tables(conn: psycopg2.extensions.connection, schema: str) -> List[str]:
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema,))
        return [row[0] for row in cur.fetchall()]


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def get_table_definition(
    conn: psycopg2.extensions.connection, schema: str, table: str
) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                a.attname AS column_name,
                pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
                NOT a.attnotnull AS is_nullable,
                pg_get_expr(ad.adbin, ad.adrelid) AS column_default
            FROM pg_catalog.pg_attribute a
            LEFT JOIN pg_catalog.pg_attrdef ad
                ON a.attrelid = ad.adrelid
               AND a.attnum = ad.adnum
            WHERE a.attrelid = (
                SELECT c.oid
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relname = %s AND n.nspname = %s
            )
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum;
            """,
            (table, schema),
        )
        columns = cur.fetchall()

        cur.execute(
            """
            SELECT a.attname
            FROM pg_catalog.pg_index i
            JOIN pg_catalog.pg_attribute a
              ON a.attrelid = i.indrelid
             AND a.attnum = ANY(i.indkey)
            JOIN pg_catalog.pg_class c ON i.indrelid = c.oid
            JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
            WHERE c.relname = %s
              AND n.nspname = %s
              AND i.indisprimary
            ORDER BY array_position(i.indkey, a.attnum);
            """,
            (table, schema),
        )
        pk_cols = [r[0] for r in cur.fetchall()]

    col_defs: List[str] = []
    for col_name, type_str, is_nullable, col_default in columns:
        is_seq = False
        if col_default and "nextval" in str(col_default):
            lowered = type_str.lower()
            type_str = (
                "BIGSERIAL"
                if "big" in lowered
                else "SMALLSERIAL"
                if "small" in lowered
                else "SERIAL"
            )
            col_default = None
            is_seq = True

        part = f"{quote_identifier(col_name)} {type_str}"
        if not is_nullable:
            part += " NOT NULL"
        if col_default and not is_seq:
            part += f" DEFAULT {col_default}"
        col_defs.append(part)

    create_sql = (
        f"CREATE TABLE {quote_identifier(schema)}.{quote_identifier(table)} (\n"
        + ",\n".join("  " + col for col in col_defs)
    )
    if pk_cols:
        create_sql += (
            ",\n  PRIMARY KEY ("
            + ",".join(quote_identifier(col) for col in pk_cols)
            + ")"
        )
    create_sql += "\n);"
    return create_sql


def get_additional_constraints(
    conn: psycopg2.extensions.connection,
    schema: str,
    table: str,
    include_foreign_keys: bool,
) -> List[str]:
    query = """
        SELECT
            c.contype,
            c.conname,
            pg_get_constraintdef(c.oid, true) AS condef
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = %s
          AND t.relname = %s
          AND c.contype IN ('u', 'f', 'c', 'x')
        ORDER BY c.contype, c.conname;
    """

    results: List[str] = []
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        for contype, conname, condef in cur.fetchall():
            if contype == "f" and not include_foreign_keys:
                continue
            stmt = (
                f"ALTER TABLE {quote_identifier(schema)}.{quote_identifier(table)} "
                f"ADD CONSTRAINT {quote_identifier(conname)} {condef};"
            )
            results.append(stmt)
    return results


def get_non_constraint_indexes(
    conn: psycopg2.extensions.connection, schema: str, table: str
) -> List[str]:
    query = """
        SELECT pg_get_indexdef(i.oid)
        FROM pg_class t
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_index ix ON ix.indrelid = t.oid
        JOIN pg_class i ON i.oid = ix.indexrelid
        LEFT JOIN pg_constraint c ON c.conindid = i.oid
        WHERE n.nspname = %s
          AND t.relname = %s
          AND c.oid IS NULL
          AND NOT ix.indisprimary
        ORDER BY i.relname;
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        return [row[0] for row in cur.fetchall()]


def ensure_schema(conn: psycopg2.extensions.connection, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )


def drop_table(conn: psycopg2.extensions.connection, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                sql.Identifier(schema), sql.Identifier(table)
            )
        )


def set_triggers(
    conn: psycopg2.extensions.connection, schema: str, table: str, enabled: bool
) -> None:
    action = sql.SQL("ENABLE TRIGGER ALL") if enabled else sql.SQL("DISABLE TRIGGER ALL")
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("ALTER TABLE {}.{} ").format(
                sql.Identifier(schema), sql.Identifier(table)
            )
            + action
        )


def copy_table_data(
    source_conn: psycopg2.extensions.connection,
    target_conn: psycopg2.extensions.connection,
    schema: str,
    table: str,
) -> None:
    copy_out = (
        sql.SQL("COPY {}.{} TO STDOUT WITH CSV QUOTE '\"' ESCAPE '\"'")
        .format(sql.Identifier(schema), sql.Identifier(table))
        .as_string(source_conn)
    )
    copy_in = (
        sql.SQL("COPY {}.{} FROM STDIN WITH CSV QUOTE '\"' ESCAPE '\"'")
        .format(sql.Identifier(schema), sql.Identifier(table))
        .as_string(target_conn)
    )

    # Spool in memory first and spill to disk only for larger tables.
    with tempfile.SpooledTemporaryFile(max_size=128 * 1024 * 1024, mode="w+t") as buffer:
        with source_conn.cursor() as src_cur:
            src_cur.copy_expert(copy_out, buffer)
        buffer.seek(0)
        with target_conn.cursor() as tgt_cur:
            tgt_cur.copy_expert(copy_in, buffer)


def sync_serial_sequences(
    conn: psycopg2.extensions.connection, schema: str, table: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
              AND column_default LIKE 'nextval(%'
            ORDER BY ordinal_position;
            """,
            (schema, table),
        )
        serial_columns = [row[0] for row in cur.fetchall()]

    for column in serial_columns:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_get_serial_sequence(%s, %s);",
                (f"{schema}.{table}", column),
            )
            seq_name_row = cur.fetchone()
            if not seq_name_row or not seq_name_row[0]:
                continue
            seq_name = seq_name_row[0]

            cur.execute(
                sql.SQL("SELECT COALESCE(MAX({}), 0) FROM {}.{}").format(
                    sql.Identifier(column),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            max_value = int(cur.fetchone()[0])

            if max_value <= 0:
                cur.execute("SELECT setval(%s, 1, false);", (seq_name,))
            else:
                cur.execute("SELECT setval(%s, %s, true);", (seq_name, max_value))


def build_plan_for_schemas(
    source_conn: psycopg2.extensions.connection,
    schemas: Sequence[str],
) -> Dict[str, List[str]]:
    plan: Dict[str, List[str]] = {}
    for schema in schemas:
        tables = list_tables(source_conn, schema)
        if tables:
            plan[schema] = tables
    return plan


def optimize_target_session(conn: psycopg2.extensions.connection) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit TO OFF;")
        conn.commit()
    except Exception:
        conn.rollback()


def _migrate_single_table(
    source_config: DbConfig,
    target_config: DbConfig,
    schema: str,
    table_name: str,
    include_foreign_keys: bool,
) -> TableTransferResult:
    source_conn = None
    target_conn = None
    step = f"{schema}.{table_name}"
    try:
        source_conn = connect(source_config, autocommit=False)
        target_conn = connect(target_config, autocommit=False)
        optimize_target_session(target_conn)

        drop_table(target_conn, schema, table_name)
        create_stmt = get_table_definition(source_conn, schema, table_name)
        with target_conn.cursor() as cur:
            cur.execute(create_stmt)
        target_conn.commit()

        set_triggers(target_conn, schema, table_name, enabled=False)
        copy_table_data(source_conn, target_conn, schema, table_name)
        sync_serial_sequences(target_conn, schema, table_name)
        set_triggers(target_conn, schema, table_name, enabled=True)
        target_conn.commit()

        constraints = get_additional_constraints(
            source_conn,
            schema,
            table_name,
            include_foreign_keys=include_foreign_keys,
        )
        indexes = get_non_constraint_indexes(source_conn, schema, table_name)
        return TableTransferResult(
            schema=schema,
            table=table_name,
            constraints=constraints,
            indexes=indexes,
        )
    except Exception as exc:
        if target_conn is not None:
            target_conn.rollback()
        return TableTransferResult(
            schema=schema,
            table=table_name,
            constraints=[],
            indexes=[],
            failure=MigrationFailure(
                object_name=step,
                step="table migration",
                error=str(exc),
            ),
        )
    finally:
        if source_conn is not None:
            source_conn.close()
        if target_conn is not None:
            target_conn.close()


def _apply_constraints_and_indexes(
    target_conn: psycopg2.extensions.connection,
    post_constraints: Dict[tuple[str, str], List[str]],
    post_indexes: Dict[tuple[str, str], List[str]],
    failures: List[MigrationFailure],
    console: Console,
) -> tuple[int, int]:
    applied_constraints = 0
    applied_indexes = 0
    total_constraints = sum(len(v) for v in post_constraints.values())
    total_indexes = sum(len(v) for v in post_indexes.values())

    with Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=42, complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        constraint_task = progress.add_task(
            "Applying constraints", total=total_constraints or 1
        )
        index_task = progress.add_task("Applying indexes", total=total_indexes or 1)

        for (schema, table_name), statements in post_constraints.items():
            for statement in statements:
                try:
                    with target_conn.cursor() as cur:
                        cur.execute(statement)
                    target_conn.commit()
                    applied_constraints += 1
                except Exception as exc:
                    target_conn.rollback()
                    failures.append(
                        MigrationFailure(
                            object_name=f"{schema}.{table_name}",
                            step="constraint creation",
                            error=str(exc),
                        )
                    )
                finally:
                    progress.advance(constraint_task)

        for (schema, table_name), statements in post_indexes.items():
            for statement in statements:
                try:
                    with target_conn.cursor() as cur:
                        cur.execute(statement)
                    target_conn.commit()
                    applied_indexes += 1
                except Exception as exc:
                    target_conn.rollback()
                    failures.append(
                        MigrationFailure(
                            object_name=f"{schema}.{table_name}",
                            step="index creation",
                            error=str(exc),
                        )
                    )
                finally:
                    progress.advance(index_task)

    return applied_constraints, applied_indexes


def migrate(
    source_conn: psycopg2.extensions.connection,
    target_conn: psycopg2.extensions.connection,
    plan: Dict[str, List[str]],
    include_foreign_keys: bool,
    console: Console,
) -> MigrationSummary:
    failures: List[MigrationFailure] = []
    post_constraints: Dict[tuple[str, str], List[str]] = {}
    post_indexes: Dict[tuple[str, str], List[str]] = {}

    total_tables = sum(len(tables) for tables in plan.values())
    if total_tables == 0:
        return MigrationSummary(
            total_tables=0,
            migrated_tables=0,
            total_constraints=0,
            applied_constraints=0,
            total_indexes=0,
            applied_indexes=0,
            failures=[],
        )

    optimize_target_session(target_conn)

    migrated_tables = 0
    applied_constraints = 0
    applied_indexes = 0

    with Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=42, complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        table_task = progress.add_task("Migrating tables", total=total_tables)

        for schema, tables in plan.items():
            ensure_schema(target_conn, schema)
            target_conn.commit()

            for table_name in tables:
                step = f"{schema}.{table_name}"
                progress.update(table_task, description=f"Migrating {step}")
                try:
                    drop_table(target_conn, schema, table_name)
                    create_stmt = get_table_definition(source_conn, schema, table_name)
                    with target_conn.cursor() as cur:
                        cur.execute(create_stmt)
                    target_conn.commit()

                    set_triggers(target_conn, schema, table_name, enabled=False)
                    copy_table_data(source_conn, target_conn, schema, table_name)
                    sync_serial_sequences(target_conn, schema, table_name)
                    set_triggers(target_conn, schema, table_name, enabled=True)
                    target_conn.commit()
                    migrated_tables += 1

                    post_constraints[(schema, table_name)] = get_additional_constraints(
                        source_conn,
                        schema,
                        table_name,
                        include_foreign_keys=include_foreign_keys,
                    )
                    post_indexes[(schema, table_name)] = get_non_constraint_indexes(
                        source_conn, schema, table_name
                    )
                except Exception as exc:
                    target_conn.rollback()
                    failures.append(
                        MigrationFailure(
                            object_name=step,
                            step="table migration",
                            error=str(exc),
                        )
                    )
                finally:
                    progress.advance(table_task)

    new_applied_constraints, new_applied_indexes = _apply_constraints_and_indexes(
        target_conn=target_conn,
        post_constraints=post_constraints,
        post_indexes=post_indexes,
        failures=failures,
        console=console,
    )
    applied_constraints += new_applied_constraints
    applied_indexes += new_applied_indexes

    return MigrationSummary(
        total_tables=total_tables,
        migrated_tables=migrated_tables,
        total_constraints=sum(len(v) for v in post_constraints.values()),
        applied_constraints=applied_constraints,
        total_indexes=sum(len(v) for v in post_indexes.values()),
        applied_indexes=applied_indexes,
        failures=failures,
    )


def migrate_parallel(
    source_config: DbConfig,
    target_config: DbConfig,
    plan: Dict[str, List[str]],
    include_foreign_keys: bool,
    console: Console,
    workers: int = 4,
) -> MigrationSummary:
    total_tables = sum(len(tables) for tables in plan.values())
    if total_tables == 0:
        return MigrationSummary(
            total_tables=0,
            migrated_tables=0,
            total_constraints=0,
            applied_constraints=0,
            total_indexes=0,
            applied_indexes=0,
            failures=[],
        )

    if workers <= 1:
        source_conn = None
        target_conn = None
        try:
            source_conn = connect(source_config, autocommit=False)
            target_conn = connect(target_config, autocommit=False)
            return migrate(
                source_conn=source_conn,
                target_conn=target_conn,
                plan=plan,
                include_foreign_keys=include_foreign_keys,
                console=console,
            )
        finally:
            if source_conn is not None:
                source_conn.close()
            if target_conn is not None:
                target_conn.close()

    failures: List[MigrationFailure] = []
    post_constraints: Dict[tuple[str, str], List[str]] = {}
    post_indexes: Dict[tuple[str, str], List[str]] = {}
    migrated_tables = 0
    requested_workers = max(1, workers)
    worker_count = min(requested_workers, total_tables)

    target_init_conn = None
    try:
        target_init_conn = connect(target_config, autocommit=False)
        optimize_target_session(target_init_conn)
        for schema in plan:
            ensure_schema(target_init_conn, schema)
        target_init_conn.commit()
    finally:
        if target_init_conn is not None:
            target_init_conn.close()

    all_tables = [(schema, table_name) for schema, tables in plan.items() for table_name in tables]

    with Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=42, complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        table_task = progress.add_task(
            f"Migrating tables with {worker_count} workers",
            total=total_tables,
        )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _migrate_single_table,
                    source_config,
                    target_config,
                    schema,
                    table_name,
                    include_foreign_keys,
                ): (schema, table_name)
                for schema, table_name in all_tables
            }

            for future in as_completed(future_map):
                schema, table_name = future_map[future]
                progress.update(table_task, description=f"Migrating {schema}.{table_name}")
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(
                        MigrationFailure(
                            object_name=f"{schema}.{table_name}",
                            step="table migration",
                            error=str(exc),
                        )
                    )
                else:
                    if result.failure is not None:
                        failures.append(result.failure)
                    else:
                        migrated_tables += 1
                        post_constraints[(result.schema, result.table)] = result.constraints
                        post_indexes[(result.schema, result.table)] = result.indexes
                finally:
                    progress.advance(table_task)

    target_apply_conn = None
    applied_constraints = 0
    applied_indexes = 0
    try:
        target_apply_conn = connect(target_config, autocommit=False)
        optimize_target_session(target_apply_conn)
        applied_constraints, applied_indexes = _apply_constraints_and_indexes(
            target_conn=target_apply_conn,
            post_constraints=post_constraints,
            post_indexes=post_indexes,
            failures=failures,
            console=console,
        )
    finally:
        if target_apply_conn is not None:
            target_apply_conn.close()

    return MigrationSummary(
        total_tables=total_tables,
        migrated_tables=migrated_tables,
        total_constraints=sum(len(v) for v in post_constraints.values()),
        applied_constraints=applied_constraints,
        total_indexes=sum(len(v) for v in post_indexes.values()),
        applied_indexes=applied_indexes,
        failures=failures,
    )
