"""the partitioned call record table takes the name `callrecord`

Revision ID: 0117
Revises: 0116
Create Date: 2026-11-18

After this revision `callrecord` is the monthly-partitioned, 64-bit table that 0116 built, and the
table it replaces is `callrecord_legacy`, which nothing writes any more. No reader changes: the
name, the columns and the index names are the same, and the new table already holds the history
the readers use.

**The history is copied before the swap, not after it.** Several readers count "today" or rebuild
a recent window from call records (the trial allowance, the spend recount, catalog statistics, the
usage rollup); a table that started with only its newest rows would give them wrong answers.

- A small table (at most `_INLINE_IDS` ids from first to last) is copied here, whole.
- A larger one must be copied first by `treg-worker admin callrecord-backfill --days N`, which
  runs for as long as it needs beside live traffic. This revision refuses, changing nothing,
  until the partitioned table reaches back `_MIN_HISTORY_DAYS` days or to the first row. To copy
  before the release is deployed, apply the earlier revisions alone: `alembic upgrade 0116`.

The steps, of which only the third blocks anyone:

1. **Catch up** (no lock; batches by id range): the rows written since the last copy, repeated
   until a few hundred ids are left. Month partitions are created as needed.
2. **Late verdicts** (no lock): `hit` and `verdict` are written after the row, when an async task
   ends, at most a day later. Rows of the last `_RESYNC_DAYS` days take the live table's values.
   The id this step reached is noted on the new table (a table comment), and a retry within
   `_RESYNC_REUSE` resyncs only the rows above it, so a retry does not repeat days of reads.
   Then step 1 runs again: the rows written while this step ran are copied now, without a lock,
   so the swap below copies at most about `_SWAP_ROWS` new rows however long this step took.
3. **The swap** (one transaction; ACCESS EXCLUSIVE on both tables, held for the statements below
   only): the rows written since step 2 are copied, then the tables and their indexes trade
   names, the id sequence moves to the new table, and the old table's grants are repeated on it.
   Audit inserts wait for this transaction and then land in the new table. `env.py`'s
   `lock_timeout` bounds the wait for the lock, and `maintenance` retries the upgrade; a retry
   repeats steps 1 and 2 over the difference only. While the lock request waits, statements on
   `callrecord` wait behind it: for a long statement to end, or, when autovacuum is working on
   the table, for `deadlock_timeout` (one second by default) until autovacuum yields.
4. **After the swap**: rows of teams that were deleted while the copy ran are removed, team by
   team through the `(org_id, id)` index.

An empty `callrecord` (a new database) is dropped instead of kept as `callrecord_legacy`. Failure
evidence blanked in the old table after its row was copied is blanked again by the next
`treg-worker admin purge-evidence` run, which selects by age.

A connection that prepared a statement returning `callrecord.id` before the swap gets
"cached statement plan is invalid" once, because the column type changed; the driver then
prepares it again. The audit writer retries a failed batch row by row, so it loses nothing.

SQLite has no partitions and its ids are 64-bit already. There the revision only drops the
foreign key to `org`, so both databases agree with the model.

Rollback floor: older code runs on the partitioned table unchanged, so a code rollback does not
need the downgrade. The downgrade puts a plain `integer` table back under the name `callrecord`
and copies into it every row the partitioned table holds, in one statement; it is meant for a
small database and fails with "integer out of range" once an id has passed 2,147,483,647.
"""
from collections.abc import Sequence
from datetime import date, datetime, timedelta

import sqlalchemy as sa
from alembic import op

revision: str = "0117"
down_revision: str | Sequence[str] | None = "0116"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
contract = True  # replaces a table; see the rollback floor note

LIVE = "callrecord"
NEW = "callrecord_partitioned"
LEGACY = "callrecord_legacy"

_INLINE_IDS = 1_000_000
# The longest look back of a reader that rebuilds from call records (the usage rollup's history).
_MIN_HISTORY_DAYS = 31
_RESYNC_DAYS = 2
_BATCH_ROWS = 20_000
# The copy repeats until at most this many ids are left, so the swap copies few rows under its lock.
_SWAP_ROWS = 500
_COPY_PASSES = 50
# Ids are handed out before their rows commit. A copy that continues above the last copied id
# starts this many ids below it, so a row that committed late is still found.
_COMMIT_MARGIN = 20_000
_MONTHS_AHEAD = 3
# A retry of this revision within this long resyncs verdicts only above the id the last attempt
# reached; a later one resyncs the whole window again.
_RESYNC_REUSE = timedelta(minutes=30)
_RESYNC_NOTE = "treg 0117: verdicts resynced up to id "

_SQLITE_FK_NAME = "fk_callrecord_org_id_org"
_NAMING_CONVENTION = {"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"}

_REFUSAL = (
    "callrecord is too large to copy during an upgrade, and callrecord_partitioned does not hold "
    "its recent history yet ({found}). Nothing was changed by this revision. Run "
    "`treg-worker admin callrecord-backfill --days N` with N of at least {days}, then upgrade again. "
    "See docs/context/architecture/data-model.md, \"monthly partitions and a 64-bit id\"."
)


def _scalar(bind, sql: str, **params):
    return bind.execute(sa.text(sql), params).scalar()


def _relkind(bind, table: str) -> str | None:
    return _scalar(bind, "SELECT relkind::text FROM pg_class WHERE oid = to_regclass(:name)", name=table)


def _columns(bind) -> str:
    names = bind.execute(sa.text(
        "SELECT attname FROM pg_attribute WHERE attrelid = to_regclass(:name) "
        "AND attnum > 0 AND NOT attisdropped ORDER BY attnum"), {"name": LIVE}).scalars().all()
    return ", ".join(f'"{name}"' for name in names)


def _index_names(bind, table: str) -> list[str]:
    return list(bind.execute(sa.text(
        "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE i.indrelid = to_regclass(:name) ORDER BY c.relname"), {"name": table}).scalars())


def _now(bind) -> datetime:
    return _scalar(bind, "SELECT now() AT TIME ZONE 'UTC'")


def _start(bind) -> int:
    """The id the copy continues above, or a refusal when the history is not there."""
    first, last = bind.execute(sa.text(f"SELECT min(id), max(id) FROM {LIVE}")).one()
    if first is None:
        return 0
    copied_first, copied_last = bind.execute(sa.text(f"SELECT min(id), max(id) FROM {NEW}")).one()
    if copied_first is None:
        if last - first >= _INLINE_IDS:
            raise RuntimeError(_REFUSAL.format(found="it is empty", days=_MIN_HISTORY_DAYS))
        return first - 1
    left_out = _scalar(bind, f"SELECT created_at FROM {LIVE} WHERE id < :floor ORDER BY id DESC LIMIT 1",
                       floor=copied_first)
    if left_out is not None and left_out > _now(bind) - timedelta(days=_MIN_HISTORY_DAYS):
        raise RuntimeError(_REFUSAL.format(found=f"the copy reaches back to {left_out:%Y-%m-%d} only",
                                           days=_MIN_HISTORY_DAYS))
    return max(copied_last - _COMMIT_MARGIN, copied_first - 1)


def _create_partitions(bind, after: int) -> None:
    """One partition per month, from the month of the first row still to copy to a few ahead."""
    today = _now(bind).date()
    oldest = _scalar(bind, f"SELECT created_at FROM {LIVE} WHERE id > :after ORDER BY id LIMIT 1",
                     after=after)
    first = min(oldest.date(), today) if oldest else today
    start = first.year * 12 + first.month - 1
    count = (today.year - first.year) * 12 + today.month - first.month + 1 + _MONTHS_AHEAD
    months = [date(index // 12, index % 12 + 1, 1) for index in range(start, start + count + 1)]
    for begin, end in zip(months, months[1:]):
        bind.execute(sa.text(
            f"CREATE TABLE IF NOT EXISTS callrecord_y{begin:%Y}m{begin:%m} PARTITION OF {NEW} "
            f"FOR VALUES FROM ('{begin}') TO ('{end}')"))


def _catch_up(bind, columns: str, reached: int) -> int:
    """Copy the ids above `reached` in batches and return the id the copy reached. Each pass
    copies what was written during the pass before it, so the passes shrink to almost nothing."""
    copy = sa.text(f"INSERT INTO {NEW} ({columns}) SELECT {columns} FROM {LIVE} "
                   "WHERE id > :after AND id <= :upto ON CONFLICT DO NOTHING")
    top = _scalar(bind, f"SELECT max(id) FROM {LIVE}") or 0
    for _ in range(_COPY_PASSES):
        for after in range(reached, top, _BATCH_ROWS):
            bind.execute(copy, {"after": after, "upto": min(after + _BATCH_ROWS, top)})
        reached = top
        top = _scalar(bind, f"SELECT max(id) FROM {LIVE}") or 0
        if top - reached <= _SWAP_ROWS:
            break
    return reached


def _first_id_at(bind, since: datetime, top: int) -> int:
    """The smallest id whose row is at or after `since`, by bisection on the primary key: ids and
    `created_at` rise together, and `callrecord` has no index on `created_at` alone."""
    low, high = 0, top + 1
    while low < high:
        mid = (low + high) // 2
        row = bind.execute(sa.text(
            f"SELECT id, created_at FROM {LIVE} WHERE id >= :mid ORDER BY id LIMIT 1"), {"mid": mid}).first()
        if row is None or row.created_at >= since:
            high = mid
        else:
            low = row.id + 1
    return low


def _resynced_up_to(bind) -> int:
    """The id an attempt within `_RESYNC_REUSE` resynced up to, from the new table's comment;
    0 when there is none."""
    note = _scalar(bind, "SELECT obj_description(to_regclass(:name), 'pg_class')", name=NEW) or ""
    upto, _, at = note.removeprefix(_RESYNC_NOTE).partition(" at ")
    if not note.startswith(_RESYNC_NOTE):
        return 0
    try:
        upto_id, when = int(upto), datetime.fromisoformat(at)
    except ValueError:
        return 0
    return upto_id if _now(bind) - when <= _RESYNC_REUSE else 0


def _resync_verdicts(bind, reached: int) -> None:
    resync = sa.text(
        f"UPDATE {NEW} n SET hit = o.hit, verdict = o.verdict FROM {LIVE} o "
        "WHERE o.id > :after AND o.id <= :upto AND n.id = o.id AND n.created_at = o.created_at "
        "AND (n.hit IS DISTINCT FROM o.hit OR n.verdict IS DISTINCT FROM o.verdict)")
    first = _first_id_at(bind, _now(bind) - timedelta(days=_RESYNC_DAYS), reached)
    for after in range(max(first - 1, _resynced_up_to(bind)), reached, _BATCH_ROWS):
        bind.execute(resync, {"after": after, "upto": min(after + _BATCH_ROWS, reached)})
    # COMMENT takes no bind parameters; the note is an integer and a timestamp this code wrote.
    bind.execute(sa.text(
        f"COMMENT ON TABLE {NEW} IS '{_RESYNC_NOTE}{int(reached)} at {_now(bind).isoformat()}'"))


def _rename(bind, table: str, to: str) -> None:
    """Rename a table and its indexes. Index names carry their table's name: `callrecord` owns
    `ix_callrecord_*`, the partitioned table before the swap `ix_callrecord_partitioned_*`, and
    the legacy table `ix_callrecord_*_legacy`."""
    for name in _index_names(bind, table):
        if name == f"{table}_pkey":
            renamed = f"{to}_pkey"
        elif to == LEGACY:
            renamed = f"{name}_legacy"
        elif table == LEGACY:
            renamed = name.removesuffix("_legacy")
        else:
            renamed = name.replace(f"ix_{table}_", f"ix_{to}_", 1)
        bind.execute(sa.text(f'ALTER INDEX "{name}" RENAME TO "{renamed}"'))
    bind.execute(sa.text(f"ALTER TABLE {table} RENAME TO {to}"))


def _repeat_grants(bind, source: str, target: str) -> None:
    grants = bind.execute(sa.text("""
        SELECT CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE quote_ident(pg_get_userbyid(a.grantee)) END,
               a.privilege_type, NULL::text
        FROM pg_class c, aclexplode(c.relacl) a
        WHERE c.oid = to_regclass(:source) AND a.grantee <> c.relowner
        UNION ALL
        SELECT CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE quote_ident(pg_get_userbyid(a.grantee)) END,
               a.privilege_type, quote_ident(t.attname)
        FROM pg_attribute t, aclexplode(t.attacl) a
        WHERE t.attrelid = to_regclass(:source) AND t.attnum > 0 AND NOT t.attisdropped
    """), {"source": source}).all()
    for grantee, privilege, column in grants:
        what = f"{privilege} ({column})" if column else privilege
        bind.execute(sa.text(f"GRANT {what} ON {target} TO {grantee}"))


def _swap(bind, columns: str, reached: int) -> None:
    # The new table first: autovacuum may be analyzing the rows just copied, and a lock request
    # waits `deadlock_timeout` before autovacuum gives way. Nothing else uses that table, so that
    # wait holds nobody up. The same wait can precede the lock on the live table.
    bind.execute(sa.text(f"LOCK TABLE {NEW} IN ACCESS EXCLUSIVE MODE"))
    bind.execute(sa.text(f"LOCK TABLE {LIVE} IN ACCESS EXCLUSIVE MODE"))
    # ON CONFLICT, not an anti-join: the new table has no statistics yet, and a row already
    # copied costs one probe of the primary key.
    bind.execute(sa.text(
        f"INSERT INTO {NEW} ({columns}) SELECT {columns} FROM {LIVE} WHERE id > :after "
        "ON CONFLICT DO NOTHING"), {"after": reached - _COMMIT_MARGIN})
    sequence = _scalar(bind, f"SELECT pg_get_serial_sequence('{LIVE}', 'id')")
    empty = _scalar(bind, f"SELECT NOT EXISTS (SELECT 1 FROM {LIVE})")
    _repeat_grants(bind, LIVE, NEW)
    bind.execute(sa.text(f"COMMENT ON TABLE {NEW} IS NULL"))
    _rename(bind, LIVE, LEGACY)
    _rename(bind, NEW, LIVE)
    bind.execute(sa.text(f"ALTER SEQUENCE {sequence} OWNED BY {LIVE}.id"))
    if empty:
        bind.execute(sa.text(f"DROP TABLE {LEGACY}"))
    else:
        bind.execute(sa.text(f"ALTER TABLE {LEGACY} ALTER COLUMN id DROP DEFAULT"))


def _delete_rows_of_deleted_teams(bind) -> None:
    """Postgres cannot skip through an index, so the distinct teams are walked one probe each;
    an anti-join against `org` would read every call record."""
    gone = bind.execute(sa.text(f"""
        WITH RECURSIVE team AS (
            SELECT min(org_id) AS org_id FROM {LIVE}
            UNION ALL
            SELECT (SELECT min(org_id) FROM {LIVE} WHERE org_id > team.org_id)
            FROM team WHERE team.org_id IS NOT NULL
        )
        SELECT team.org_id FROM team
        WHERE team.org_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM org WHERE org.id = team.org_id)
    """)).scalars().all()
    for org_id in gone:
        bind.execute(sa.text(f"DELETE FROM {LIVE} WHERE org_id = :org_id"), {"org_id": org_id})


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        with op.batch_alter_table(LIVE, naming_convention=_NAMING_CONVENTION) as batch:
            batch.drop_constraint(_SQLITE_FK_NAME, type_="foreignkey")
        return
    if _relkind(bind, LIVE) != "p":
        columns = _columns(bind)
        with op.get_context().autocommit_block():
            start = _start(bind)
            _create_partitions(bind, start)
            reached = _catch_up(bind, columns, start)
            _resync_verdicts(bind, reached)
            # The resync takes as long as it takes; what was written meanwhile is copied here,
            # without a lock, not by the swap under it.
            reached = _catch_up(bind, columns, reached)
        _swap(op.get_bind(), columns, reached)
    with op.get_context().autocommit_block():
        _delete_rows_of_deleted_teams(op.get_bind())


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        with op.batch_alter_table(LIVE, naming_convention=_NAMING_CONVENTION) as batch:
            batch.create_foreign_key(_SQLITE_FK_NAME, "org", ["org_id"], ["id"])
        return
    if _relkind(bind, LIVE) != "p":
        return
    columns = _columns(bind)
    sequence = _scalar(bind, f"SELECT pg_get_serial_sequence('{LIVE}', 'id')")
    bind.execute(sa.text(f"LOCK TABLE {LIVE} IN ACCESS EXCLUSIVE MODE"))
    if _relkind(bind, LEGACY) is None:
        bind.execute(sa.text(f"CREATE TABLE {LEGACY} (LIKE {LIVE} INCLUDING DEFAULTS)"))
        bind.execute(sa.text(f"ALTER TABLE {LEGACY} ALTER COLUMN id TYPE integer"))
        bind.execute(sa.text(f"ALTER TABLE {LEGACY} ADD CONSTRAINT {LEGACY}_pkey PRIMARY KEY (id)"))
        bind.execute(sa.text(f"ALTER TABLE {LEGACY} ADD CONSTRAINT callrecord_org_id_fkey "
                             "FOREIGN KEY (org_id) REFERENCES org (id)"))
        for name in _index_names(bind, LIVE):
            if name != f"{LIVE}_pkey":
                definition = _scalar(bind, "SELECT pg_get_indexdef(to_regclass(:name))", name=name)
                bind.execute(sa.text(
                    definition.replace(f"INDEX {name} ON ONLY ", f"INDEX {name}_legacy ON ", 1)
                    .replace(f".{LIVE} ", f".{LEGACY} ", 1)))
    bind.execute(sa.text(
        f"INSERT INTO {LEGACY} ({columns}) SELECT {columns} FROM {LIVE} c "
        f"WHERE NOT EXISTS (SELECT 1 FROM {LEGACY} l WHERE l.id = c.id)"))
    _repeat_grants(bind, LIVE, LEGACY)
    _rename(bind, LIVE, NEW)
    _rename(bind, LEGACY, LIVE)
    bind.execute(sa.text(f"ALTER TABLE {LIVE} ALTER COLUMN id SET DEFAULT nextval('{sequence}')"))
    bind.execute(sa.text(f"ALTER SEQUENCE {sequence} OWNED BY {LIVE}.id"))
