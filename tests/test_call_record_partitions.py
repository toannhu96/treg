"""`callrecord` becomes a monthly-partitioned table with a 64-bit id (alembic 0115 to 0117).

The PostgreSQL cases run against a real server (`TREG_TEST_DB_URL`, the Postgres CI job): the swap
keeps ids, cursors, grants and concurrent audit writes; a large table is swapped only after its
history was copied; month partitions are created without waiting for a writer and dropped by the
retention setting. On SQLite the same code is a no-op.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
from datetime import timedelta
from importlib import import_module

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import func, select, text

from treg import audit, worker
from treg.application import call_record_partitions as partitions
from treg.config import get_settings
from treg.domain.governance.teams import cascade_delete_org
from treg.infra import db
from treg.models import CallRecord, CallUsageCursor, Org
from treg.timeutil import utcnow_naive
from tests.test_alembic_baseline import _drop_everything, _upgrade_to


swap = import_module("treg.alembic.versions.0117_call_record_swap")
real_backfill = partitions.backfill
INT4_MAX = 2_147_483_647

_ROWS = text("""
    INSERT INTO callrecord (org_id, user_email, tool_name, method, path, status_code, kind, client,
                            call_ref, budget_dim, budget_val, created_at, cached)
    SELECT :org_id, 'member@example.test', 'echo', 'GET', '/', 200, 'call', '', 'ref-' || n, '', '',
           CAST(:newest AS timestamp) - (:count - n) * CAST(:step AS interval), false
    FROM generate_series(1, :count) AS n
""")


async def _swap_in_process() -> None:
    """Run 0117 here, not through Alembic's own loader, so a test can change its constants."""
    def run(connection):
        context = MigrationContext.configure(connection)
        with context.begin_transaction(), Operations.context(context):
            swap.upgrade()

    await _upgrade_to("0116")
    async with db._engine.connect() as connection:
        await connection.run_sync(run)
        await connection.commit()
    await db._engine.dispose()   # as after a deploy: no connection outlives the swap


async def _scalar(sql: str, **params):
    async with db.session_maker() as session:
        return (await session.execute(text(sql), params)).scalar()


@pytest.fixture
async def before_swap(monkeypatch):
    """A database at 0115 with one team and 40 calls, one per hour, the newest a minute old."""
    if db._engine.dialect.name != "postgresql":
        pytest.skip("requires an isolated PostgreSQL database via TREG_TEST_DB_URL")
    await audit.drain()
    await _drop_everything()
    try:
        await _upgrade_to("0115")
        async with db.session_maker() as session:
            session.add(org := Org(name="swap", slug="swap"))
            await session.flush()
            org_id = org.id
            await session.execute(_ROWS, {"org_id": org_id, "count": 40, "step": timedelta(hours=1),
                                          "newest": utcnow_naive() - timedelta(minutes=1)})
            await session.commit()
        yield org_id
    finally:
        await audit.drain()
        await _drop_everything()
        await db.reset_db()


async def test_swap_keeps_every_id_and_lets_ids_pass_32_bits(before_swap):
    async with db.session_maker() as session:
        await session.execute(text("DROP ROLE IF EXISTS callrecord_reader"))
        await session.execute(text("CREATE ROLE callrecord_reader"))
        await session.execute(text("GRANT SELECT (id, endpoint_id) ON callrecord TO callrecord_reader"))
        await session.commit()
    try:
        await _upgrade_to("head")

        assert await _scalar("SELECT relkind::text FROM pg_class WHERE oid = 'callrecord'::regclass") == "p"
        assert await _scalar("SELECT count(*) FROM callrecord_legacy") == 40
        async with db.session_maker() as session:
            assert (await session.execute(select(CallRecord.id).order_by(CallRecord.id))).scalars().all() \
                == list(range(1, 41))
            names = set((await session.execute(text(
                "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE i.indrelid = 'callrecord'::regclass"))).scalars())
            assert names == {index.name for index in CallRecord.__table__.indexes} | {"callrecord_pkey"}
            assert (await session.execute(text(
                "SELECT has_column_privilege('callrecord_reader', 'callrecord', 'endpoint_id', 'SELECT')"
                " AND NOT has_column_privilege('callrecord_reader', 'callrecord', 'user_email', 'SELECT')"
            ))).scalar_one()

            # The id continues from the old table's sequence, then passes the 32-bit range.
            session.add(next_row := CallRecord(user_email="a@example.test", tool_name="echo",
                                               method="GET", path="/", status_code=200))
            await session.flush()
            assert next_row.id == 41
            await session.execute(text(
                "SELECT setval(pg_get_serial_sequence('callrecord', 'id'), :value)"), {"value": INT4_MAX})
            session.add(wide := CallRecord(user_email="a@example.test", tool_name="echo",
                                           method="GET", path="/", status_code=200))
            session.add(CallUsageCursor(id="callrecord", cursor_id=INT4_MAX + 1, history_from="2026-01-01"))
            await session.commit()
            assert wide.id == INT4_MAX + 1
            assert (await session.get(CallRecord, INT4_MAX + 1)).tool_name == "echo"
    finally:
        async with db.session_maker() as session:
            await session.execute(text("DROP OWNED BY callrecord_reader"))
            await session.execute(text("DROP ROLE callrecord_reader"))
            await session.commit()


async def test_audit_rows_written_during_the_swap_all_land_in_the_new_table(before_swap):
    """The writer's connection prepared its INSERT against the old table. The swap changes the
    type of the returned id under it, and the batch retry absorbs the one refused statement."""
    written = 0
    upgrade = asyncio.create_task(_upgrade_to("head"))
    while not upgrade.done():
        audit.record_call(org_id=before_swap, user_email="w@example.test", tool_name="during-swap",
                          method="GET", path="/", status_code=200)
        written += 1
        await asyncio.sleep(0.002)
    await upgrade
    for _ in range(50):
        audit.record_call(org_id=before_swap, user_email="w@example.test", tool_name="during-swap",
                          method="GET", path="/", status_code=200)
        written += 1
    await audit.drain()

    assert await _scalar("SELECT count(*) FROM callrecord WHERE tool_name = 'during-swap'") == written
    assert await _scalar("SELECT count(*) FROM callrecord") == 40 + written


async def test_a_large_table_is_not_swapped_until_its_history_was_copied(before_swap, monkeypatch):
    """Readers count "today" and rebuild recent windows from call records, so a large table must
    not start with only its newest rows. Nothing changes until the copy reaches back far enough."""
    monkeypatch.setattr(swap, "_INLINE_IDS", 10)
    with pytest.raises(RuntimeError, match="callrecord-backfill"):
        await _swap_in_process()
    assert await _scalar("SELECT relkind::text FROM pg_class WHERE oid = 'callrecord'::regclass") == "r"

    # One day of 40 hourly rows is copied; the swap asks for more than that.
    copied = await partitions.backfill(1, batch_rows=10, pause=0)
    assert copied["phase"] == "before_swap" and copied["done"] is True
    with pytest.raises(RuntimeError, match="the copy reaches back to"):
        await _swap_in_process()

    # Reaching the first row is enough, however young that row is.
    await partitions.backfill(31, batch_rows=10, pause=0)
    assert await _scalar("SELECT count(*) FROM callrecord_partitioned") == 40
    await _swap_in_process()
    assert await _scalar("SELECT relkind::text FROM pg_class WHERE oid = 'callrecord'::regclass") == "p"
    assert await _scalar("SELECT count(*) FROM callrecord") == 40


async def test_rows_written_or_changed_after_the_copy_reach_the_new_table(before_swap, monkeypatch):
    """Between the copy and the swap the old table is still live: new calls arrive, an async call
    gets its verdict, and a team is deleted."""
    monkeypatch.setattr(swap, "_INLINE_IDS", 10)
    monkeypatch.setattr(swap, "_COMMIT_MARGIN", 2)
    monkeypatch.setattr(partitions, "COMMIT_MARGIN", 2)
    await _upgrade_to("0116")
    assert (await partitions.backfill(31, batch_rows=10, pause=0))["copied"] == 40

    async with db.session_maker() as session:
        session.add(gone := Org(name="gone", slug="gone"))
        await session.flush()
        await session.execute(_ROWS, {"org_id": gone.id, "count": 3, "step": timedelta(seconds=1),
                                      "newest": utcnow_naive()})
        await session.commit()
        gone_id = gone.id
    # A second run copies only what was written since the first.
    assert (await partitions.backfill(31, batch_rows=10, pause=0))["copied"] == 3

    async with db.session_maker() as session:
        await session.execute(text("UPDATE callrecord SET hit = true, verdict = 'valid' WHERE id = 40"))
        await session.execute(text("DELETE FROM callrecord WHERE org_id = :id"), {"id": gone_id})
        await session.execute(text("DELETE FROM org WHERE id = :id"), {"id": gone_id})
        await session.execute(_ROWS, {"org_id": before_swap, "count": 2, "step": timedelta(seconds=1),
                                      "newest": utcnow_naive()})
        await session.commit()

    await _swap_in_process()
    async with db.session_maker() as session:
        assert (await session.execute(select(CallRecord.id).order_by(CallRecord.id))).scalars().all() \
            == [*range(1, 41), 44, 45]
        late = await session.get(CallRecord, 40)
        assert (late.hit, late.verdict) == (True, "valid")


async def test_downgrade_puts_a_plain_table_back_with_every_row(before_swap):
    from alembic import command
    from treg.maintenance import _alembic_config

    await _upgrade_to("head")
    async with db.session_maker() as session:
        session.add(CallRecord(user_email="a@example.test", tool_name="after-swap", method="GET",
                               path="/", status_code=200))
        await session.commit()
    await db._engine.dispose()
    await asyncio.to_thread(command.downgrade, _alembic_config(), "0115")

    assert await _scalar("SELECT relkind::text FROM pg_class WHERE oid = 'callrecord'::regclass") == "r"
    assert await _scalar("SELECT count(*) FROM callrecord") == 41
    assert await _scalar("SELECT to_regclass('callrecord_partitioned')") is None
    assert await _scalar("INSERT INTO callrecord (user_email, tool_name, method, path, status_code, "
                         "kind, client, call_ref, budget_dim, budget_val, created_at, cached) VALUES "
                         "('a', 'b', 'GET', '/', 200, 'call', '', '', '', '', now(), false) RETURNING id") == 42


async def test_the_copy_runs_newest_first_and_a_stopped_run_resumes(before_swap):
    await _upgrade_to("0116")
    first = await partitions.backfill(1, batch_rows=4, max_seconds=0, pause=0)
    assert (first["phase"], first["copied"], first["done"]) == ("before_swap", 4, False)
    assert await _scalar("SELECT min(id) FROM callrecord_partitioned") == 37

    # Rows are an hour apart, so one day of history is the newest 24: ids 17 to 40.
    rest = await partitions.backfill(1, batch_rows=4, pause=0)
    assert rest["done"] is True
    assert await _scalar("SELECT min(id) FROM callrecord_partitioned") == 17
    assert (await partitions.backfill(1, batch_rows=4, pause=0))["copied"] == 0


async def test_older_history_can_follow_after_the_swap(before_swap, monkeypatch):
    monkeypatch.setattr(swap, "_INLINE_IDS", 10)
    monkeypatch.setattr(swap, "_MIN_HISTORY_DAYS", 1)
    await _upgrade_to("0116")
    await partitions.backfill(1, batch_rows=100, pause=0)
    await _swap_in_process()
    assert await _scalar("SELECT count(*) FROM callrecord") == 40   # one batch, taken whole

    async with db.session_maker() as session:
        await session.execute(text("DELETE FROM callrecord WHERE id <= 30"))
        await session.commit()
    more = await partitions.backfill(90, batch_rows=7, pause=0)
    assert (more["phase"], more["copied"], more["done"]) == ("after_swap", 30, True)
    assert await _scalar("SELECT count(*) FROM callrecord") == 40


async def test_team_deletion_clears_the_legacy_rows_and_the_table_can_be_dropped(before_swap):
    await _upgrade_to("head")
    async with db.session_maker() as session:
        org = await session.get(Org, before_swap)
        await cascade_delete_org(org, session)
        await session.commit()
    assert await _scalar("SELECT count(*) FROM callrecord_legacy") == 0
    assert await _scalar("SELECT count(*) FROM callrecord") == 0

    assert await partitions.drop_legacy() == {"dropped": True, "error": None}
    assert await partitions.drop_legacy() == {"dropped": False, "error": None}
    # The id sequence belongs to the new table and outlives the legacy one.
    assert await _scalar("SELECT nextval(pg_get_serial_sequence('callrecord', 'id'))") == 41


async def test_month_partitions_are_created_ahead_and_dropped_by_retention(before_swap, monkeypatch):
    await _upgrade_to("head")
    today = utcnow_naive().date()
    ahead = {partitions._name(partitions._month(today, offset))
             for offset in range(partitions.MONTHS_AHEAD + 1)}
    async with db.session_maker() as session:
        assert ahead <= {name for name, _, _ in (await partitions._month_tables(session)).values()}

    old = partitions._month(today, -5)
    assert await partitions.ensure([old]) == [partitions._name(old)]
    assert await partitions.ensure([old]) == []
    async with db.session_maker() as session:
        session.add(CallRecord(user_email="a@example.test", tool_name="old", method="GET", path="/",
                               status_code=200, created_at=utcnow_naive().replace(
                                   year=old.year, month=old.month, day=2)))
        await session.commit()

    assert await partitions.maintain() == {"created": [], "dropped": []}   # retention is off
    monkeypatch.setattr(get_settings(), "callrecord_retention_months", 2)
    kept = await _scalar("SELECT count(*) FROM callrecord WHERE tool_name <> 'old'")
    assert (await partitions.maintain())["dropped"] == [partitions._name(old)]
    assert await _scalar("SELECT to_regclass(:name)", name=partitions._name(old)) is None
    assert await _scalar("SELECT count(*) FROM callrecord") == kept


async def test_a_new_partition_does_not_wait_for_an_open_audit_insert(before_swap):
    await _upgrade_to("head")
    far = partitions._month(utcnow_naive().date(), 24)
    async with db.session_maker() as writer:
        writer.add(CallRecord(user_email="a@example.test", tool_name="open", method="GET", path="/",
                              status_code=200))
        await writer.flush()   # holds its lock on `callrecord` until the block ends
        assert await asyncio.wait_for(partitions.ensure([far]), timeout=2) == [partitions._name(far)]
        await writer.commit()


async def test_without_partitions_every_step_is_a_no_op():
    """SQLite, and a PostgreSQL database built from the models (as the other tests use)."""
    await db.reset_db()
    assert await partitions.maintain() == {"created": [], "dropped": []}
    assert await partitions.drop_expired(1) == []
    assert (await partitions.backfill(30))["copied"] == 0
    assert await partitions.drop_legacy() == {"dropped": False, "error": None}
    async with db.session_maker() as session:
        session.add(CallRecord(user_email="a@example.test", tool_name="echo", method="GET",
                               path="/", status_code=200))
        await session.commit()
        assert (await session.execute(select(func.count()).select_from(CallRecord))).scalar_one() == 1


async def test_rows_written_during_the_resync_are_copied_before_the_lock(before_swap, monkeypatch):
    """The resync of late verdicts reads days of rows while audit inserts continue. What they
    wrote meanwhile is copied after it, without a lock, so the swap copies at most `_SWAP_ROWS`
    new rows under its ACCESS EXCLUSIVE lock however long the resync took."""
    monkeypatch.setattr(swap, "_SWAP_ROWS", 5)
    monkeypatch.setattr(swap, "_COMMIT_MARGIN", 2)
    real_resync, real_swap, left_under_lock = swap._resync_verdicts, swap._swap, []

    def resync_while_calls_arrive(bind, reached):
        real_resync(bind, reached)
        bind.execute(_ROWS, {"org_id": before_swap, "count": 60, "step": timedelta(milliseconds=1),
                             "newest": utcnow_naive()})

    def swap_and_measure(bind, columns, reached):
        left_under_lock.append(bind.execute(text("SELECT max(id) FROM callrecord")).scalar() - reached)
        real_swap(bind, columns, reached)
    monkeypatch.setattr(swap, "_resync_verdicts", resync_while_calls_arrive)
    monkeypatch.setattr(swap, "_swap", swap_and_measure)

    await _swap_in_process()

    assert left_under_lock and left_under_lock[0] <= swap._SWAP_ROWS
    assert await _scalar("SELECT count(*) FROM callrecord") == 100


async def test_a_retry_resyncs_only_above_where_the_last_attempt_reached(before_swap, monkeypatch):
    """A lock timeout re-runs the whole upgrade; the late-verdict resync must not read its days
    of rows again each time, only what came after the attempt before."""
    monkeypatch.setattr(swap, "_BATCH_ROWS", 5)
    real_swap = swap._swap
    resyncs: list[str] = []

    def count(conn, cursor, statement, *args):
        if statement.lstrip().startswith("UPDATE callrecord_partitioned"):
            resyncs.append(statement)

    def lock_timeout_once(bind, columns, reached):
        monkeypatch.setattr(swap, "_swap", real_swap)
        raise RuntimeError("canceling statement due to lock timeout")
    monkeypatch.setattr(swap, "_swap", lock_timeout_once)
    sqlalchemy_event.listen(db._engine.sync_engine, "before_cursor_execute", count)
    try:
        with pytest.raises(RuntimeError, match="lock timeout"):
            await _swap_in_process()
        first_attempt = len(resyncs)
        async with db.session_maker() as session:
            await session.execute(_ROWS, {"org_id": before_swap, "count": 10,
                                          "step": timedelta(seconds=1), "newest": utcnow_naive()})
            await session.commit()
        resyncs.clear()
        await _swap_in_process()
    finally:
        sqlalchemy_event.remove(db._engine.sync_engine, "before_cursor_execute", count)

    assert first_attempt == 8          # 40 rows of the last two days, 5 per batch
    assert len(resyncs) == 2           # only the 10 rows above where the first attempt reached
    assert await _scalar("SELECT relkind::text FROM pg_class WHERE oid = 'callrecord'::regclass") == "p"
    assert await _scalar("SELECT count(*) FROM callrecord") == 50
    assert await _scalar("SELECT obj_description('callrecord'::regclass, 'pg_class')") is None


def test_the_backfill_paces_itself_by_default(monkeypatch):
    seen = {}

    async def backfill(days, **options):
        seen.update(options, days=days)
        return {}
    monkeypatch.setattr(partitions, "backfill", backfill)
    assert worker.main(["admin", "callrecord-backfill", "--days", "31"]) == 0
    assert seen == {"days": 31, "batch_rows": 5_000, "max_seconds": None, "pause": 0.5}
    assert worker.main(["admin", "callrecord-backfill", "--days", "31", "--pause-seconds", "2"]) == 0
    assert seen["pause"] == 2
    defaults = inspect.signature(real_backfill).parameters
    assert (defaults["batch_rows"].default, defaults["pause"].default) == (5_000, 0.5)


async def test_a_failed_partition_upkeep_fails_the_rollup_run_after_the_fold(monkeypatch, caplog):
    """A missing month refuses every audit row, so an upkeep failure must reach an alert: the
    run logs it as an error and exits 1, and the fold still runs."""
    import treg.infra.db as infra_db
    from treg.application import call_usage_rollup

    async def verified():
        return None

    async def broken():
        raise RuntimeError("canceling statement due to lock timeout")
    folded = []

    async def refresh(**options):
        folded.append(options)
        return {"rows": 0}
    monkeypatch.setattr(infra_db, "verify_db", verified)
    monkeypatch.setattr(partitions, "maintain", broken)
    monkeypatch.setattr(call_usage_rollup, "refresh", refresh)

    with caplog.at_level("ERROR", logger="treg.worker"):
        assert await worker._usage_rollup(argparse.Namespace(max_rows=10)) == 1
    assert folded and "callrecord partition upkeep failed" in caplog.text
