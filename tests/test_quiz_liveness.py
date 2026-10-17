"""A slow chat quiz must stay visibly alive and still finish (2026-10-16).

A knowledge quiz delivered one question of five, went silent while the rest
were judged and rewritten, and the browser's 90s stall watchdog cancelled it:
the student saw nothing and nothing was saved.
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from services import planned_practice as planned
from services import practice_api


async def collect(events):
    return [e async for e in events]


class HeartbeatFillsSilence(unittest.TestCase):
    def test_silence_becomes_heartbeats_and_events_pass_through(self):
        async def slow():
            yield {"status": "generating"}
            await asyncio.sleep(0.25)
            yield {"status": "question_ready"}

        out = asyncio.run(collect(practice_api._with_heartbeat(slow(), every=0.05)))
        self.assertEqual(out[0], {"status": "generating"})
        self.assertEqual(out[-1], {"status": "question_ready"})
        self.assertGreaterEqual(sum(e["status"] == "heartbeat" for e in out), 3)

    def test_an_error_before_the_first_question_still_reaches_the_caller(self):
        async def broken():
            yield {"status": "material_analyzing"}
            raise planned.PlanUnavailable("no goals")

        with self.assertRaises(planned.PlanUnavailable):
            asyncio.run(collect(practice_api._with_heartbeat(broken(), every=0.05)))

    def test_a_quiet_stream_is_never_padded(self):
        async def quick():
            yield {"status": "quiz_complete"}

        self.assertEqual(asyncio.run(collect(practice_api._with_heartbeat(quick(), every=1))),
                         [{"status": "quiz_complete"}])


GOALS = [{"id": f"g{i}", "topic": f"Topic {i}"} for i in range(5)]


class BatchDeadline(unittest.TestCase):
    def run_batch(self, delays):
        async def write_checked(slot, goal, *args, **kwargs):
            await asyncio.sleep(delays[slot["index"]])
            return {"question": f"Q{slot['index']}"}

        async def understand_session(*args, **kwargs):
            return []

        async def resolve_plan(**kwargs):
            return {"id": "req-x", "selection": {}, "formats": ["mcq"], "cursor": 0}

        slots = [{"index": i, "goalId": f"g{i}"} for i in range(len(delays))]
        with mock.patch("services.document_understanding.understand_session", understand_session), \
                mock.patch.object(planned.plan_model, "compact_goals", return_value=GOALS), \
                mock.patch.object(planned.plan_model, "schedule", return_value=slots), \
                mock.patch.object(planned, "resolve_plan", resolve_plan), \
                mock.patch.object(planned, "plan_summary", return_value={}), \
                mock.patch.object(planned, "write_checked", write_checked), \
                mock.patch.object(planned, "advance_cursor"), \
                mock.patch.object(planned.request_brief, "checkable", return_value=[]), \
                mock.patch.object(planned, "BATCH_DEADLINE_S", 0.3):
            return asyncio.run(collect(planned.stream_planned(
                session=SimpleNamespace(), chat_id="c", view={"items": []}, total=5, count=len(delays),
                difficulty="medium", question_types=["mcq"], quiz_mode="knowledge", learning_objective="general")))

    def test_a_stuck_slot_does_not_hold_back_the_ones_she_has(self):
        events = self.run_batch([0.01, 0.02, 30, 30, 30])
        self.assertEqual(sorted(e["question"]["question"] for e in events if e["status"] == "question_ready"), ["Q0", "Q1"])
        self.assertEqual(events[-1]["status"], "quiz_complete")
        self.assertEqual(events[-1]["total_generated"], 2)

    def test_past_the_deadline_it_still_waits_for_her_first_question(self):
        events = self.run_batch([0.5, 30])
        self.assertEqual(events[-1]["total_generated"], 1)


if __name__ == "__main__":
    unittest.main()
