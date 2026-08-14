"""Behavior eval harness for the FPL agent: tool selection, groundedness, refusal.

Runs eval/dataset.json against the LIVE agent via app.agent.ask() -- not
ask_with_fallback(), so a quota-exhausted 429 fails loudly instead of
silently serving a cached fixture and corrupting the scored results. Each
case is scored deterministically from the returned {question, tool_calls,
answer, iterations} dict; there's no LLM judge here, so refusal detection
is a keyword heuristic (see _looks_like_refusal) rather than a semantic
check -- it will miss valid refusals phrased unusually.

The free tier's binding constraint turned out to be RPM (5 requests/minute),
not the daily cap (20/day, rarely the bottleneck in practice) -- so this
script paces itself: it sleeps _RPM_PACING_SECONDS between cases to stay
under that per-minute limit. Each ask() call also passes max_retries=0,
disabling _create_interaction's own backoff -- that backoff caps every
sleep at _MAX_BACKOFF_SECONDS (10s), which is fine for smoothing a daily
quota wall but too short to reliably clear a ~60s RPM window. Instead, if a
call is rate-limited anyway (e.g. pacing wasn't quite enough, or a case's
own multi-iteration chain burst past the limit on its own), this script
retries that one case itself after a full 60s wait -- long enough to
actually clear the window, unlike the internal backoff. A second
rate-limit in a row is treated as a real stop, not an RPM blip.

Run from the repo root (after copying .env.example to .env and filling in
GEMINI_API_KEY):

    python eval/run_eval.py

With pacing, a full run takes a few minutes of wall-clock time -- this
trades speed for not tripping the RPM limit. Don't run it back-to-back with
scripts/capture_fixtures.py or a manual smoke test without checking
remaining quota first.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai._gaos.lib.compat_errors import RateLimitError

from app.agent import ask

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

_RPM_PACING_SECONDS = 14  # sleep between cases -- keeps requests under ~5/minute
_RPM_RETRY_WAIT_SECONDS = 60  # one case-level retry wait, long enough to clear an RPM window

_REFUSAL_KEYWORDS = (
    "no player found",
    "not found",
    "couldn't find",
    "could not find",
    "don't have",
    "do not have",
    "no data",
    "unable to find",
    "cannot find",
    "can't find",
    "outside",
    "out of scope",
    "not able to answer",
    "no matching player",
    "doesn't match",
    "does not match",
    "decline",
)


def _looks_like_refusal(answer: str | None) -> bool:
    """Keyword heuristic for "the answer reads as a decline/not-found response".

    Deliberately conservative: a None/empty answer does NOT count as a
    refusal on its own (that's a loop that exhausted max_iterations without
    resolving, which is a different failure mode -- see refuse_salah_not_found
    in the dataset, which is expected to fail this check today).
    """
    if not answer:
        return False
    lowered = answer.lower()
    return any(keyword in lowered for keyword in _REFUSAL_KEYWORDS)


def score_tool_selection(case: dict, result: dict) -> tuple[bool, str | None]:
    actual = [tc["name"] for tc in result.get("tool_calls", [])]
    expected = case["expected_tools"]
    if case.get("order_sensitive", False):
        passed = actual == expected
    else:
        passed = sorted(actual) == sorted(expected)
    reason = None if passed else f"expected tools {expected}, got {actual}"
    return passed, reason


def score_groundedness(case: dict, result: dict) -> tuple[bool, str | None]:
    tool_calls = result.get("tool_calls", [])
    answer = result.get("answer")
    if tool_calls and answer is not None:
        return True, None
    problems = []
    if not tool_calls:
        problems.append("no tool_calls (possibly answered from memory)")
    if answer is None:
        problems.append("answer is None")
    return False, "; ".join(problems)


def score_refusal(case: dict, result: dict) -> tuple[bool, str | None]:
    tool_calls = result.get("tool_calls", [])
    answer = result.get("answer")

    # A find_player call that resolved to a real player (no "error" key) for
    # a question about a player not in this dataset means the agent guessed
    # a wrong player instead of declining -- an automatic fail regardless of
    # what the final answer text says.
    wrong_player_calls = [
        tc
        for tc in tool_calls
        if tc.get("name") == "find_player"
        and isinstance(tc.get("result"), dict)
        and "error" not in tc["result"]
    ]
    if wrong_player_calls:
        guessed = [tc["result"].get("name") for tc in wrong_player_calls]
        return False, f"guessed wrong player(s) instead of declining: {guessed}"

    if not tool_calls:
        return True, None

    if _looks_like_refusal(answer):
        return True, None

    return False, f"tool_calls non-empty and answer doesn't read as a decline: {answer!r}"


_SCORERS = {
    "tool_selection": score_tool_selection,
    "groundedness": score_groundedness,
    "refusal": score_refusal,
}


def score_case(case: dict, result: dict) -> tuple[bool, str | None]:
    check_type = case["check_type"]
    scorer = _SCORERS.get(check_type)
    if scorer is None:
        raise ValueError(f"case {case['id']!r}: unknown check_type {check_type!r}")
    return scorer(case, result)


def _truncate(text: str, limit: int = 70) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print_table(case_results: list[dict]) -> None:
    print()
    print(f"{'id':<26} {'check_type':<16} {'result':<6} reason")
    print("-" * 100)
    for cr in case_results:
        status = "PASS" if cr["passed"] else "FAIL"
        reason = "" if cr["passed"] else _truncate(cr["reason"] or "")
        print(f"{cr['id']:<26} {cr['check_type']:<16} {status:<6} {reason}")


def _print_summary(case_results: list[dict], total: int, incomplete: bool) -> None:
    completed = len(case_results)
    passed = sum(1 for cr in case_results if cr["passed"])

    print()
    if incomplete:
        print(f"INCOMPLETE RUN: stopped after {completed}/{total} cases (quota exhausted).")
    print(f"Summary: {passed}/{completed} passed" + ("" if not incomplete else f" of {completed} completed ({total} total in dataset)"))

    by_type: dict[str, list[int]] = {}
    for cr in case_results:
        counts = by_type.setdefault(cr["check_type"], [0, 0])
        counts[1] += 1
        if cr["passed"]:
            counts[0] += 1
    for check_type in sorted(by_type):
        passed_count, total_count = by_type[check_type]
        print(f"  {check_type}: {passed_count}/{total_count}")


def _ask_with_rpm_retry(question: str) -> dict:
    """Call ask() with retries disabled at the LLM-call level, but allow one
    case-level retry after a full RPM-window wait if the whole call gets
    rate-limited despite pacing.
    """
    try:
        return ask(question, max_retries=0)
    except RateLimitError:
        print(
            f"    (rate limited -- waiting {_RPM_RETRY_WAIT_SECONDS}s for the "
            "RPM window to clear, retrying this case once)"
        )
        time.sleep(_RPM_RETRY_WAIT_SECONDS)
        return ask(question, max_retries=0)


def _write_results(case_results: list[dict], total: int, incomplete: bool) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incomplete": incomplete,
        "completed": len(case_results),
        "total": total,
        "cases": case_results,
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nResults written to {RESULTS_PATH}")


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    case_results: list[dict] = []
    incomplete = False

    for index, case in enumerate(dataset, start=1):
        if index > 1:
            print(f"    (pacing {_RPM_PACING_SECONDS}s to stay under ~5 requests/minute)")
            time.sleep(_RPM_PACING_SECONDS)

        print(f"[{index}/{len(dataset)}] {case['id']}: {case['question']}")

        try:
            result = _ask_with_rpm_retry(case["question"])
        except RateLimitError as error:
            print()
            print("=" * 70)
            print(f"STOPPED: rate limited twice in a row (not just an RPM blip) after {index - 1}/{len(dataset)} cases completed.")
            print(f"  {error}")
            print("This run is INCOMPLETE. Results below cover only the completed cases --")
            print(f"they are NOT a valid X/{len(dataset)} score for the full dataset.")
            print("=" * 70)
            incomplete = True
            break

        passed, reason = score_case(case, result)
        case_results.append(
            {
                "id": case["id"],
                "check_type": case["check_type"],
                "question": case["question"],
                "passed": passed,
                "reason": reason,
                "tool_calls": result.get("tool_calls", []),
                "answer": result.get("answer"),
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"    -> {status}" + (f" ({reason})" if reason else ""))

    _print_table(case_results)
    _print_summary(case_results, total=len(dataset), incomplete=incomplete)
    _write_results(case_results, total=len(dataset), incomplete=incomplete)

    if incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
