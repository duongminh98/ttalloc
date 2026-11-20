# P1 checkpoint and unattended execution: self-review

This is the author's self-review under PLAN 8.2, not an independent P1a/P1b review (waived in ADR-018).

- Full and official DEER persist completed (qid, seed) rows atomically. Resume validates checksum, config,
  role and exact requested keys, rejects duplicates, starts a new run and never regenerates completed keys.
- Equivalence checkpoints both sides independently, reconstructs all 400 questions and both seeds before
  computing the original paired bootstrap report. No statistical threshold or token definition changed.
- Operational config uses groups/concurrency 32 and logger flush 8. All settings are in config_hash;
  a test proves logging semantics/hash remain identical to qwen3.yaml. Official DEER commit/profile retained.
- Pipeline stops on the PLAN MATH baseline tolerance, inconclusive/failed equivalence, failed commands or
  sync. No automatic parameter search, extra seeds or favorable test rerun.
- Server supervisor kills the entire process group, bounds final synchronization, and destroys only its
  own CONTAINER_ID on completion, time/budget margin, missing progress, or errors. A failed final sync
  does not bypass destruction. Deadline includes provisioning/rental start, not merely model runtime.
- External API/host failures and delayed credit settlement cannot be made into an absolute billing guarantee;
  reserve and an independent local monitor provide additional protection. No machine is rented by tests.
- 11 new tests exercise interruption/resume for Full/DEER/equivalence, corrupt/foreign/duplicate checkpoints,
  time/budget/stall rules, actual supervisor control flow with mocked API/processes, failed sync cleanup,
  scientific stage gates and unchanged log_hash. All CPU checks: 282 passed, 10 deselected.
- Ruff, mypy (66 source files), split verification (5 splits), shell syntax and diff whitespace passed.
  GPU regression is the first step of p1_all before test_main baseline. Any GPU failure aborts and destroys.
- Audit delegation and P1 rental authorization are recorded in ADR-021. No completion claim precedes data.
