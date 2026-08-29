# Re-review brief (2026-08-29)

Repo: /home/david/dev/pygeoinf (branch `refactor`). Package under review: `pygeoinf2/`
(the v2 refactor of `pygeoinf/`, "v1"). Python env is the poetry venv already on PATH.

## Context you must read first
1. `pygeoinf2/REVIEW.md` sections 0, 10 and 11 (the synthesis, the work plan, and decisions
   D-1…D-13 the user took — these are settled; do not re-litigate them).
2. Your area's appendix in `pygeoinf2/review/` (named in your task). Its Must/Should/Consider
   lists are the checklist. The appendices were written on 2026-08-27; 63 commits have landed
   since (`git log --since=2026-08-27`). `pygeoinf2/CURRENT_STATE.md` describes the package as
   it now claims to be; `pygeoinf2/V1_CATALOGUE.md` maps v1 → v2 (treat "Ported" rows as
   unverified claims).
3. `pygeoinf2/review/parallel.md` — already done today; do not redo parallelism, but note where
   an `n_jobs` hook is missing if you meet one.

## Standing rules (from the user; violations are findings)
- **Check v1 before judging.** Where v2 replaces a v1 entry point, read the v1 code
  (`pygeoinf/`) and say what changed and whether the change loses a capability or a default
  that encoded a practical reason. A regression documented as a design decision is the worst
  outcome.
- **Metric bug rule.** Every metric bug so far was an expression right for a diagonal Gram
  matrix and wrong otherwise. Any numerical check you run must include
  `make_dense_metric_space()` from `pygeoinf2/tests/conftest.py`. Prefer closed-form checks to
  sampling checks. Remember: a trace is the component matrix's; an inner product carries G;
  covariance of components is G^-1 C_gal G^-1.
- Verify claims by reading the code or running it — never from commit messages or docstrings.
  Mark each finding *(verified: how)* or *(unverified)*.

## What the user wants from this round
The user is "particularly interested in potential optimisations". So, beyond checking that the
review items were addressed correctly:
- **Profile** representative workloads in your area at realistic sizes (sphere `lmax` 64–128
  where relevant; pyslfp uses lmax 256 with 10^4–10^5 observation points) with `cProfile` or
  timing scripts. Find redundant work: repeated transforms, Gram solves that could be cached
  or avoided, dense assembly where matrix-free would do, Python loops over components, allocations
  in hot loops, adjoints computed by four transforms where v1 used none, caches keyed wrongly,
  operators rebuilt per call, etc.
- For each opportunity give: the file:line, what is recomputed and why it is unnecessary,
  the measured cost now, the estimated (or measured, if you prototype it in a scratch file)
  saving, and the risk. Rank by gain × confidence. Do not propose micro-optimisations under 10%.
- Other agents are profiling concurrently on a 16-thread laptop that throttles under load.
  Timings will be noisy: compare alternatives *interleaved in the same run*, repeat, and report
  ratios rather than absolute claims. Set `OMP_NUM_THREADS=1` for profiling runs unless
  the point is BLAS threading.

## Rules of engagement
- Do **not** modify anything under the repo. Throwaway scripts go in
  `/tmp/claude-1000/-home-david-dev-pygeoinf/f9c0fa04-81e1-449d-8680-359b1c9dd92b/scratchpad/<yourarea>/`.
- Run your area's tests (`pytest pygeoinf2/tests/test_<x>.py -q`, default excludes `slow`)
  and report pass/fail counts; run a `slow` test only if it is central to a finding.
- Read whole functions before judging them; the code is heavily documented and the docstrings
  sometimes claim more than the code does.

## Output
Write `/tmp/claude-1000/-home-david-dev-pygeoinf/f9c0fa04-81e1-449d-8680-359b1c9dd92b/scratchpad/rereview_<area>.md` with exactly these sections:
1. **Review items — status.** A table: item (from the appendix Must/Should/Consider and from
   REVIEW.md §10 for your area) | status (done / partial / not done / done differently) |
   how verified (file:line, test name, or the script you ran) | note. Include D-decisions
   that touch your area.
2. **Bugs and regressions found now.** Verified ones first, each with a reproduction
   (inputs → wrong output) and file:line. Then unverified suspicions, labelled.
3. **Optimisations, ranked.** As specified above, with numbers.
4. **Open questions for the user.** Only ones that change what should be done.
Keep it tight: file:line references, no narrative padding, no praise. Under ~250 lines.
Return a ~15-line summary of the highlights as your final message.
