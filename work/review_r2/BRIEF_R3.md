# Brief for the third round (2026-08-30)

You are implementing part of the action plan in `pygeoinf2/REVIEW2.md`. Read it
first — all of it, but especially:

* **§5, the user's answers.** Each question has a `Response:` line written by
  David. These are decisions, not suggestions, and they are not up for
  re-litigation.
* **The `**DONE** (`hash`)` markers** through §3, §4 and §6. Those items were
  implemented yesterday by another model in six commits (`542de76` …
  `8511651`). **Do not undo, rewrite, "simplify" or revert any of that work.**
  If you believe something in it is wrong, stop and report it to the main
  session — do not change it. The user asked for this explicitly.
* Your area's appendix in `pygeoinf2/review/` (named in your task) for the
  file:line references and the measurements behind each item.

## Where you work

You have your own git worktree, named in your task. **Work only there.** Do
not touch `/home/david/dev/pygeoinf` (another agent is using it), and do not
touch the other worktrees. Your branch is already checked out.

Throwaway scripts go in
`/tmp/claude-1000/-home-david-dev-pygeoinf/f9c0fa04-81e1-449d-8680-359b1c9dd92b/scratchpad/r3/<your-area>/`.

## Rules that are not negotiable

1. **Stay inside your file set.** It is listed in your task. Other agents are
   editing the other files at the same time, and a merge conflict costs more
   than the change is worth. If an item needs a change outside your set, do
   the part inside it and report the rest.
2. **The metric rule.** Every metric bug in this package has been an
   expression that is right for a diagonal Gram matrix and wrong otherwise.
   Any test touching the metric uses `make_dense_metric_space()` from
   `pygeoinf2/tests/conftest.py` (now well conditioned at any size). Prefer a
   closed form to a sampling check. A trace is the component matrix's; an
   inner product carries `G`; the covariance of components is `G^-1 C_gal G^-1`.
3. **Check v1 before inventing.** `pygeoinf/` is the working v1. Where an item
   restores or changes a v1 behaviour, read v1 first and say what it did.
4. **One commit per item.** Each commit: the change, its tests, the full
   default suite green (`python -m pytest pygeoinf2 -q`, ~90 s), and the
   measured before/after in the message where the item is an optimisation.
   Never commit with a failing suite. Message style: a short imperative
   subject line, then prose explaining *why*, then the numbers. End every
   commit message with:

   ```
   Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_015dw9KzYBvf6hcPrj58z2eK
   ```
5. **Measure honestly.** `OMP_NUM_THREADS=1`, interleave the before and after
   in one run, repeat, quote ratios. Four agents are profiling at once on a
   16-thread laptop that throttles, so absolute numbers drift; ratios do not.
   To measure "before", `git stash` or use the pristine copy at
   `/home/david/dev/pygeoinf` **read-only** (`PYTHONPATH=/home/david/dev/pygeoinf`).
   To measure "after" from your worktree, set
   `PYTHONPATH=<your worktree>` — a bare `python` imports the installed
   package, which is the *other* checkout.
6. **Docstrings carry the contract.** `tests/test_code_practice.py` enforces
   Args/Returns/Raises on public definitions, keyword-only optional
   arguments, and annotations. Run it early; it will catch you.
7. Do not change `pygeoinf2/REVIEW2.md`, `pygeoinf2/review/*`, `V1_CATALOGUE.md`
   or `CURRENT_STATE.md` unless your task says so — the main session or another
   agent owns them.

## What to report

A short final message: one line per commit (hash + subject), the measured
gains, anything you left undone and why, and anything you found that is not in
REVIEW2. Keep it under 20 lines. Do not paste diffs.
