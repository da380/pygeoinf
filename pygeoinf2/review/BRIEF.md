# The 2.0 review: brief for every reviewer

Working directory `/home/david/dev/pygeoinf`, package `pygeoinf2`, run with
`poetry run` (tests: `MPLBACKEND=Agg poetry run pytest pygeoinf2/tests/<file>
-q -p no:cacheprovider`). You are reviewing, not editing: **make no change
to any file under `pygeoinf2/` except your own findings file.** Scratch
scripts go in `/tmp/claude-1000/-home-david-dev-pygeoinf/4bec2589-8482-4f4c-843c-b27f331a8905/scratchpad/`.

## What this is

`pygeoinf2` is version 2 of a library for inference on Hilbert spaces, about
to be released as 2.0 after this review. It is feature complete. The
review asks, layer by layer, whether the public surface is right, and
reports what is not. Its findings will be read by the library's author and
acted on one by one, so each must be specific enough to act on and honest
about how sure you are.

Read first, in this order:

1. `pygeoinf2/CURRENT_STATE.md` §1 to §6 (the one idea, the package map,
   conventions, what is exact and estimated, where dense matrices form, the
   backend contract) and §7 (open questions: do not re-report what is there;
   do say if you think one is wrong).
2. `pygeoinf2/DECISIONS.md`: the entries from D-163 on, and any earlier entry
   a docstring or your own question leads you to. Decisions are settled; a
   finding that contradicts one must say so and argue it.
3. The principles the author holds the library to:
   - **The problem is posed on a space appropriate to the forward model**
     (one on which it is bounded); the prior is a measure on that space;
     nothing is "relative to L2".
   - **No identity covariance on a function space**: a covariance on a
     space that discretizes a function space is trace class, of Sobolev
     order above half the dimension; the library refuses otherwise without
     `unsafe=True`. Set answers harden with `ambient_ball`.
   - **Matrix-free by default**; where a dense fallback exists it is chosen
     by counting applications against the alternative, not by dimension
     alone; theoretical scaling beats toy timings.
   - **Every independent loop takes `n_jobs=`**, processes not threads.
   - **The checks in `testing.py`** take `measure=`, `rtol=`, `atol=` and
     return a `CheckReport`.
   - **Docstrings are self-contained**: what the thing is, its arguments,
     returns, raises; no decision numbers, no history, no "v1", no planning
     talk, anywhere in code, docstrings, tests or examples.
   - **Naming**: v1's names unless there was a strong reason; `expectation`
     not `mean`; `property` not `target`; `forward_problem`; `max_iterations`;
     American spelling in code; "tensor product" not "Kronecker".
   - **Nothing provisional**: every public name ships as committed API.

## What to look for, in your layer

For every public class and function (what `__all__` exports, and what the
top-level `pygeoinf2/__init__.py` re-exports from your layer):

- **Posing and refusal.** Does it sit on the right space, carry the right
  traits, and refuse what the principles say it should (a density on a
  function space, an identity covariance, an operator on the wrong space,
  a non-trace-class measure)? Does it accept something it should refuse,
  or refuse something legitimate?
- **Cost.** Is it matrix-free where it can be? Where it assembles, does it
  say so and count applications? Does every loop that could run in
  parallel take `n_jobs=`? Is any default chosen by dimension where it
  should count applications?
- **Correctness you can test.** Try it on a non-orthonormal metric (the
  tests' `make_dense_metric_space` and `make_weighted_space` in
  `pygeoinf2/tests/conftest.py`), a direct sum, a function space. Run the
  `testing.check_*` routines on what you build. A bug claim must come with
  a reproduction that fails; run it.
- **Consistency.** The same keyword meaning the same thing everywhere
  (`level`, `rtol`, `atol`, `measure=`, `rng=`, `n_jobs=`, `unsafe=`,
  `solver=`, `method=`); the same return type for the same kind of answer;
  errors of the same class for the same kind of fault, with messages that
  say what to do.
- **Surface.** Public names that should be private, private names the
  tests or mini-apps reach for (grep `pygeoinf2/tests` and
  `pygeoinf2/miniapps` for `._`), duplicated logic, dead code, a method
  whose only caller is a test. Anything the mini-apps worked around rather
  than used: read each app's `IMPLEMENTATION.md` for your layer's names and
  ask why the app did it that way.
- **Docstrings.** Self-contained, accurate against the code, no history. A
  docstring that describes behavior the code no longer has is a finding.
- **Naming.** Against the preferences above, and against each other.

Do not report style you would merely have done differently. Do not
propose rewrites of what works. Do not re-report CURRENT_STATE §7.

## Your findings file

Write `pygeoinf2/review/<layer>.md` (the layer name you were given). Put
findings in severity order under three headings:

- **Must fix before 2.0**: wrong answers, refusals missing where the
  principles demand them, public API that will have to change.
- **Should fix**: cost, consistency, surface, docstrings that mislead.
- **Could fix**: the rest.

Each finding, in this shape:

```
### <short title>
`path/file.py:line` — <one-paragraph statement of the defect, what it
costs or breaks, and how sure you are (verified by running / read only)>.
Reproduction: <the minimal code or command, and what it printed>.
Proposed fix: <one or two sentences>.
```

End the file with a short section **What is sound**: the parts of the
layer you exercised and found right, so the author knows what was covered.
Then report back to the caller with the counts under each heading and the
three findings you consider most important, in two paragraphs at most.

## Rules

- Keep every command under two minutes and every problem small; no
  large-scale runs. `MPLBACKEND=Agg`; never show a figure.
- Do not commit. Do not edit anything but your findings file.
