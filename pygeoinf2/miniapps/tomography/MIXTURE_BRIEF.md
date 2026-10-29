# Brief: a mixture-prior part for the tomography mini-app (2026-10-29)

The maintainer's decision: a part on the linear disk showing the mixture
functionality, which exists to put a discrete hypothesis inside a
continuous problem. It goes after the set answer and before the ball, so
it becomes part E, the ball F and the bent rays G; move the letters, not
the files, everywhere they appear (docstrings, `run_all.py`, the README's
part list, answers section and file table, the test, the figure titles,
the random-number keys). Only the linear disk; the bent-ray version (a
mixture of Laplace approximations) is deliberately not built. Delete this
file when done; the README carries the design.

## Read first

The app as it stands after today's revision (`README.md`, `problem.py`,
`bayesian.py`, `sets.py`, `common.py`, `run_all.py`) and its test;
`pygeoinf2/probability/mixture.py` (`GaussianMixture`) and
`pygeoinf2/inference/mixture.py` (`LinearGaussianMixtureInversion`:
components updated by linear Gaussian inversions, weights rescored from
the components' log evidences, `push_forward` onto the properties) with
`pygeoinf2/tests/test_mixture.py` for how they are built and read.

## The part (`mixture.py`, part E)

- **The hypothesis.** A plume sits at one of a handful of candidate
  locations in the mantle (five at the full scale, three at the quick
  scale: different angles and depths, some under dense ray coverage and
  some under sparse). Each component of the prior is the Gaussian whose
  mean is the reference slowness plus a bump at its candidate (the bump
  the app already knows how to make, of the amplitude the truth uses) and
  whose covariance is the app's prior covariance, shared by every
  component; the prior weights are equal. The truth is the bump at one
  candidate (choose one under moderate coverage) plus a small draw from
  the shared covariance, and the data come from it with the app's noise.
- **The answer.** `LinearGaussianMixtureInversion` with the app's solver
  and formalism; the posterior weights against the prior's; the
  posterior mixture pushed onto the three properties, its marginals
  against the prior mixture's; the winning component's mean as the
  field. Then a sweep of the noise level (say four values from a tenth
  of the app's to ten times it, the data redrawn at each) showing the
  weights sharpen from the prior's equal ones to one candidate as the
  noise falls, and where coverage leaves two candidates indistinguishable
  if it does. Say in the README what the weights are (evidence ratios)
  and why a single Gaussian posterior could not say this.
- **Figures**, each titled "Tomography, part E: ...": the candidates on
  the disk with the rays and the truth; the weights against the noise
  level; the prior and posterior marginals of the three properties (a
  mixture is drawn as its density, not as a Gaussian); the winning
  component's mean with the truth.
- **The test**: at the quick scale, the posterior weight of the true
  candidate is the largest at the app's noise level, the weights sum to
  one, and they move toward equal as the noise grows.

## House rules

As the other apps: no decision numbers, planning or history talk in code
or docs; American spelling; ruff clean; `MPLBACKEND=Agg`, never `show()`;
quick scale first; no run over two minutes; figures per method and
titled; the app's slow test (`-m ""`) until the end, then the whole fast
suite once; touch only `pygeoinf2/miniapps/tomography/`, its test and its
row in `pygeoinf2/miniapps/README.md`; no library edits (record problems
with the smallest reproduction); other agents are working under
`waveform/` and `planet_density/`, leave them alone; do not commit.
Report: the weights at the full scale across the sweep, the choices made,
library problems, anything undone.
