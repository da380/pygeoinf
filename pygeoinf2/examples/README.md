# Examples

Short scripts, one new idea each, meant to be read in order. Every one runs on
its own:

```
poetry run python pygeoinf2/examples/05_derivative_and_gradient.py
```

They are executed by the test suite, so an example that stops working is a
failing test rather than a surprise later.

## The sequence

| | | |
|---|---|---|
| 1 | `spaces` | vectors are raw backend objects; the space does the arithmetic |
| 2 | `coordinates` | a basis is a capability, not a requirement; the Gram matrix |
| 3 | `operators` | adjoints, and where the metric enters |
| 4 | `traits` | structure that survives the algebra |
| 5 | `derivative_and_gradient` | **the one to read twice** |
| 6 | `nonlinear` | `at()`, and why value and derivative come together |
| 7 | `solvers` | coordinate-free Krylov; declared preconditions |
| 8 | `functional_calculus` | `f(A)` without a matrix |
| 9 | `randomised` | low-rank factorisation and trace estimation |
| 10 | `measures` | Gaussians, sampling, pushforward |
| 11 | `direct_sums` | the joint model, linear and nonlinear |
| 12 | `optimisation` | why working in the metric changes the iteration count |
| 13 | `convex` | proximal operators as geometry |
| 14 | `fields` | circles, tori, boxes, spheres |
| 15 | `worked_example` | all of it, on a small inverse problem |

## If you read only one

Number 5. The distinction between a derivative and a gradient is the reason
most of the rest of the design looks the way it does, and it is the mistake the
library exists to make hard.
