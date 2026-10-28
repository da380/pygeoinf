# Mini-apps

Mini-apps are small, complete applications of `pygeoinf2`, in the spirit of
MFEM's: each is a self-contained directory with a README stating the problem,
real or realistic data, several features of the library composed into one
inverse problem, and a run of at most a few minutes that prints its numbers
and saves its figure. They live under the package directory but are not imported
by it; they are read, built where they need building, and run from their own
directory. The test suite runs each one that can be run on the machine at
hand and skips the rest with a reason.

| directory | problem |
|---|---|
| `cpp_backend/` | a C++ vector type with its own inner product, bound with nanobind and exposed as a `HilbertSpace` with no array underneath; an inverse problem run on it end to end |
