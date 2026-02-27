## Plan: Coefficient Space Notation and Metric

Make Chapter 2 more explicit about coefficient representations by (i) introducing standard-basis symbols for coefficient spaces ($\ell^2$, $(\ell^2)'$, $\mathbb{R}^N$, $(\mathbb{R}^N)'$), and (ii) stating clearly that the coefficient space carries an induced inner product/norm so coefficient inner products agree with those of the original Hilbert space.

**Phases**
1. **Phase 1: Add coefficient basis notation**
    - **Objective:** Define standard basis vectors/functionals for coefficient spaces once, so later sections can reference them unambiguously.
    - **Files/Functions to Modify/Create:** pygeoinf/theory/TECHNICAL_MANUAL.tex (Notation and Conventions tables)
    - **Tests to Write:** None
    - **Steps:**
        1. Add $e_j\in\ell^2$ and $e^j\in(\ell^2)'$ with $\langle e^i, e_j\rangle=\delta^i_j$ and $e^j(c)=c_j$.
        2. Add $\mathbf{e}_j\in\mathbb{R}^N$ and (optionally) $\mathbf{e}^j\in(\mathbb{R}^N)'$ with matching pairing.
        3. Add a brief remark that $(\ell^2)'\cong\ell^2$ via the Riesz map for the standard $\ell^2$ inner product, while keeping primes in the text for clarity.

2. **Phase 2: Clarify coefficient-space metric and induced geometry**
    - **Objective:** Make the “coefficient space” viewpoint explicit: $\ell^2$ is the carrier of coefficients, but geometry is inherited from $H$ via synthesis.
    - **Files/Functions to Modify/Create:** pygeoinf/theory/TECHNICAL_MANUAL.tex (§2.6 Bases and representations; §2.7 Gram matrix)
    - **Tests to Write:** None
    - **Steps:**
        1. Add a short paragraph distinguishing norm equivalence vs equality for coefficients.
        2. Define the induced coefficient inner product $(c,d)_\pi := (\pi c,\pi d)_H$ (and its finite-dimensional analogue) so that $\|c\|_\pi=\|\pi c\|_H$ by definition.
        3. Connect this induced geometry to the Gram matrix representation in $\mathbb{R}^N$.

3. **Phase 3: Revise dual-basis subsection to use basis notation and coefficient-space language**
    - **Objective:** Make dual-side analysis/synthesis explicit with named basis vectors $e_j$ and evaluation functionals $e^j$.
    - **Files/Functions to Modify/Create:** pygeoinf/theory/TECHNICAL_MANUAL.tex (§2.6.4 The dual basis)
    - **Tests to Write:** None
    - **Steps:**
        1. Introduce $e_j$ and $e^j$ and define $[\ell']_j:=\ell'(e_j)$ for $\ell'\in(\ell^2)'$.
        2. Keep and expand the explanation of $\psi_i=\Riesz(\phi^i)$ and its relation to biorthogonality.
        3. Ensure the coefficient-space framing is explicit (these operators act on coefficient representations).

4. **Phase 4: Recompile and finalize**
    - **Objective:** Ensure the manual compiles cleanly after edits.
    - **Files/Functions to Modify/Create:** pygeoinf/theory/TECHNICAL_MANUAL.tex
    - **Tests to Write:** None
    - **Steps:**
        1. Run pdflatex to confirm no LaTeX errors.
        2. Prepare a single git commit message summarizing the documentation clarification.

**Open Questions**
1. None (notation choice confirmed: $e_j$ for $\ell^2$, $\mathbf{e}_j$ for $\mathbb{R}^N$; include remark $(\ell^2)'\cong\ell^2$).
