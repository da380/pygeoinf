# Sobolev Space Factory Demo

This notebook demonstrates the new Sobolev space factory with spectral inner products.

## Key Features

1. **Laplacian Eigenfunction Bases**: Only Fourier-based bases that correspond to Laplacian eigenfunctions
2. **Spectral Inner Product**: Uses the mathematically correct spectral definition: ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k
3. **Boundary Conditions**: Supports periodic, Dirichlet, and Neumann boundary conditions
4. **No Weak Derivatives**: Uses spectral representation, avoiding the need for weak derivative computation

## Usage

```python
# Create a Sobolev space H^s with dimension d
space = Sobolev.factory(d, s, interval=(a, b), boundary_conditions={'type': 'periodic'})

# Available boundary conditions:
# - {'type': 'periodic'}: Full Fourier basis (cos + sin)
# - {'type': 'dirichlet'}: Pure sine basis
# - {'type': 'neumann'}: Cosine basis + constant
```

## Mathematical Background

For Laplacian eigenfunctions φ_k with eigenvalues λ_k:
- **Periodic**: λ_0 = 0, λ_{2k-1} = λ_{2k} = (kπ/L)²
- **Dirichlet**: λ_k = (kπ/L)² for k = 1, 2, ...
- **Neumann**: λ_0 = 0, λ_k = (kπ/L)² for k = 1, 2, ...

The H^s inner product becomes: ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k

This is mathematically rigorous and computationally efficient.
