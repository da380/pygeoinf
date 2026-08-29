import mfem.ser as mfem, numpy as np, scipy.sparse as sp, gc, time, sys
mode=sys.argv[1]
M=sp.random(400,400,density=0.02,random_state=0,format='csr')+sp.eye(400,format='csr'); M=M.tocsr(); M.sort_indices()
I=M.indptr.astype(np.int32); J=M.indices.astype(np.int32); D=M.data.astype(float)
A=mfem.SparseMatrix([I,J,D,400,400])
x=np.random.default_rng(0).standard_normal(400); ref=M@x
if mode=="free":
    del I,J,D,M; gc.collect(); junk=[np.zeros(100000) for _ in range(50)]
vx=mfem.Vector(x); vy=mfem.Vector(400); A.Mult(vx,vy)
print(mode, "Mult matches:", np.allclose(vy.GetDataArray(), ref), "nnz", A.NumNonZeroElems()); sys.stdout.flush()
if mode=="time":
    for n,d in ((2000,0.008),(6000,0.009)):
        B=sp.random(n,n,density=d,random_state=1,format='csr')+sp.eye(n,format='csr'); B=B.tocsr(); B.sort_indices()
        I=B.indptr.astype(np.int32); J=B.indices.astype(np.int32); D=B.data.astype(float)
        t=time.perf_counter(); C1=mfem.SparseMatrix([I,J,D,n,n]); t1=time.perf_counter()-t
        t=time.perf_counter(); coo=B.tocoo(); C2=mfem.SparseMatrix(n,n)
        for r,c,v in zip(coo.row,coo.col,coo.data): C2.Add(int(r),int(c),float(v))
        C2.Finalize(); t2=time.perf_counter()-t
        print(f"nnz {B.nnz}: list-form {1e3*t1:.2f} ms, per-entry Add {1e3*t2:.0f} ms")
