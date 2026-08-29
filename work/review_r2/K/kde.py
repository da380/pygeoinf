import numpy as np, time
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
rng=np.random.default_rng(0)
draws=np.concatenate([rng.multivariate_normal([0,0],[[1,.5],[.5,1]],10000), rng.multivariate_normal([4,3],[[1,-.3],[-.3,2]],10000)])
for res in (100,200):
    hx=np.linspace(-4,8,res); hy=np.linspace(-4,8,res); mx,my=np.meshgrid(hx,hy); grid=np.vstack([mx.ravel(),my.ravel()])
    t=time.perf_counter(); full=gaussian_kde(draws.T)(grid).reshape(mx.shape); tf=time.perf_counter()-t
    t=time.perf_counter(); sub=gaussian_kde(draws[::5].T)(grid).reshape(mx.shape); ts=time.perf_counter()-t
    t=time.perf_counter(); kde=gaussian_kde(draws.T); H,_,_=np.histogram2d(draws[:,0],draws[:,1],bins=[np.r_[hx-(hx[1]-hx[0])/2,hx[-1]+(hx[1]-hx[0])/2],np.r_[hy-(hy[1]-hy[0])/2,hy[-1]+(hy[1]-hy[0])/2]]); bw=np.sqrt(np.diag(kde.covariance))/np.array([hx[1]-hx[0],hy[1]-hy[0]]); hist=gaussian_filter(H.T,bw,mode="constant"); hist/= hist.sum()*(hx[1]-hx[0])*(hy[1]-hy[0]); th=time.perf_counter()-t
    print(f"grid {res}x{res}: full kde {tf:.2f}s | 1/5 subsample {ts:.2f}s (max rel err {np.abs(sub-full).max()/full.max():.3f}) | binned+filter {th:.3f}s (max rel err {np.abs(hist-full).max()/full.max():.3f})")
