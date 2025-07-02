from pygeoinf.other_space.interval import Sobolev


Sobolev_Interval = Sobolev(100, 1, 0.1, interval=(0, 1))

print("Interval:", Sobolev_Interval.interval)
print("Number of points:", Sobolev_Interval.n_points)
print("Random point:", Sobolev_Interval.random_point())