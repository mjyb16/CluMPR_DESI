import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.convolution import convolve, Box2DKernel
from scipy.interpolate import interp2d

north_members_with_centers = Table.read("/global/homes/m/mjyb16/north_members_final_2023.fits").to_pandas()
north_members = north_members_with_centers[north_members_with_centers.galaxy_z != north_members_with_centers.cluster_z].copy()

bins_z = np.linspace(0.1, 1.0, 21)
reweightings = np.ones((20, 200))

for i in range(20):
    members_bin = north_members[
        (north_members.cluster_z > bins_z[i]) &
        (north_members.cluster_z < bins_z[i+1])
    ]
    deltas = members_bin.galaxy_z - members_bin.cluster_z
    hist_deltas, edges = np.histogram(deltas, bins=np.linspace(-0.6, 0.6, 201))
    ratio = hist_deltas[::-1] / hist_deltas
    reweightings[i] = np.minimum(np.nan_to_num(reweighting_bin, nan=1), 1)

smoothed_reweightings = np.nan_to_num(convolve(reweightings, Box2DKernel(3), boundary="extend"), nan=1)

interpolated_reweightings = interp2d(
    x=0.5*(edges[1:] + edges[:-1]),
    y=0.5*(bins_z[:-1] + bins_z[1:]),
    z=smoothed_reweightings
)

np.save("reweighting_north_extrapolated", smoothed_reweightings)
