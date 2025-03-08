import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.convolution import convolve, Box2DKernel
from scipy.interpolate import interp2d

south_members_with_centers = Table.read("/global/homes/m/mjyb16/south_members_final_2023.fits").to_pandas()

south_members = south_members_with_centers[
    south_members_with_centers.galaxy_z != south_members_with_centers.cluster_z
].copy()

bins_z = np.linspace(0.1, 1.0, 21)
reweightings = np.ones((20, 200))

for i in range(20):
    members_bin = south_members[
        (south_members.cluster_z > bins_z[i]) &
        (south_members.cluster_z < bins_z[i+1])
    ]
    deltas = members_bin.galaxy_z - members_bin.cluster_z
    hist_deltas, edges = np.histogram(deltas, bins=np.linspace(-1, 1, 201))
    reweighting_bin = np.minimum(hist_deltas[::-1]/hist_deltas, 1)
    reweightings[i] = reweighting_bin

smoothed_reweightings = np.nan_to_num(convolve(reweightings, Box2DKernel(3), boundary="extend"), nan=1e-3)

interpolated_reweightings = interp2d(
    x=0.5*(edges[1:] + edges[:-1]),
    y=0.5*(bins_z[:-1] + bins_z[1:]),
    z=smoothed_reweightings
)

np.save("reweighting_south_nan_fill", smoothed_reweightings)
