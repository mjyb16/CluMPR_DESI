import numpy as np
import pandas as pd
from scipy import spatial
from scipy.interpolate import interp1d, interp2d
from scipy.stats import norm
from astropy.cosmology import LambdaCDM as Cos
from astropy.table import Table
import fitsio

# Assumed inputs (must be initialized beforehand):
# massive_sample_original, iterator, indexable, gauss_array, oversample

# Initialization of reweighting interpolation
bins_z = np.linspace(0.1, 1.0, 21)
smoothed_reweightings = np.load("reweighting_south_extrapolated.npy")
smoothed_reweightings[smoothed_reweightings == 0] = 1e-2
edges = np.linspace(-0.6, 0.6, 201)
interpolated_reweighting_extrapolate = interp2d(
    x=0.5 * (edges[1:] + edges[:-1]),
    y=0.5 * (bins_z[:-1] + bins_z[1:]),
    z=smoothed_reweightings
)

# Cluster finder function definition
def cluster_finder(massive_sample_original, iterator, indexable, gauss_array, maxRA, maxDEC, minRA, minDEC, oversample):

    cos = Cos(H0=69.6, Om0=.286, Ode0=.714)

    # Radius and redshift thresholds
    z_array = np.linspace(1e-2, 2, 500)
    sparse_radius = (1 + z_array) / cos.comoving_distance(z_array)
    radius_threshold = interp1d(z_array, sparse_radius, kind="linear", fill_value="extrapolate")

    z_source = Table.read("threshold_training_south_2.fits", format="fits")
    z_data_array = z_source.to_pandas()
    z_threshold = interp1d(z_data_array.z, z_data_array.sigma_z_threshold, kind="linear", fill_value="extrapolate")

    t_source = Table.read("neighbor_training_south_2_2.fits", format="fits")
    t_data_array = t_source.to_pandas()
    thresh1 = interp1d(t_data_array.z, t_data_array.neighbor_threshold1, kind="linear", fill_value="extrapolate")
    thresh2 = interp1d(t_data_array.z, t_data_array.neighbor_threshold2, kind="linear", fill_value="extrapolate")

    def mass_limit(z):
        return np.minimum(1.3620186928378857 * z + 9.968545069745126, 11.2)

    def mass_coefficient(z):
        return np.exp(1.04935943 * z ** 2 + 0.39573094 * z + 0.28347756)

    massive_sample = massive_sample_original.copy()

    # Tree Algorithm
    tree = spatial.cKDTree(indexable[:, 1:3], copy_data=True)
    for i, row in iterator.iterrows():
        neighbors = tree.query_ball_point([row.x, row.y], radius_threshold(row.Z_PHOT_MEDIAN))
        if len(neighbors) > 0:
            local_data = indexable[neighbors]
            gauss = gauss_array[neighbors]

            z_c = z_threshold(row.Z_PHOT_MEDIAN)
            cylinder = np.abs(np.reshape(np.concatenate(gauss.flatten()), (len(gauss), oversample)) - row.Z_PHOT_MEDIAN)
            weights = (cylinder < 2 * z_c).sum(axis=1) / oversample

            deltas = local_data[:, 0] - row.Z_PHOT_MEDIAN
            idx = np.argsort(np.argsort(deltas))
            deltas_sorted = np.sort(deltas)
            reweights = interpolated_reweighting_extrapolate(deltas_sorted, row.Z_PHOT_MEDIAN)[idx]

            approx_cluster = np.column_stack([local_data, weights, weights * reweights])
            cluster = approx_cluster[approx_cluster[:, -2] > 0]

            if len(cluster) > 0:
                r_smaller = radius_threshold(row.Z_PHOT_MEDIAN)
                small_cluster = cluster[np.sqrt((cluster[:, 1] - row.x)**2 + (cluster[:, 2] - row.y)**2) < 0.5 * r_smaller]
                mini_cluster = cluster[np.sqrt((cluster[:, 1] - row.x)**2 + (cluster[:, 2] - row.y)**2) < 0.1 * r_smaller]

                massive_sample.at[i, "z_average_no_wt"] = np.mean(cluster[:, 0])
                massive_sample.at[i, "z_average_prob"] = np.average(cluster[:, 0], weights=cluster[:, -1])
                massive_sample.at[i, "z_average_mass_prob"] = np.average(cluster[:, 0], weights=cluster[:, -1] * cluster[:, 3])

                massive_sample.at[i, "neighbors"] = cluster[:, -2].sum()
                massive_sample.at[i, "neighbors_reweighted"] = cluster[:, -1].sum()

                mass_co = mass_coefficient(row.Z_PHOT_MEDIAN)
                c_mask = cluster[:, 3] > mass_limit(row.Z_PHOT_MEDIAN)
                cluster_limited = cluster[c_mask]
                massive_sample.at[i, "neighbor_mass"] = np.log10(np.sum((10 ** cluster_limited[:, 3]) * cluster_limited[:, -2]) * mass_co)

                membership = np.column_stack([
                    local_data[:, 4],  # galaxy
                    np.full(len(local_data), row.gid),  # cluster
                    local_data[:, 3],  # galaxy_mass
                    local_data[:, 0],  # galaxy_z
                    np.full(len(local_data), row.Z_PHOT_MEDIAN),  # cluster_z
                    local_data[:, 5],  # galaxy_z_std
                    reweights  # galaxy_reweight
                ])
                massive_sample.at[i, "neighbor_gids"] = membership

    clusters = massive_sample[(massive_sample.neighbors >= thresh1(massive_sample.Z_PHOT_MEDIAN)) &
                              (massive_sample.local_neighbors >= thresh2(massive_sample.Z_PHOT_MEDIAN))].copy()

    # Aggregation
    clusters.sort_values("neighbor_mass", inplace=True, ascending=False)
    clusters.reset_index(drop=True, inplace=True)

    tree = spatial.cKDTree(clusters[["x", "y"]])
    clusters["ncluster"] = 0
    clusternum = 1

    for i, row in clusters.iterrows():
        if clusters.at[i, "ncluster"] == 0:
            clusters.at[i, "ncluster"] = clusternum
            neighbors = tree.query_ball_point([row.x, row.y], 1.5 * radius_threshold(row.Z_PHOT_MEDIAN))
            for idx in neighbors:
                if clusters.at[idx, "ncluster"] == 0 and abs(clusters.at[idx, "Z_PHOT_MEDIAN"] - row.Z_PHOT_MEDIAN) < 2 * z_threshold(row.Z_PHOT_MEDIAN):
                    clusters.at[idx, "ncluster"] = clusternum
            clusternum += 1

    cluster_centers = clusters.sort_values(['ncluster', 'neighbor_mass'], ascending=[True, False]).groupby('ncluster').head(1)
    selected_centers = cluster_centers[(cluster_centers.RA < maxRA) & (cluster_centers.RA > minRA) & (cluster_centers.DEC < maxDEC) & (cluster_centers.DEC > minDEC)]

    membership_list = np.vstack(selected_centers.neighbor_gids.values)
    membership_df = pd.DataFrame(membership_list, columns=["galaxy", "cluster", "galaxy_mass", "galaxy_z", "cluster_z", "galaxy_z_std", "galaxy_reweight"])

    membership_df["prob"] = 2 * (1 - norm.cdf(abs(membership_df.galaxy_z - membership_df.cluster_z), scale=membership_df.galaxy_z_std))
    members_final = membership_df[membership_df.prob > 0.0027]

    return selected_centers, members_final
