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
def cluster_finder(massive_sample_original, iterrator, indexable, gauss_array, maxRA, maxDEC, minRA, minDEC, oversample, radius = 1, small_radius = 0.5, mini_radius = 0.1):
    #Mass fitting parameters and equations
    a = 1.3620186928378857  
    b = 9.968545069745126
    j= 1.04935943 
    k = 0.39573094 
    l = 0.28347756
    def mass_limit(z):
        return np.minimum((a*z + b), 11.2)

    def mass_coefficient(z):
        return np.exp(j*z**2 + k*z + l)
    
    #Radii
    radius = 1
    small_radius = 0.5
    mini_radius = 0.1
    #Cosmological Parameters: radius and cylinder length
    #Setting cylinder size
    cos = Cos(H0 = 69.6, Om0 = .286, Ode0 = .714)
    z_array = np.linspace(1e-2, 2, 500)
    sparse_radius = (1+z_array)/(cos.comoving_distance(z_array))
    radius_threshold = interp1d(z_array, sparse_radius, kind = ""linear"", fill_value = ""extrapolate"")
    z_source = Table.read(""threshold_training_south_2.fits"", format = ""fits"")
    #z_source = Table.read(""threshold_training_south.fits"", format = ""fits"")
    z_data_array = z_source.to_pandas()
    z_threshold = interp1d(z_data_array.z, z_data_array.sigma_z_threshold, kind = ""linear"", fill_value = ""extrapolate"")
    
    #Setting neighbor thresholds
    t_source = Table.read(""neighbor_training_south_2_2.fits"", format = ""fits"")
    #t_source = Table.read(""neighbor_training_south.fits"", format = ""fits"")
    t_data_array = t_source.to_pandas()
    thresh1 = interp1d(t_data_array.z, t_data_array.neighbor_threshold1, kind = ""linear"", fill_value = ""extrapolate"")
    thresh2 = interp1d(t_data_array.z, t_data_array.neighbor_threshold2, kind = ""linear"", fill_value = ""extrapolate"")
    
    #Reweightings
    bins_z = np.linspace(0.1, 1.0, 21)
    
    smoothed_reweightings = np.load(""reweighting_south_extrapolated.npy"")
    smoothed_reweightings[smoothed_reweightings == 0] = 1e-2
    edges = np.linspace(-.6, .6, 201)
    interpolated_reweighting_extrapolate = interp2d(x = 0.5*(edges[1:]+edges[:-1]), y = 0.5*(bins_z[:-1] + bins_z[1:]), z = smoothed_reweightings)
    
    
    #Mass fitting parameters and equations
    a = 1.3620186928378857  
    b = 9.968545069745126
    j= 1.04935943 
    k = 0.39573094 
    l = 0.28347756
    def mass_limit(z):
        return np.minimum((a*z + b), 11.2)

    def mass_coefficient(z):
        return np.exp(j*z**2 + k*z + l)
    
    #Tree Algorithm
    massive_sample = massive_sample_original.copy()
    tree = spatial.cKDTree(indexable[:, 1:3].astype(float), copy_data = True)
    for i, row in iterrator.iterrows():
        neighbors = tree.query_ball_point([row.x, row.y], radius_threshold(row.Z_PHOT_MEDIAN))
        if len(neighbors) > 0:
            local_data = indexable[neighbors]
            gauss = gauss_array[neighbors]
            
            z_c = z_threshold(row.Z_PHOT_MEDIAN)
            #cylinder = np.abs(np.vstack(gauss) - row.Z_PHOT_MEDIAN) #try reshape, try ravel, try flatten to make it 1-D. Try adding interp1D
            cylinder = np.abs(np.reshape(np.concatenate(gauss.flatten()), (len(gauss), oversample)) - row.Z_PHOT_MEDIAN)
            weight_array = cylinder < 2*z_c
            weights = weight_array.sum(axis = 1)/oversample
            deltas = local_data[:, 0] - row.Z_PHOT_MEDIAN
            idx = np.argsort(np.argsort(deltas))
            deltas_sorted = np.sort(deltas)
            reweights = interpolated_reweighting_extrapolate(deltas_sorted, row.Z_PHOT_MEDIAN)[idx]
            
            approx_cluster_0_0 = np.append(local_data, np.reshape(weights, newshape = (len(weights), 1)), axis = 1)
            approx_cluster = np.append(approx_cluster_0_0, np.reshape(weights*reweights, newshape = (len(weights), 1)), axis = 1)
            
            cluster = approx_cluster[approx_cluster[:, -2] > 0]
            if len(cluster)>0:
                r_smaller = radius_threshold(row.Z_PHOT_MEDIAN)
                small_cluster = cluster[np.sqrt(np.array((cluster[:, 1] - row.x)**2 + (cluster[:, 2] - row.y)**2, dtype = float)) < 0.5*r_smaller]
                mini_cluster = cluster[np.sqrt(np.array((cluster[:, 1] - row.x)**2 + (cluster[:, 2] - row.y)**2, dtype = float)) < 0.1*r_smaller]

                massive_sample.at[i, ""z_average_no_wt""] = np.mean(cluster[:, 0], dtype=np.float64)
                massive_sample.at[i, ""z_average_prob""] = np.average(cluster[:, 0], weights = cluster[:, -1])
                massive_sample.at[i, ""z_average_mass_prob""] = np.average(cluster[:, 0], weights = cluster[:, -1]*cluster[:, 3])

                std_no_weight = np.std(cluster[:, 0], dtype=np.float64)
                std_prob = np.sqrt(np.cov(cluster[:, 0], aweights = cluster[:, -1]))
                std_prob_mass = np.sqrt(np.cov(cluster[:, 0], aweights = cluster[:, -1]*cluster[:, 3]))
                massive_sample.at[i, ""z_std_no_wt""] = std_no_weight
                massive_sample.at[i, ""z_std_prob""] = std_prob
                massive_sample.at[i, ""z_std_mass_prob""] = std_prob_mass
                
                n_no_weight = len(cluster[:, 0])
                n_prob = np.sum(cluster[:, -1])**2/np.sum((cluster[:, -1])**2)
                n_mass_prob = np.sum(cluster[:, -1]*cluster[:, 3])**2/np.sum((cluster[:, -1]*cluster[:, 3])**2)
                massive_sample.at[i, ""z_stde_no_wt""] = std_no_weight/np.sqrt(n_no_weight)
                massive_sample.at[i, ""z_stde_prob""] = std_prob/np.sqrt(n_prob)
                massive_sample.at[i, ""z_stde_mass_prob""] = std_prob_mass/np.sqrt(n_mass_prob)

                massive_sample.at[i, ""neighbors""] = np.sum(cluster[:, -2])
                massive_sample.at[i, ""local_neighbors""] = np.sum(small_cluster[:, -2])
                massive_sample.at[i, ""ultra_local_neighbors""] = np.sum(mini_cluster[:, -2])
                
                massive_sample.at[i, ""neighbors_reweighted""] = np.sum(cluster[:, -1])
                massive_sample.at[i, ""local_neighbors_reweighted""] = np.sum(small_cluster[:, -1])
                massive_sample.at[i, ""ultra_local_neighbors_reweighted""] = np.sum(mini_cluster[:, -1])

                mass_co = mass_coefficient(row.Z_PHOT_MEDIAN)
                massive_sample.at[i, ""correction_factor""] = mass_co
                c_mask = cluster[:, 3]>mass_limit(row.Z_PHOT_MEDIAN)
                cluster_limited = cluster[c_mask.astype(""bool""), :]
                c_mask_small = small_cluster[:, 3]>mass_limit(row.Z_PHOT_MEDIAN)
                small_cluster_limited = small_cluster[c_mask_small.astype(""bool""), :]
                c_mask_mini = mini_cluster[:, 3]>mass_limit(row.Z_PHOT_MEDIAN)
                mini_cluster_limited = mini_cluster[c_mask_mini.astype(""bool""), :]
                massive_sample.at[i, ""neighbor_mass""] = np.log10(np.sum(((10**cluster_limited[:, 3]))*cluster_limited[:, -2])*mass_co)
                massive_sample.at[i, ""local_neighbor_mass""] = np.log10(np.sum((10**small_cluster[:, 3])*small_cluster[:, -2]))
                massive_sample.at[i, ""ultra_local_neighbor_mass""] = np.log10(np.sum((10**mini_cluster[:, 3])*mini_cluster[:, -2]))
                massive_sample.at[i, ""corr_local_neighbor_mass""] = np.log10(np.sum((10**small_cluster_limited[:, 3])*small_cluster_limited[:, -2])*mass_co)
                massive_sample.at[i, ""corr_ultra_local_neighbor_mass""] = np.log10(np.sum((10**mini_cluster_limited[:, 3])*mini_cluster_limited[:, -2])*mass_co)
                
                massive_sample.at[i, ""corr_neighbor_mass_reweighted""] = np.log10(np.sum(((10**cluster_limited[:, 3]))*cluster_limited[:, -1])*mass_co)
                massive_sample.at[i, ""corr_local_neighbor_mass_reweighted""] = np.log10(np.sum((10**small_cluster_limited[:, 3])*small_cluster_limited[:, -1])*mass_co)
                massive_sample.at[i, ""corr_ultra_local_neighbor_mass_reweighted""] = np.log10(np.sum((10**mini_cluster_limited[:, 3])*mini_cluster_limited[:, -1])*mass_co)
                
                clusterid = np.ones((1, len(local_data)))*row.gid
                clusterz = np.ones((1, len(local_data)))*row.Z_PHOT_MEDIAN
                membership = np.concatenate((local_data[:, 4].reshape((1, len(local_data))), clusterid, local_data[:, 3].reshape((1, len(local_data))), local_data[:, 0].reshape((1, len(local_data))), clusterz, local_data[:, 5].reshape((1, len(local_data))), reweights.reshape((1, len(local_data)))), axis = 0).T
                massive_sample.at[i, ""neighbor_gids""] = membership
                
                
            else:
                clusterid = np.ones((1, len(local_data)))*row.gid
                clusterz = np.ones((1, len(local_data)))*row.Z_PHOT_MEDIAN
                membership = np.concatenate((local_data[:, 4].reshape((1, len(local_data))), clusterid, local_data[:, 3].reshape((1, len(local_data))), local_data[:, 0].reshape((1, len(local_data))), clusterz, local_data[:, 5].reshape((1, len(local_data))), reweights.reshape((1, len(local_data)))), axis = 0).T
                massive_sample.at[i, ""neighbor_gids""] = membership
            
            
    #Thresholding
    clusters = massive_sample[np.logical_and(massive_sample.neighbors >= thresh1(massive_sample.Z_PHOT_MEDIAN), massive_sample.local_neighbors >= thresh2(massive_sample.Z_PHOT_MEDIAN))].copy()
    clusters.sort_values(""local_neighbor_mass"", inplace = True, ascending = False)
    clusters.reset_index(inplace= True, drop = True)
    
    #Aggregation
    tree = spatial.cKDTree(clusters[[""x"", ""y""]], copy_data = True)
    clusters[""ncluster""] = np.zeros(len(clusters))
    clusternum = 1
    iterrator2 = clusters.copy()
    for i, row in iterrator2.iterrows():
        if clusters.iloc[i].ncluster == 0:
            clusters.at[i, ""ncluster""] = clusternum
            neighbors = tree.query_ball_point([row.x, row.y], 1.5*radius_threshold(row.Z_PHOT_MEDIAN))
            for index in neighbors:
                if np.logical_and(clusters.at[index, ""ncluster""] == 0, np.abs(clusters.at[index, ""Z_PHOT_MEDIAN""] - row.Z_PHOT_MEDIAN) < 2*z_threshold(row.Z_PHOT_MEDIAN)):
                    clusters.at[index, ""ncluster""] = clusternum
                    #clusters.at[i, ""neighbor_gids""] = np.concatenate((clusters.at[i, ""neighbor_gids""], clusters.at[index, ""neighbor_gids""]), axis = 0) 
            clusternum += 1
    
    #Results
    cluster_center = clusters.sort_values(by = ['ncluster','ultra_local_neighbor_mass'], ascending = [True, False]).groupby('ncluster').head(1).copy()
    cluster_center_selected = cluster_center[np.logical_and.reduce((cluster_center.RA < maxRA, cluster_center.RA > minRA, cluster_center.DEC < maxDEC, cluster_center.DEC > minDEC))].copy()
    
    #Membership
    membership = pd.DataFrame(cluster_center_selected.neighbor_gids.values)
    membership_data = np.zeros((1, 7))
    for i in range(0, len(membership)):
        temp = np.stack(membership.values[i])[0]
        membership_data = np.concatenate([membership_data, temp], axis = 0)
    membershippd = pd.DataFrame(membership_data[1:], columns = [""galaxy"", ""cluster"", ""galaxy_mass"", ""galaxy_z"", ""cluster_z"", ""galaxy_z_std"", ""galaxy_reweight""], dtype = float)
    membershippd[""z_dist""] = np.abs(membershippd.galaxy_z - membershippd.cluster_z)
    membershippd.sort_values(""z_dist"", ascending = True, inplace = True)
    membershippd.drop_duplicates(subset = ""galaxy"", inplace = True)
    membershippd.reset_index(inplace = True, drop = True)
    
    membershippd[""prob""] = 2*(1-norm.cdf(x = np.abs(membershippd.galaxy_z-membershippd.cluster_z), loc = 0, scale = membershippd.galaxy_z_std))
    membershippd[""reweighted_prob""] = membershippd.prob * membershippd.galaxy_reweight
    memberspd = membershippd[membershippd.prob > 0.0027].astype({""galaxy"": ""int64"", ""cluster"": ""int64"", ""galaxy_mass"": ""float64""}).drop(columns = {""z_dist"", ""galaxy_reweight""})
    
    #Cleaning things up
    cluster_center_selected[""BRICKNAME""] = cluster_center_selected[""BRICKNAME""].astype('|S80')
    clusters_final = cluster_center_selected[[""RA"", ""DEC"", ""Z_PHOT_MEDIAN"", ""z_average_no_wt"", ""z_average_prob"", ""z_average_mass_prob"", ""Z_PHOT_STD"", ""z_std_no_wt"", ""z_std_prob"", ""z_std_mass_prob"", ""z_stde_no_wt"", ""z_stde_prob"", ""z_stde_mass_prob"", ""RELEASE"", ""BRICKID"", ""OBJID"", ""MASKBITS"", ""gid"", ""mass"", ""neighbor_mass"", ""corr_local_neighbor_mass"", ""corr_ultra_local_neighbor_mass"", ""corr_neighbor_mass_reweighted"", ""corr_local_neighbor_mass_reweighted"", ""corr_ultra_local_neighbor_mass_reweighted"", ""correction_factor"", ""neighbors"", ""local_neighbors"", ""ultra_local_neighbors"", ""neighbors_reweighted"", ""local_neighbors_reweighted"", ""ultra_local_neighbors_reweighted""]].copy()
    clusters_final.columns = [""RA_central"", ""DEC_central"", ""z_median_central"", ""z_average_no_wt"", ""z_average_prob"", ""z_average_mass_prob"", ""z_std_central"", ""z_std_no_wt"", ""z_std_prob"", ""z_std_mass_prob"", ""z_stde_no_wt"", ""z_stde_prob"", ""z_stde_mass_prob"", ""RELEASE"", ""BRICKID"", ""OBJID"", ""MASKBITS"", ""gid"", ""mass_central"", ""neighbor_mass"", ""local_neighbor_mass"", ""ultra_local_neighbor_mass"", ""corr_neighbor_mass_reweighted"", ""corr_local_neighbor_mass_reweighted"", ""corr_ultra_local_neighbor_mass_reweighted"", ""correction_factor"", ""neighbors"", ""local_neighbors"", ""ultra_local_neighbors"", ""neighbors_reweighted"", ""local_neighbors_reweighted"", ""ultra_local_neighbors_reweighted""]
    
    return clusters_final, memberspd