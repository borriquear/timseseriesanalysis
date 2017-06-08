#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
time series characterization

@author: jaime
"""
import pdb
import sys
import pandas as pd
import numpy as np
import os
from operator import itemgetter 
from collections import OrderedDict

import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from scipy.spatial.distance import euclidean
from scipy import stats
from scipy import signal

from datetime import datetime, date, time
import statsmodels.tsa.stattools as tsa
import nibabel as nib
from nilearn import input_data
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker 
from nilearn import plotting
from nilearn import datasets
from nilearn import input_data
from nilearn.connectome import ConnectivityMeasure
from nipy.labs.viz import plot_map, mni_sform, coord_transform
import seaborn as sns
from fastdtw import fastdtw
import nolds
import itertools
import nitime
# Import the time-series objects:
import nitime.timeseries as ts
# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer, GrangerAnalyzer, SeedCoherenceAnalyzer, CoherenceAnalyzer
from plotly import __version__
import plotly.plotly as py
py.plotly.tools.set_credentials_file(username='borriquear', api_key='ZGW7Nrb6GptJzSV7W6VY')
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from nitime.viz import drawmatrix_channels, drawgraph_channels
from mpl_toolkits.mplot3d import Axes3D
import warnings


def test_clustering_in_rs(epi_file=None):
    """Hierarchical clustering (Ward) in rs data
    """
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.feature_extraction import image
    import time
    from nilearn.plotting import plot_roi, plot_epi, show
    from nilearn.image import mean_img

    
    if epi_file is None:
        [epi_file, epi_params]= load_fmri_image_name()
        print " EPI file:", epi_file, " EPI params:", epi_params
    
    nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                                      mask_strategy='epi', memory_level=1,
                                      standardize=False)  
    #compute the mask and extracts the time series form the file(s)
    fmri_masked = nifti_masker.fit_transform(epi_file)
    #retrieve the nup array of the mask
    mask = nifti_masker.mask_img_.get_data().astype(bool)
    #compute the connectivity matrix for the mask
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)
    
    start = time.time()  
    #FeatureAgglomeration clustering algorithm from scikit-learn 
    n_clusters=100
    ward = FeatureAgglomeration(n_clusters, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
    ward.fit(fmri_masked)
    print("Ward agglomeration %d clusters: %.2fs" % (n_clusters, time.time() - start))
    #visualioze the results
    labels = ward.labels_ + 1
    labels_img = nifti_masker.inverse_transform(labels)
    mean_func_img = mean_img(epi_file)
    first_plot = plot_roi(labels_img, mean_func_img, title="Ward parcellation V-Alive",
                      display_mode='ortho', cut_coords=(0,-52,18))
    cut_coords = first_plot.cut_coords
    print cut_coords
    labels_img.to_filename('parcellation.nii')

    
def test_connectome_timeseries(epi_file=None, dim_coords=None):
    """ connectome time series characterization
    Run tests for collinearity, run_autoregression_on_df, stationarity and calculates the connectome of the given coords
    Example: test_connectome_timeseries() 
    test_connectome_timeseries(None, [(20,20,20),(30,12,7)])
    test_connectome_timeseries(None, 'cort-maxprob-thr25-2mm')
    test_connectome_timeseries(None, get_MNI_coordinates('DMN'))    
    """
    if epi_file is None:
        [epi_file, epi_params]= load_fmri_image_name()
        print " EPI file:", epi_file, " EPI params:", epi_params    
    if dim_coords is None:
        # Coordinates mask: masker = load_masker(dim_coords=[(x,y,z), (x,y,z)])
        # Harvard atlas mask: masker = load_masker(dim_coords='cort-maxprob-thr25-2mm')
        label = 'DMN'
        dim_coords = get_MNI_coordinates(label)
        masker = load_masker(dim_coords)
    else:
        masker = load_masker(dim_coords)
    time_series = masker.fit_transform(epi_file)
    timepoints,  nb_oftseries = time_series.shape
    print "Plotting time series", timepoints,  nb_oftseries
    plot_ts_network(time_series, dim_coords.keys())
    if time_series.shape[0] != epi_params:
        warnings.warn("The time series number of points is !=116 check your bold_data.nii !!", Warning)
    print "Masker is:", masker, " \n \t and time series shape", time_series.shape
    ts_nonlin_characterization = False
    if ts_nonlin_characterization is True:
        print "Testing for collinearity in the time series"
        res_lagged = run_collinearity_on_df(time_series)
        print "Calling to run_autoregression_on_ts(y) to test for autocorrelation in the time series" 
        res_acf = run_autoregression_on_df(time_series)      
        print "Calling to run_test_stationarity_adf(y) to test for stationarity using augmented Dickey Fuller test" 
        res_stationarity = run_test_stationarity_adf(time_series)                                    
        print "Calling to ARIMA model fit"
        res_arima = run_arima_fit(time_series, order=None)
       #print "Calling to run_forecasting to study predictability of the time series for the ARIMA"
       #res_forecasting = run_forecasting(res_arima)
       #return masker, time_series, res_lagged, res_acf, res_stationarity, res_arima 
    display_connectome = False
    if display_connectome is True:
        kind_of_correlation = 'correlation'
        print "Displaying the connectome for corr_type:", kind_of_correlation
        msgtitle = kind_of_correlation + ' DMN (Cad)'
        # Plotting time series and network in brain overlayed using  nilearn.connectome.ConnectivityMeasure
        corr_matrix = plot_ts_connectome(time_series, kind_of_correlation)
        #plotting kind_of_correlation as a heatmap from nitime
        fig_C_drawx = drawmatrix_channels(corr_matrix, dim_coords.keys(), size=[10., 10.], color_anchor=0, title= msgtitle)            
        #plotting kind_of_correlation as a network with nodes of variable size and label in it from nitime
        fig_C_drawg = drawgraph_channels(corr_matrix, dim_coords.keys(),title=msgtitle)

    testforGranger = False
    if testforGranger is True:
        #Granger causality test of the time series
        grangerresult = test_of_grangercausality(time_series)
        # Calculate the G causality among pairs of time series  
        G = compute_granger_pair_of_ts(time_series)
        freq_idx_G = np.where((G.frequencies > 0) * (G.frequencies <= 0.2))[0]
        listoffreqs = G.frequencies[:]
        #for ff in range(0, len(listoffreqs)):
            #print "Freq:", ff, " / ", listoffreqs, " idx ", freq_idx_G
            #print "XY Granger at freq:", listoffreqs[ff], " =", G.causality_xy[:,:,freq_idx_G[ff]]
            #print "YX Granger at freq:", listoffreqs[ff], " =", G.causality_yx[:,:,freq_idx_G[ff]]
            # average on last dimension (frequency and plot)
        msgtitle = 'Granger "causality" DMN (Cad)'
        #plotting GC as a heatmap from nitime
        fig_G_drawx = drawmatrix_channels(np.mean(G.causality_xy[:, :, freq_idx_G], -1), dim_coords.keys(), size=[10., 10.], color_anchor=0, title= msgtitle)            
        #plotting GC as a network with nodes of variable size and label in it from nitime
        fig_G_drawg = drawgraph_channels(np.nan_to_num(np.mean(G.causality_xy[:, :, freq_idx_G], -1)), dim_coords.keys(),title=msgtitle)
         
    fourier_analysis = True
    if fourier_analysis is True:
        print "Calculating the PSD of the time series"
        psd_original = fourier_spectral_estimation(time_series)
        #compute coherence
        corr_type = 'Mean Coherence across frequencies (Cad)'
        [coherence_mat, seeds, targets] = compute_coherence_pairs_of_ts(time_series)
        print "Plotting functional connectivity of the Frequency spectrum"
        plot_coupling_heatmap(coherence_mat, seeds, targets, dim_coords.keys(), corr_type=corr_type)
        msgtitle = 'Coherency DMN (Cad)'
        #SeedCoherency
        compute_seed_coherence(time_series)
                       
    surrogate_analysis = False
    if surrogate_analysis is True:
        num_realizations = 10
        nullmodel = 'gaussian noise'
        surrogate_data = generate_surrogate_data(time_series, num_realizations, nullmodel)
        print "Built the surrogate_data data frame dimesnion:", surrogate_data.shape
        correlation_ts = correlation_of_ts_vs_surrogate(time_series, surrogate_data)
        print "The correlation mean  between original time series and the surrogate is:, ", correlation_ts[0], \
        "\n and the std is:", correlation_ts[1]
        #generate surrogate data with other models: orsntein-uhlenbeck regrression to the mean(homeostatic)
        print "Computing statistics for non linear quantities(correlation dimension, \
        Lyapunov exponents and sample Entropy for the Original time series and \
        the surrogate data generated for the ",nullmodel , " model"
        nonlstats_orig= []
        nonlstats_surrogate = []
        for i in range(0,nb_oftseries):
            nonlstats_ith_surrogate = []
            original_stats = compute_nonlinear_statistics(time_series[:,i])
            print "The nonlinear statistics for ith:", i, "  Original timeseries: ", original_stats
            nonlstats_orig.append(original_stats)
            for j in range(0, num_realizations): 
                jindex = num_realizations*i + j
                surrogate_stats = compute_nonlinear_statistics(surrogate_data.iloc[:,jindex])
                nonlstats_ith_surrogate.append(surrogate_stats)
            nonlstats_surrogate.append(nonlstats_ith_surrogate)    
        print "The nonlinear statistics for the Original time series is :", nonlstats_orig, 
        "\n to access the resulst: nonlstats_orig[0]['lyap'|'corr_dim'|sampen] \n"
        print "\n\n The nonlinear statistics for the Surrogate time series is :", nonlstats_surrogate, 
        "\n to access the resulst: nonlstats_surrogate[0][9]['lyap'|'corr_dim'|sampen] \n"
        ts_and_ps = ttest_orig_vs_surrogate(nonlstats_orig, nonlstats_surrogate, realizations=num_realizations)
    
    seed_based_connectome = False
    if seed_based_connectome is True:
        if masker is None:
            label = 'DMN'
            dim_coords = get_MNI_coordinates(label)
            masker = load_masker(dim_coords) 
        seed_masker = masker
        if time_series is None:
            time_series = seed_masker.fit_transform(epi_file)
        seed_time_series = time_series    
        brain_masker = load_masker('brain-wide')
        brain_time_series = brain_masker.fit_transform(epi_file)
        print("seed time series shape: (%s, %s)" % seed_time_series.shape)
        print("brain time series shape: (%s, %s)" % brain_time_series.shape)
        #select the seed
        seed_time_series = seed_time_series[:,0]
        seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / \
                          seed_time_series.shape[0]
        #pdb.set_trace()
        print "seed-based correlation shape", seed_based_correlations.shape
        print "seed-based correlation: min =", seed_based_correlations.min(), " max = ", seed_based_correlations.max()
        #Fisher-z transform the data to achieve a normal distribution. 
        #The transformed array can now have values more extreme than +/- 1.
        seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
        print "seed-based correlation Fisher-z transformed: min =", seed_based_correlations_fisher_z.min(), \
        " max =", seed_based_correlations_fisher_z.max()                                                                                             
        seed_based_correlation_img = brain_masker.inverse_transform(seed_based_correlations.T)
        seed_based_correlation_img.to_filename('sbc_z.nii.gz')
        pcc_coords = dim_coords.values()[0]
        #pcc_coords = [(0, -52, 18)]
        MNI152Template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_symmetric.nii.gz'
        #remove out-of brain functional connectivity using a mask
        icbms = datasets.fetch_icbm152_2009()
        masker_mni = NiftiMasker(mask_img=icbms.mask)
        data = masker_mni.fit_transform('sbc_z.nii.gz')
        masked_sbc_z_img = masker_mni.inverse_transform(data)
        display = plotting.plot_stat_map(masked_sbc_z_img , cut_coords=pcc_coords, threshold=0.6, title= 'PCC-based corr. V-A', dim='auto', display_mode='ortho')
        #Coherency

        #pdb.set_trace()
        #display.add_markers(marker_coords=pcc_coords, marker_color='g', marker_size=300)
        # At last, we save the plot as pdf.
        #display.savefig('sbc_z.pdf')
        #correlate the seed signal with the signal of each voxel. \
        #The dot product of the two arrays will give us this correlation. 
        #Note that the signals have been variance-standardized during extraction. 
        #To have them standardized to norm unit, we further have to divide the 
        #result by the length of the time series.
        
def  load_fmri_image_name(image_file=None):
    """ load a bold data file 
    
    image_file: path of the bold_data.nii
    return the path of the image and the parameters (TR, n,fs, nfft, duration_in_s)
    """
    # Load fmri image (4D object)
    image_params = {'TR':2.5, 'n':120-4, 'fs': 0.4, 'nfft':129,'duration_in_s':120*2.5}
    image_params = {'TR':2.5, 'n':120, 'fs': 0.4, 'nfft':129,'duration_in_s':120*2.5}
    if image_file is None:
        dir_name = '/Users/jaime/vallecas/mario_fa/RF_off'
        dir_name = '/Users/jaime/vallecas/data/cadavers/nifti/bcpa_0537/session_1/PVALLECAS3/reg.feat'
        dir_name = '/Users/jaime/vallecas/data/surrogate/bcpa0537_1'
        f_name = 'bold_data_mcf2standard.nii'
        f_name = 'wbold_data.nii'
        image_file = os.path.join(dir_name, f_name)
    return image_file, image_params

def plot_ts_network(ts, lab):
    # plot_ts_network  Plot time series for a network 
    # Input: ts time series object
    # lab: list of labels
    plt.figure()
    for ts_ix, label in zip(ts.T, lab):
        plt.plot(np.arange(0,len(ts_ix)), ts_ix, label=label)
        
    plt.title('Default Mode Network Time Series')
    plt.xlabel('Time points (TR=2.5s)')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.tight_layout()
    
def plot_ts_connectome(ts, kind_of_corr=None, labelnet=None):
    """plot correlation netween time series using nilearn.plotting.plot_connectome
    
    ts: time series at least 2
    kind_of_corr:  {“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
    labelnet: label on top ledt of the overlayed brain figure. If none DMN   
    
    """
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    if kind_of_corr is None:
        kind_of_corr='correlation'
    if labelnet is None:
        labelnet='DMN'  
        #connectivity_measure = ConnectivityMeasure(kind=kind_of_corr) 
        #connectivity_measure = ConnectivityMeasure(EmpiricalCovariance(assume_centered=False, block_size=1000, store_precision=False), kind=kind_of_corr) 
        connectivity_measure = ConnectivityMeasure(EmpiricalCovariance(assume_centered=True), kind=kind_of_corr) 
        correlation_matrix = connectivity_measure.fit_transform([ts])[0] 
        #print  correlation_matrix
        coords_dict = get_MNI_coordinates(labelnet)  
        plotting.plot_connectome(correlation_matrix, coords_dict.values(),edge_threshold='05%',
                         title=labelnet,display_mode="ortho",edge_vmax=.5, edge_vmin=-.5)
        return correlation_matrix    
           
def load_masker(dim_coords=None):   
    """Returns the Masker object from an Atlas (mask_file) or a list of voxels
    
    Input: dim_coords None || [] returns None (in the future will return a list of brain voxels, the full brain).
    dim_coords: dictionary of a network of interest eg DMN.
    dim_coords: list of coordinates of voxels.    
    dim_coords: atlas_harvard_oxford, eg 'cort-maxprob-thr25-2mm'  
    dim_coords: 'brain-wide'

    Example: load_masker() #returns masker for the entire brain
    load_masker(get_MNI_coordinates('MNI'))  #returns masker for the the MNI coordinates
    load_masker([(0, -52, 18),(10, -52, 18)]) #returns masker for a list of coordinates 
    load_masker('cort-maxprob-thr25-2mm') #returns masker for the atlas_harvard_oxford atlas thr25 and 2mm
                                             
    """
    standarize = True # standardize If standardize is True, the time-series are centered and normed: their mean is set to 0 and their variance to 1 in the time dimension.
    radius = 8 #in mm. Default is None (signal is extracted on a single voxel
    smoothing_fwhm = None # If smoothing_fwhm is not None, it gives the full-width half maximum in millimeters of the spatial smoothing to apply to the signal.

    if dim_coords is None or len(dim_coords) == 0:    
        print "No mask used, process the entire brain"
        return None
    elif type(dim_coords) is dict or type(dim_coords) is OrderedDict:  
        # Extract the coordinates from the dictionary
        print " The mask is the list of voxels:", dim_coords.keys(), "in MNI space:", dim_coords.values()
        masker = input_data.NiftiSpheresMasker(dim_coords.values(), radius=radius,
                                               detrend=True, smoothing_fwhm=smoothing_fwhm,standardize=standarize,
                                               low_pass=0.2, high_pass=0.001, 
                                               t_r=2.5,memory='nilearn_cache', 
                                               memory_level=1, verbose=2, allow_overlap=False)
        print masker
    elif type(dim_coords) is list:  
        # Extract the coordinates from the dictionary
        print " The mask is the list of voxels:", dim_coords
        masker = input_data.NiftiSpheresMasker(dim_coords, radius=radius,
                                               detrend=True, smoothing_fwhm=smoothing_fwhm,standardize=standarize,
                                               low_pass=0.2, high_pass=0.001, 
                                               t_r=2.5,memory='nilearn_cache', 
                                               memory_level=1, verbose=2) 
    elif dim_coords == 'brain-wide':
        # Extract the maksker of the entire brain
        masker = input_data.NiftiMasker(smoothing_fwhm=6, detrend=True, standardize=standarize,
                                              low_pass=0.2, high_pass=0.001, t_r=2.5,
                                              memory='nilearn_cache', memory_level=1, verbose=2)
    else:
        # The mask is an Atlas
        dataset = datasets.fetch_atlas_harvard_oxford(dim_coords)
        atlas_filename = dataset.maps
        plotting_atlas = False
        if plotting_atlas is True:
            plotting.plot_roi(atlas_filename)
        # Build the masker object
        masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=False)
        #time_series = masker.fit_transform(image_file)
    return masker
        
def createDataFrame_from_image(image_file=None, masker=None):
    # createDataFrame to work with pandas time series from nifti image and Masker object
    # If no mask, the mask is the entire brain and return pvalues array[x,y,zpavlue] if mask is given return the time series for each 
    # ROI of the mask. nomask doesnt return time series but 1 if the voxel is non stationay 0 if it is.
    if image_file is None:
        # Load image
        image_file, image_params = load_fmri_image_name(image_file=None)
        #data_path = '/Users/jaime/vallecas/mario_fa/mario_fa_dicoms/FA10_SIN_rf' 
        #image_file = '20170126_172022rsfMRIFA4s005a1001.nii.gz'
    image_data = nib.load(image_file).get_data()
    dims_image = image_data.shape
    # Create DataFrame from ndarray
    # Generate the voxels_list All brain voxels or specific voxel coordinates that correspond to areas of interest
    if masker is None:
        # The mask is the entire brain
        tot_voxels = ((dims_image[0]-1)*(dims_image[1]-1)*(dims_image[2]-1),3)
        print "Total number of voxels ", tot_voxels
        #voxels_list = np.zeros(tot_voxels)
        voxels_list = []
        for i in np.arange(0, dims_image[0]-1):
            label = []
            for j in np.arange(0, dims_image[1]-1):
                for k in np.arange(0, dims_image[2]-1):
                    label = [i,j,k]
                    voxels_list.append(label)
                    
        pvalues = extract_ts_from_brain(image_data, voxels_list)
        return pvalues
        # plot number of voxels that are non stationary
        
    else:
        # The masker can be an Atlas or a list of voxels 
        time_series = masker.fit_transform(image_file)   
    return time_series

def compute_seed_coherence(time_series, image_file=None, image_params=None):
    """compute seed based coherency analysis for one seed and the brain
    
    time_series: time series of theseeds
    """
    import nitime.fmri.io as io
    f_lb = 0
    f_ub = 0.2
    if image_file is None: [image_file,image_params] = load_fmri_image_name()
    #ts of the only one seed passed as an argument
    #ts_seed = np.transpose(time_series)
    mni_coordinates = get_MNI_coordinates('DMN')
    seeds = mni_coordinates.values()
    coords_seeds = np.array(seeds[0]).T
    volume_shape= nib.load(image_file).get_data().shape[:-1]
    coords = list(np.ndindex(volume_shape))
    coords_target = np.array(coords).T
    time_series_seed = io.time_series_from_file(image_file,
                                coords_seeds,
                                TR=image_params['TR'],
                                normalize='percent',
                                filter=dict(lb=f_lb,
                                            ub=f_ub,
                                            method='boxcar'))
    time_series_target = io.time_series_from_file(image_file,
                                          coords_target,
                                          TR=image_params['TR'],
                                          normalize='percent',
                                          filter=dict(lb=f_lb,
                                                      ub=f_ub,
                                                    method='boxcar'))
    #remove nan
    [nn, pp] = time_series_target.shape
    for i in range(0,nn):
        time_series_target.data[i] = nitime.utils.thresholded_arr(time_series_target.data[i], fill_val=0.)
    print "Calculating     SeedCoherenceAnalyzer"
    pdb.set_trace()
    A = SeedCoherenceAnalyzer(time_series_seed, time_series_target)
    freq_idx = np.where((A.frequencies > f_lb) * (A.frequencies < f_ub))[0]
    coh = []
    coh.append(np.mean(A.coherence[0][:, freq_idx], -1))
    coords_indices = list(coords_target)
    vol_coh = []
    vol_coh.append(np.empty(volume_shape))
    vol_coh[-1][coords_indices] = coh[0]
    
    
def compute_coherence_pairs_of_ts(time_series, image_params=None):
    """spectral analysis of the signal for the coherence. It uses CoherenceAnalyzer 
    and or SeedCoherenceAnalyzer, (nitime.analysis) are equivalent.    
    """
    f_lb = 0
    f_ub = 0.2
    if image_params is None: image_params = load_fmri_image_name()[1]
    combinations = list(itertools.combinations(range(time_series.shape[1]),2)) 
    time_series = np.transpose(time_series)
    #time_series = nitime.timeseries.TimeSeries(time_series, sampling_interval=image_params['TR'])
    seeds = []
    targets = []
    for c in combinations:
        seeds.extend([c[0]])
        targets.extend([c[1]])
    #remove repes
    seeds = list(set(seeds))
    targets = list(set(targets))
    ts_seed = time_series[seeds]
    ts_target = time_series[targets]
    T = ts.TimeSeries(np.vstack([ts_seed, ts_target]), sampling_interval=image_params['TR'])
    C1 = CoherenceAnalyzer(T)
    print "Coherence seed:",seeds, " target:", targets, " = ", C1.coherence[0, 1], "\n Delay:", C1.delay[0, 1], "\n phases:", C1.phase[0, 1]
    #C2 = CoherenceAnalyzer(T)
    #print "Seed Coherence:", C2.coherence[1], " Delay:", C2.delay[1], " rphases:", C2.relative_phases[1]
    freq_idx_C1 = np.where((C1.frequencies > f_lb) * (C1.frequencies < f_ub))[0]
    rows, cols, freqs = C1.coherence[:, :, freq_idx_C1].shape # coh.shape
    rows2 = len(seeds)
    cols2 = len(targets)
    #obtain minimum matrix(Seeds x Targets), elimating redundant pairs 
    #coh_seedsxtargets = coh[:rows2,cols2:]
    coh_seedsxtargets =  C1.coherence[:rows2, cols2:, freq_idx_C1]     
    print "Calculating Coherence for Seeds:", seeds, " x targets:", targets, " for frequencies : ", C1.frequencies
    print " Coherence SeedxTarget for each frequency: [[seed-target]freq]", C1.coherence[:rows2, cols2:, freq_idx_C1]
    meanoverf = np.mean(C1.coherence[:rows2, cols2:, freq_idx_C1], -1)
    print "\n The Mean coherence over frequencies is:\n", meanoverf
    #fig01 = drawmatrix_channels(meanoverf, ['a','b','c','d'], size=[10., 10.], color_anchor=0)
    return meanoverf,seeds,targets

def plot_coupling_heatmap(corrmat, seeds, targets, labels, corr_type=None):
    """plot heatmap from a corrrelation matrix 
    
    corrmat: Coherenceanalyzer object
    seeds: list of elements in the origin
    targets: list of elemensts in the target
    labels: name of the ROI whose correlation arre being displayed
    example: plot_correlation_heatmap(corrat, [0,1,2], [1,2,3], dim_coords.keys(), 'coherence')  
    """
    print "Calling to  pyplot version:",  __version__ # requires version >= 1.9.0 
    seed_labels = itemgetter(*seeds)(labels) 
    #target_labels = list(reversed(itemgetter(*targets)(labels)))
    target_labels = itemgetter(*targets)(labels)
    trace = go.Heatmap(z=corrmat,x=target_labels , y=seed_labels)
    data=[trace]
    #https://plot.ly/python/axes/
    layout = go.Layout(
                title=corr_type,
                width = 500, height = 500,
                yaxis=dict(tickfont=dict(size=12), tickangle=-45),
                xaxis=dict(tickfont=dict(size=12)), 
                autosize = False,
                margin=go.Margin(l=100,r=50,b=100,t=100,pad=4)
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)
    

def compute_granger_pair_of_ts(time_series, image_params=None):
    """Calculate the Granger causality for pairs of tiem series using nitime.analysis.coherence
    """
    
    if image_params is None: image_params = load_fmri_image_name()[1]
    #frequencies = np.linspace(0,0.2,129)
    combinations = list(itertools.combinations(range(time_series.shape[1]),2)) 
    time_series = np.transpose(time_series)
    time_series = nitime.timeseries.TimeSeries(time_series, sampling_interval=image_params['TR'])
    
    granger_sys = GrangerAnalyzer(time_series, combinations)
    listoffreqs = granger_sys.frequencies[:]
    corr_mats_xy = granger_sys.causality_xy[:,:,:]
    corr_mats_yx = granger_sys.causality_yx[:,:,:]
    return granger_sys #[corr_mats_xy, corr_mats_yx, listoffreqs]

def test_of_grangercausality(time_series):
    """Test for Granger Cauality in the original time series. The Null hypothesis for grangercausalitytests is that the time series \
    in the second column, x2, does NOT Granger cause the time series in the first column, x1. \
    The null hypothesis for all four test is that the coefficients corresponding \
    to past values of the second time series are zero.
    """
    nb_of_timeseries = len(time_series)
    Gres_all = []
    if nb_of_timeseries < 2:
         warnings.warn("ERROR: We cant calculate Granger for only 1 tie series", Warning)
    else:
        combinations = list(itertools.combinations(range(time_series.shape[1]),2)) 
        for co in combinations:
            pairofseries = [pd.Series(time_series[:,co[0]]), pd.Series(time_series[:,co[1]])]
            pairofseries = np.asarray(pairofseries)
            pairofseries = np.transpose(pairofseries)
            print " Calculating Granger test for timeseries pair:", co
            grangerres = grangercausalitytests(pairofseries, maxlag=10, verbose=False)
            Gres_all.append(grangerres)
        return Gres_all
    
def run_collinearity_on_df(y):
    """Test for multicollinearity for y (Pandas.time series or ndarray) 
    Multicollinearity (collinearity) is a phenomenon in which two or more predictor variables
    in a multiple regression model are highly correlated. Shift the timeseries lag index times 
    and calculate the correlation between the coefficients in the regression.
        
    
    """
    # Test for multicollinearity check type of data first
    if type(y) is pd.core.series.Series:
        run_collinearity_on_ts(y)
    if type(y) is np.ndarray:
        # Convert df into timeseries and call each time
        summary_list = []
        #return mask
        for index in np.arange(0,y.shape[1]):
            print "Estimating collinearity for ts ROI:", index, "/", y.shape[1]-1
            # print pd.Series(y[:,index])
            res_lagged = run_collinearity_on_ts(pd.Series(y[:,index]),index)
            summary_list.append(res_lagged)
    return summary_list           

def run_collinearity_on_ts(y, index=None):
    """Test for multicollinearity of time series
        
    """
    if index is None: index=0
    lags = 6
    # Create a dataframe with lagged values of timeseries (y), shifts the index i periods
    df = (pd.concat([y.shift(i) for i in range(lags)], axis=1,keys=['y'] + ['y%s' % i for i in range(1, lags)]).dropna())
    df.head()
    #return df
    # Fit the lagged model using statsmodels
    mod_lagged = smf.ols('y ~ trend + y1 + y2 + y3 + y4 + y5',data=df.assign(trend=np.arange(len(df))))
    res_lagged = mod_lagged.fit()
    print res_lagged.summary()
    # If there is no multicollinearity, the coefficient estimation is max in abs value for \beta1 and min for \beta5
    # If  the lagged values are highly correlated with each other the estimates of the slopes are not reliable
    plotresults = True
    if plotresults is True:
        plot_collinearity_on_ts(res_lagged, df,"timeseries:%d" % (index))
    return res_lagged

def plot_collinearity_on_ts(res_lagged,df,titlemsg):
        plt.figure()
        plt.subplot(2,1,1)
        sns.heatmap(df.corr(),vmin=-1.0, vmax=1.0) 
        titlemsg2 = "Shifted " + titlemsg + "  correlation"
        plt.title(titlemsg2)      
        plt.subplot(2,1,2)
        #axarr[0,1] = plt.axes()
        #ax2.set_title('Correlation of the lagged coefficients: y ~ b(0) + ...+ b(5)')
        res_lagged.params.drop(['Intercept', 'trend']).plot.bar(rot=0)
        plt.ylim(-1,1)
        plt.ylabel('Coefficient')
        titlemsg = 'Correlation of the lagged coefficients: y ~ b(0) + ...+ b(5)'
        plt.title(titlemsg)
        sns.despine()  
        #plt.show() 
        
def run_autoregression_on_df(y):
    """Run autoregression on a dataframe
       Autocorrelation also called serial correlation is observed  for 
       patterns between the observed and the predicted in the regression model.
    
    """
    if type(y) is pd.core.series.Series:
        run_autoregression_on_ts(y)
    if type(y) is np.ndarray:
        # Convert df (np.ndarray) into timeseries and call each time
        summary_list = []
        for index in np.arange(0,y.shape[1]):
            print "Estimating autoregression for ts ROI:", index, "/", y.shape[1]-1 
            res_trend  = run_autoregression_on_ts(pd.Series(y[:,index]), index)                                                                
            summary_list.append(res_trend)                                                                
            print res_trend.summary()
                                                                            
    return summary_list        
                                                                
def run_autoregression_on_ts(y,index=None):  
    """Run autoregression on a timeseries (ndarray)
    """
    if index is None: index=0
    mod_trend = sm.OLS.from_formula('y ~ trend', data=y.to_frame(name='y').assign(trend=np.arange(len(y))))
    res_trend = mod_trend.fit()
    # Residuals (the observed minus the expected, or $\hat{e_t} = y_t - \hat{y_t}$) are supposed to be white noise. 
    # Plot the residuals time series, and some diagnostics about them
    plotacfpcf = True
    if plotacfpcf is True:    
        plot_autoregression_on_ts(res_trend.resid,msgtitle = "timeseries:%d" % (index), lags=36)
    return res_trend

def plot_autoregression_on_ts(y,msgtitle,lags=None):
    """
    """
    figsize=(10, 8)
    fig = plt.figure(figsize=figsize)    
    layout = (2, 2)           
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)   
    acf_ax = plt.subplot2grid(layout, (1, 0))  
    pacf_ax = plt.subplot2grid(layout, (1, 1)) 
    msgtitle = "Regression residuals " + msgtitle
    msgtitle= msgtitle + " :y_t - \hat{y_t}"
    y.plot(ax=ts_ax, title=msgtitle) 
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)        
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)    
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]] 
    sns.despine()                            
    plt.tight_layout() 
    #plt.title(msgtitle)
    return ts_ax, acf_ax, pacf_ax  

def run_test_stationarity_adf(y):
    """Test stationarity of the dataframe using the dickey fuller test
    http://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
    autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}
    H0: ts has unit root (non stationary)
    """
    if type(y) is pd.core.series.Series:
        run_test_stationarity_adf_ts(y)
    if type(y) is np.ndarray:    
        # Convert df (np.ndarray) into timeseries and call each time
        summary_list = []
        for index in np.arange(0,y.shape[1]):
           print "===Estimating stationarity for ts ROI:", index, "/", y.shape[1]-1                                                               
           res_trend  = run_test_stationarity_adf_ts(pd.Series(y[:,index]), index)   
           summary_list.append(res_trend)                                                                          
    return summary_list                                                                        
                                                                             
def run_test_stationarity_adf_ts(timeseries, index=None):
    """ dickey fuller test of stationarity for time series
    H0: ts has unit root (non stationary)
    """ 
    if index is None: index = 0
    toplot = False
    window = 10
    #Plot rolling statistics
    if toplot is True:
        plt.figure()
        rolmean = pd.rolling_mean(timeseries, window = window)
        rolstd = pd.rolling_std(timeseries, window = window)
        orig = plt.plot(timeseries, color = 'blue', label = 'Original')
        mean = plt.plot(rolmean, color = 'red', label = 'Rolling mean')
        std = plt.plot(rolstd, color = 'black', label = 'Rolling std')
        plt.legend(loc='best')
        msgtitle = "Rolling (window:" + ` window`+ ")" +" mean and std deviation timseries: " + `index`
        plt.title(msgtitle)
    #Perform Dickey-Fuller test
    autolag = 'BIC'
    #print 'Results of Dickey Fuller test with ', autolag
    dftest = adfuller(timeseries,maxlag=None, regression='c', autolag=autolag, store=False, regresults=False)
    # print dftest
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print "Timseries:", index, "H0: ts has unit root (non stationary). p-value:", dfoutput['p-value'] 
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value                
    return dfoutput

def run_arima_fit(y, order=None):
    """Fit a dataframe with ARIMA(order) model
    Calls to run_arima_fit_ts to fit timeseries with a ARIMA (SRIMAX) sesonal 
    autoregressive integrated moving average with exogenbeous regressors 
    Example: run_arima_fit(timeseries, order = [1,1,1])
    """
    if type(y) is pd.core.series.Series:
        run_arima_fit_ts(y, order)
    if type(y) is np.ndarray:   
        # Convert df (np.ndarray) into timeseries and call each time
        summary_list = []
        for ind in np.arange(0,y.shape[1]):
            print "Fitting ARIMA, ROI:", ind, "/", y.shape[1]-1   
            #res_trend  = run_arima_fit_ts(pd.Series(y[:,ind]),order) 
            #You need datetime index when you use pandas, but you can use a numpy ndarray for the series which does not need any index.
            try:
                res_trend = []
                res_trend  = run_arima_fit_ts(y[:,ind], ind, order)
            except:
                warnings.warn("ERROR: The ARIMA model is not invertible!!!", Warning)
            summary_list.append(res_trend) 
    print "ARIMA per time series objects created:", summary_list        
    return summary_list                                                          
        
def run_arima_fit_ts(y, indexts=None, order=None):
    """Fit timeseries with a ARIMA (SRIMAX) sesonal autoregressive integrated moving average with exogenbeous regressors 
    """
    res_arima = []
    if indexts is None: indexts=0                              
    if order is None: 
        pe = 1
        de = 0
        qu = 1
    else: 
        pe = order[0]
        de = order[1]
        qu = order[2]
    #res_arima = sm.tsa.ARIMA(y, order=(p,d,q)).fit(method='mle', trend='nc')
    try:
        res_arima = sm.tsa.ARIMA(y, order=(pe,de,qu)).fit(method='mle', trend='nc')
        print "ARIMA summary"
        print res_arima.summary()
        #print "ARIMA AIC=", res_arima.aic
        arima_plot = True
        if arima_plot is True:
            msgtitle= ' for Timeseries:' + `indexts`+ ' using ARIMA(p,d,q):'+ `pe` + `de` + `qu`
            plot_autoregression_on_ts(pd.Series(res_arima.resid[2:]), msgtitle,lags=36)
    except:
            warnings.warn("ERROR: The ARIMA model is not invertible!!!", Warning)
    return res_arima    
                                      
def run_forecasting(res_arima):
    """ forecasting for ARIMA(model)
    """
    # number of time series to plot
    
    nb_of_ts =  len(res_arima)
    for i in range(0,nb_of_ts):
        pred = res_arima[i].predict()
        print "Prediction for ts:", i, " :", pred
        #pred_ci = pred.conf_int()
        ax = res_arima[i].plot_predict()
        #pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
        #ax.fill_between(pred.index,pred.iloc[:, 0],res_arima, color='k', alpha=.2)å
#        plt.legend()
#        msgtitle = 'Predictability of the timseries:' + `i` + ' using ARIMA'
#        plt.title(msgtitle)
#        sns.despine()

def extract_ts_from_one_voxel(image_file, voxel=[]):
    # Obtain the time series from a voxel
    image_data = nib.load(image_file).get_data()
    dims_image = image_data.shape
    timeseries = pd.Series(image_data[voxel[0],voxel[1],voxel[2]])
    print "Getting time series for voxel:", voxel, "\n is:", timeseries
    return timeseries
    
def extract_ts_from_brain(image_data, voxels_list=[]):
    #        
    dim_brains = image_data.shape
    voxels_all = []
    #df = pd.DataFrame(image_data[0, 0, 0])
    # threshold pvalue for H0 unit root test
    threshold_pvalue = 0.001
    # number of voxels that are non stationary
    counter_nonstat = 0
    counter_total = 0
    # initialize array of pvalues one for each voxel
    totdims = [dim_brains[0], dim_brains[1], dim_brains[2],1]
    pvalues = np.zeros(totdims)
    # Initialize dataframe that include all time series. Dont use this for entire brain, only for mask
    df = pd.DataFrame()
    for i in np.arange(0, dim_brains[0]-1):
        label = []
        for j in np.arange(0, dim_brains[1]-1):
            for k in np.arange(0, dim_brains[2]-1):
                counter_total = counter_total + 1
                label = [i,j,k]
                voxels_all.append(label)
                label_str = str(label[0]) +  str(label[1]) +  str(label[2]) 
                # df[label[0],label[1],label[2]] = pd.Series(image_data[label[0],label[1],label[2]])
                timeseries = pd.Series(image_data[label[0],label[1],label[2]], name='v%s' % counter_total)
                # print timeseries.name
                # concatenate Series with the df to create the dataframe with all the time series for each voxel 200 x totdims
                df[i,j,k]= timeseries
                #df = pd.concat([df, timeseries], axis = 1 )
                # to access time series by name of column 
                #print df[df.columns[-1]]               
                if np.mean(timeseries) > 10:
                    print " testing stationarity for time series in index:",i,j,k, "mean:", np.mean(timeseries)
                    res_col = run_collinearity_on_ts(timeseries) 
                    res_acf = run_autoregression_on_ts(timeseries) 
                    res_stat = run_test_stationarity_adf_ts(timeseries)
                    # print "H0: unit root non stationary, p value :", dfoutput[1]
                    if res_stat[1] > threshold_pvalue:
                        counter_nonstat =  counter_nonstat + 1
                        pvalues[i,j,k] =  res_stat[1]
                        print "We reject H0: unit root or non stationary, p value at x,y,z", dfoutput[1], label
    print "Non stationary voxels:", counter_nonstat, "/", totdims
    return pvalues    

def generate_surrogate_data(time_series, n, nullmodel=None):
    """Generate surrogate data time series from the given time series sharing 
    some properties of the observed time series, for example, mean , variance or power spectrum 
    but are otherwise random as specified by the null hypothesis
    Ref: http://www.sciencedirect.com/science/article/pii/016727899290102S
    The surrogate data consists on 'n' realizations according to the 'nullmodel'
    Return a list (number of time serties) x (number of realizations) of surrogate data
    Example: df = generate_surrogate_data(time_series, 100) # generates 100 samples for Gaussian noise
    """
    df = pd.DataFrame()
    same_spectra_as_raw = False
    if nullmodel is None:
        nullmodel='gaussian noise'
    if nullmodel == "gaussian noise":
        mu, sigma = 0, 1 # mean and std
#    elif nullmodel == "arima":
#        # generate ARMA model estimate the parameters with the original time series
#        for i in np.arange(0, n):
#            if same_spectra_as_raw:
#                df[i] = generate_surrogate_data_same_spectrum(timeseries,n=100)
#            else:
#                mu, sigma = 0, 1 # mean and std
#                noise = np.random.normal(mu, sigma, timeseries.size)
#                residuals = run_arima_fit_ts(timeseries.values, order=None) 
#                #pdb.set_trace()
#                residuals = residuals.fittedvalues
#                residuals = pd.Series(residuals) + noise[1:]
#                df[i] = residuals
                  
    print "Generating surrogate data for n=", n, " samples and for Hypothesis=",  nullmodel
    for ts_ix in range(0, time_series.shape[1]):
        time_series[:,ts_ix] = pd.Series(time_series[:,ts_ix])
        #df_surr = [] # List of surrogate data (number of time serties timseries) x (realizartions)
        #generate surogate data randomizing the phase ifft(randomized_phase(fft())
        for i in np.arange(0, n):
            noise = np.random.normal(mu, sigma, time_series.shape[0])
            #signalplusnoise = pd.Series(time_series[:,ts_ix] + noise)  
            signalplusnoise = pd.Series(noise) 
            df = pd.concat([df, signalplusnoise], axis = 1)                             
    print "Surrogate data created with dimensions", df.shape
    return df

def generate_surrogate_data_same_spectrum(ts, n):
    # This fucntion is now OK, the inverse fourier needs to be onto a symmetric
    Xf = np.fft.rfft(ts.values)
    Xf_mag = np.abs(Xf)
    Xf_phase= np.angle(Xf)
    Xf_phase_new = Xf_phase + np.random.uniform(-2*np.pi,2*np.pi, len(Xf_phase))
    Xf_new  = Xf_mag*np.exp(1j*Xf_phase_new)
    X_new = np.fft.ifft(Xf_new)
    # the inverse Fourier must be real, it it is complex is because we have not symmetrize the phases!!
    return pd.Series(np.abs(X_new)) 
    
def correlation_of_ts_vs_surrogate(original_ts, dataframe):
    """It returns the mean of the correlation between two ts (pearson, spearman, kendall) 
    for each ts with the n surrogate time series, the std and all the correlations 
    Example: [corr_mean_perts, corr_std_perts, list_all_corr] = hypothesis_testing(original_ts, dataframe)
    """
    nb_of_points = original_ts.shape[0]
    nb_of_ts = original_ts.shape[1]
    realizations = dataframe.shape[1]/nb_of_ts
    
    correlation_array = []
    correlation_array_std = []
    y = [] #array 2D with all pairwise correlations
    x = []
    for i in range(0, nb_of_ts):
        ts_orig = pd.Series(original_ts[:,i])
        y=[]
        for j in range(0, realizations): 
            jindex = realizations*i + j
            ts_tocompare = dataframe.iloc[:,jindex]
            # shuffle the time series
            ts_tocompare= ts_tocompare.sample(frac=1)
            #pairwise correlation(default is pearson) betwee oroginal ts and one of the relizations
            #method='pearson|kendall|spearman', min_periods=None
            corr = ts_orig.corr(ts_tocompare, method='spearman')
            print "correlation for i:", i, " jindex:", jindex, " =",corr
            x.append(corr)
        y.append(x)
        print "The correlation mean and std for timeseries:", i, " is ", np.mean(y), np.std(y)    
        correlation_array_std.append(np.std(y))
        correlation_array.append(np.mean(y))                     
    return correlation_array, correlation_array_std, y 

def plot_correlation_histogram(correlation_array, correlation_array_std, all_correlations, lags):
    # Plot the bar and the histogram for the original time series correlation with the surrogate data for each lag
    # Plot the bar of the mean correlation between the original ts and the n realizations for each lag
    x_pos = len(correlation_array)
    listlags = ('1','2','3','4','5','6','7','8','9','10')
    fig = plt.figure()
    plt.bar(np.arange(x_pos), correlation_array[:], align='center', alpha=0.5, yerr=correlation_array_std[:])
    plt.xticks(np.arange(x_pos), listlags)
    plt.ylabel('Correlation ts~ts shifted')
    plt.xlabel('Time lags')
    plt.title('H0 time series is iid Gaussian noise')
    # Plot distribution of the correlation 
    plot_distrib= False
    if len(all_correlations) and plot_distrib > 0:
        plt.figure()
        # convert list into array
        corr_array = all_correlations[0:]
        #Plot the hisogram for lags
        corr_array[0]
        bins = np.linspace(0, 1, 1000)
        plt.hist(corr_array[1], bins, alpha=0.5)
        plt.hist(corr_array[2], bins, alpha=0.5)
        plt.hist(corr_array[3], bins, alpha=0.5)
        # add a 'best fit' line
        plt.show()
        
def compute_nonlinear_statistics(timeseries, nb_of_timepoints=None):
    """Calculate a battery of significant statistics (non linear measures) for a time structure. 
    The argument "timeseries" can be a Pandas dataframe, Pandas time series or a ndarray.
    The measures that are calculated are: correlation dimension, sample entropy,
    Lyapunov exponents. It calls to compute_nonlinear_statistics_ts for eac time series
    We use the numpy-based library NonLinear measures 
    for Dynamical Systems (nolds) 
    https://pypi.python.org/pypi/nolds/0.2.0
    """
    if nb_of_timepoints is None: nb_of_timepoints = 116
    if isinstance(timeseries, pd.core.frame.DataFrame):
        #rename time series tructure
        dataframe = timeseries
        print "Time series is a data frame, calling to calculate_nonlinearmeasures for each time series"
        for i in range(0, dataframe.shape[1]):
            x = []
            ts_toanalyze = dataframe.iloc[:,i].values
            ts_measures = compute_nonlinear_statistics_nda(ts_toanalyze)
            print "The non linear statistics for the ith:", i, " surrogate data is: ", ts_measures
            x.append(ts_measures)
        return x    
    elif isinstance(timeseries, pd.core.frame.Series):
        print "Calling to calculate_nonlinearmeasures for the Pandas time series..."
        timeseries = timeseries.values
    elif isinstance(timeseries,np.ndarray):
        print "Calling to calculate_nonlinearmeasures for the time series ndarray..."
    return compute_nonlinear_statistics_nda(timeseries)
       

def compute_nonlinear_statistics_nda(timeseries):
    """Calculate a battery of significant statistics (non linear measures) of 
    a given Pandas time series.
    """
    #print "\t Calculating the correlation dimension (degrees of freedom) using Grassberger-Procaccia_algorithm ..."
    # http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm
    #emb_dim=1 because we are dealing with one dimensional time series
    emb_dim = 1 
    #Call corr_dimension with rvals by default (iterable of float): rvals=logarithmic_r(0.1 * std, 0.5 * std, 1.03))
    corr_dim = nolds.corr_dim(timeseries, emb_dim)
    #print "\t Calculating the Lyapunov exponents(initial conditions sensibility)) of the time series..."
    lyap = nolds.lyap_r(timeseries) # lyap_r for largest exponent, lyap_e for the whole spectrum
    #print "\t Calculating the sample entropy(complexity of time-series) based on approximate entropy..."
    ap_entropy = nolds.sampen(timeseries)
    nonlmeasures = {'corr_dim':corr_dim,'sampen':ap_entropy,'lyap':lyap}
    return nonlmeasures   

def stat_hyp_test_orig_vs_surr(orig_ts, surrogate_df):
    """"Significance test of the original time series versus the surrogate data
     frame genererated by the null model using p-values with rank statistics. For example, 
     if the observed time series in in the lower one percentile of all surrogate statistics 
     (for at least n=100 surrogates generated) then a two-sided p-value p=0.02 could be quoted. 
    """
    print "Computing a t-test ..."
    tandpvalues = ttest_orig_vs_surrogate(orig_ts, surrogate_df)
    return tandpvalues 

def ttest_orig_vs_surrogate(orig_stats, surrogate_stats, realizations=None):
    """ t-test to test whether the mean of the original time series \
    sample differs in a statistically significant way from the surrogate data set.
    The arguments are type list
    """
    print "\n"
    labels = ['corr_dim','lyap','sampen']
    nb_ts = len(orig_stats)
    if realizations is None: realizations = len(surrogate_stats[0])
    if realizations != len(surrogate_stats[0]): 
        print "ERROR in the numbr of realization in the surtrogate data"
        return -1
    tstats = []
    pvalues = []  
    
    for i in range(0,nb_ts):
        orig_stats_ith =  orig_stats[i]
        surrogate_stats_jth = []
        t_ith=[]
        prob_ith = []
        for labsix in range(0,len(labels)):
            surrogate_stats_jth = surrogate_stats[i][:]
            metric_surr = []
            for j in range(0,len(surrogate_stats_jth)):
                metric_surr.append(surrogate_stats_jth[j][labels[labsix]])
            [t, p] = stats.ttest_1samp(metric_surr, orig_stats_ith[labels[labsix]])
            print "ts: ", i, "ttest for :" , labels[labsix], "  t:", t, " p:", p
            t_ith.append(t)
            prob_ith.append(p)
        tstats.append(t_ith)
        pvalues.append(prob_ith)         
    return [tstats, pvalues]          
                         

def plot_significance_test(ts_sig_test, df_sig_test):
    # Plot scatter plot with statistics for the surrogate data versus the original dat set
    print "Plotting Lyapunov Exponent and Sample Entropy in a scatter plot"
    #max_emb_dim = 4
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ts_y = ts_sig_test['lyap']
    ts_x = ts_sig_test['sampen']
    ts_z = ts_sig_test['corr_dim']
    df_x, df_y, df_corr = [], [], []
    ax.scatter(ts_x, ts_y, ts_z, c='red', alpha=0.5)
    for i in np.arange(len(df_sig_test)):
        df_y.append(df_sig_test[i]['lyap'])
        df_x.append(df_sig_test[i]['sampen'])   
        #df_corr.append(df_sig_test[i]['corr_dim'][0:max_emb_dim])
        df_corr.append(df_sig_test[i]['corr_dim'])       
    ax.scatter(df_x, df_y, df_corr, c='blue', alpha=0.5)
    ax.set_xlabel('Sample entropy')
    ax.set_ylabel('Lyapunov exponent')
    ax.set_zlabel('Correlation dimension')
    #ax.savefig('samendLyapunovcorr_dim.jpg')
    
    # Plot correlation dimension
#    fig2 = plt.figure()
#    
#    dimensions = np.arange(1,max_emb_dim+1)
#    ts_sig_test['corr_dim'][0:max_emb_dim]
#    plt.scatter(dimensions, ts_sig_test['corr_dim'][0:max_emb_dim], c='red', alpha=0.5)    
#    #traspose the list of correlation values to have them index by the dimension
#    df_corr= map(list, zip(*df_corr))
#    for xe, ye in zip(dimensions, df_corr[:]):
#        #pdb.set_trace()
#        plt.scatter([xe] * len(ye), ye)
#        
#    plt.xlabel('dimension')
#    plt.ylabel('Correlation dimension')
#    fig2.suptitle('correlation dimension test', fontsize=12)
#    plt.xticks(range(1,max_emb_dim+1))
#    fig2.savefig('corr_dimension.jpg')

def ts_similarity(df1, df2):    
    #Calculate dynamic time warping between time series
    # return a matrix nxn fn is the number of time series Mij is the DTW distamce between two time series
    DTW_list,KStest = [], []
    cols1, cols2 = 1, 1
    if isinstance(df1, pd.core.frame.Series) and isinstance(df2, pd.core.frame.Series):
        distance, path = ts_DTW(df1, df2)
        DTW_list = distance
        print "DTW of the two time series is", distance
        # The Kolmogorov–Smirnov statistic quantifies a distance between 
        KStest = ts_KS_2samples(df1, df2)[1]
        
    elif isinstance(df1, pd.core.frame.Series):
        if isinstance(df2, pd.core.frame.DataFrame):
            print "Calculating DTW of ts1 versus dataframe2"
            rows2, cols2 = df2.shape            
            for i in np.arange(cols2):
                y = df2[i]
                distance, path = ts_DTW(df1, y)
                DTW_list.append(distance)
                #pdb.set_trace()
                KStest.append((ts_KS_2samples(df1, y)[1]))
                
    elif isinstance(df2, pd.core.frame.Series):        
        if isinstance(df1, pd.core.frame.DataFrame):
            print "Calculating DTW of ts2 versus dataframe1"
            rows1, cols1 = df1.shape
            for i in np.arange(cols1):
                x = df1[i]
                distance, path = ts_DTW(x, df2)
                KStest = ts_KS_2samples(x, df2)
                DTW_list.append(distance)
                KStest.append(KStest)
    elif isinstance(df1, pd.core.frame.DataFrame) and isinstance(df2, pd.core.frame.DataFrame):
       print "Calculating DTW of two dataframes"
       rows1, cols1 = df1.shape
       rows2, cols2 = df2.shape
       for i in np.arange(cols1):
           for j in np.arange(cols2):
               x = df1[i]
               y = df2[j]
               distance, path = ts_DTW(x, y)
               #ts_KS_2samples(x, y)
               DTW_list.append(distance)
               KStest.append(ts_KS_2samples(x, y)[1])
    print "The Total DTW is:", DTW_list , "KS p-value test is:",  KStest[1]         
    
    plot_Heatmap = True
    if plot_Heatmap:
        plt.figure()
        DTW_array = np.asarray(DTW_list)
        DTW_array = DTW_array.reshape(cols1, cols2)
        ax = sns.heatmap(DTW_array, xticklabels=5, yticklabels=False)
        ax.set(xlabel='null model voxels', ylabel='observed ts')
        ax.set_title('Dynamic Time Warping distance')
        KS_array = np.asarray(KStest)
        #pdb.set_trace()
        #KS_array = KS_array.reshape(cols1, cols2)
        plt.figure()
        ax = sns.heatmap([KS_array], xticklabels=5, yticklabels=False)
        ax.set(xlabel='null model voxels', ylabel='observed ts', title = 'KS 2 samples')
        #ax.set_title('KS 2 sample test')
        
    return DTW_list, KStest   
         

            
def ts_DTW(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance, path 

def ts_KS_2samples(ts1, ts2):
    # Calcualtes the Kolmogorov Smirnov statistic for 2 samples, or time series
    # It is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
    KS, pvalue = stats.ks_2samp(ts1, ts2)
    KSres = [KS, pvalue]
    return KSres
        
def fourier_spectral_estimation(ts, image_params=None):
    """Calculate the PSD estimate from a time series using Welch
    The power spectrum calculates the area under the signal plot using the discrete Fourier Transform
    The PSD assigns units of power to each unit of frequency and thus, enhances periodicities 
    Welch’s method computes an estimate of the power spectral density by dividing 
    the data into overlapping segments, computing a modified periodogram for each segment and averaging the periodograms.
    by default constant detrend, one-sided spectrum, 
    noverlap = nperseg / 2 (noverlap is 0, this method is equivalent to Bartlett’s method). 
    scaling 'density'V**2/Hz 'spectrum' V**2.
    returns array of sampling frerquencies and the power spectral density of the ts
    
    Example: f, Pxx_den = fourier_spectral_estimation(ts)
    
    """
     
    #data_img, image_params = load_fmri_image_name()
    #data = np.zeros(1, 1)  # initialize for number of rois and number of samples
    # The cadillac of spectral estimates is the multi-taper estimation, 
    # which provides both robustness and granularity, but notice that 
    # this estimate requires more computation than other estimates  
    psd_results = []
    plotPxx = True
    if plotPxx is True: 
        fig, ax = plt.subplots(ts.shape[1], sharex=True, sharey=True)
    if image_params is None: image_params = load_fmri_image_name()[1]
    for i in range(0, ts.shape[1]):
        #nperseg is the length of each segment, 256 by default
        nperseg=20
        f, Pxx_den = signal.welch(ts[:,i], image_params['fs'], nperseg=nperseg, detrend='constant', nfft =image_params['nfft'], scaling = 'density')  
        pxx = [f, Pxx_den]
        psd_results.append(pxx)
        print "Timeseries:", i," frequency sampling", f, " Pxx_den:", Pxx_den, " Max Amplitude is:", np.mean(Pxx_den.max())
        if plotPxx is True:
            #plt.figure()
            #ax[i].semilogy(f, Pxx_den)
            ax[i].plot(f, Pxx_den)
            ax[i].set_xlabel('frequency [Hz]')
            ax[i].set_ylabel('PSD [V**2/Hz]')
            ax[i].set_title('PSD')
    print psd_results        
    return  psd_results   

def get_MNI_coordinates(label):
    #get_MNI_coordinates return the dictionary MNI coordinates of the brain structure or network label
    dim_coords = []
    if label is 'DMN':
        # http://sprout022.sprout.yale.edu/mni2tal/mni2tal.html
        # DMN coordinates from HEDDEN ET AL (2009) PCC
        # DMN = PCC (-5, -53, 41) is BA31 http://www.sciencedirect.com/science/article/pii/S187892931400053X
        # ,MPFC (0, 52, -6)  LLPC (-48, -62, 36) RLPC (46, -62, 32)
        label = ['Post. Cing. Cortex','Left Tmp.Ptl. junction','Right Tmp.Ptl. junction','Medial PFC '] 
        # Dictionary of network with MNI components. 
        # http://nilearn.github.io/auto_examples/03_connectivity/plot_adhd_spheres.html#sphx-glr-auto-examples-03-connectivity-plot-adhd-spheres-py
        # dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)] 
        #dim_coords = {label[0]:(0, -52, 18),label[1]:(-46, -68, 32), label[2]:(46, -68, 32),label[3]:(1, 50, -5)} #from nilearn page
        dim_coords = OrderedDict([(label[0],(0, -52, 18)),(label[1],(-46, -68, 32)), (label[2],(46, -68, 32)),(label[3],(1, 50, -5))])
        #make sure dictionary respects the order of the keys
        #pdb.set_trace()
    elif label is 'SN':
        # Salience network
        dim_coords = []    
    else: 
        print " ERROR: label:", label, " do not found, returning empty list of coordinates!"   
    return dim_coords   