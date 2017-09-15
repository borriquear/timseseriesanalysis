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
import scipy as sp
from scipy import stats, signal, linalg

from datetime import datetime, date, time
import statsmodels.tsa.stattools as tsa
import nibabel as nib
from nilearn import input_data
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker 
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nipy.labs.viz import plot_map, mni_sform, coord_transform
from nilearn.connectome import GroupSparseCovarianceCV
from sklearn.covariance import GraphLassoCV, ledoit_wolf
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
import networkx as nx

#Global variables
#standarize transform time series in unit variance
# if True the time-series are centered and normed: 
# their mean is set to 0 and their variance to 1 in the time dimension.
#low_pass, high_pass, detrend, smoothing_fwhm,t_r,verbose
standarize = True
detrend = True
#smoothing_fwhm = None  If smoothing_fwhm is not None, it gives the full-width 
#half maximum in millimeters of the spatial smoothing to apply to the signal.
smoothing_fwhm = 8
t_r = 2.5
low_pass = 0.2
high_pass = 0.001
verbose = 2

plt.close('all')
#plt.clf()


def test_group_analysis(epi_file_list=None):
    """group analysis of bold data. Calculates the components of a list of bold images
        
    """
    from nilearn.decomposition import CanICA
    from nilearn.plotting import plot_prob_atlas
    from nilearn.image import iter_img
    from nilearn.plotting import plot_stat_map, show

    if epi_file_list is None:
        epi_file_list = []
        subjects_list = ['bcpa0537_0','bcpa0517', 'bcpa0530', 'bcpa0540','bcpa0543', 'bcpa0545','bcpa0576','bcpa0577','bcpa0578_0','bcpa0581','bcpa0650_0']
        subjects_list = ['bcpa0537_1','bcpa0578_1', 'bcpa0650_1']
        dir_name = '/Users/jaime/vallecas/data/surrogate/'
        f_name = 'wbold_data.nii'
        for i in range(0,len(subjects_list)):
            subjname = os.path.join(dir_name,subjects_list[i])
            epi_file_list.append(os.path.join(subjname, f_name))            
    canica = CanICA(n_components=20, smoothing_fwhm=smoothing_fwhm, #6.
                memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)    
    canica.fit(epi_file_list)
    # Retrieve the independent components in brain space
    components_img = canica.masker_.inverse_transform(canica.components_)
    # save components as a nifti image 
    components_img.to_filename('canica_resting_state.nii.gz')   
    # Plot all ICA components together
    plot_prob_atlas(components_img, title='All ICA components 3/3 Alive',cut_coords=(0,-52,18))
    # plot the map for each ICA component separately
    for i, cur_img in enumerate(iter_img(components_img)):
        plot_stat_map(cur_img, display_mode="z", title="IC %d" % i, cut_coords=1, colorbar=False)
    show()
    
    
def test_clustering_in_rs(epi_file=None):
    """Hierarchical clustering (Ward) in rs data
    """
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.feature_extraction import image
    #import time
    from nilearn.plotting import plot_roi, plot_epi, show
    from nilearn.image import mean_img

    
    if epi_file is None:
        [epi_file, epi_params]= load_fmri_image_name()
        print " EPI file:", epi_file, " EPI params:", epi_params
    
    nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                                      mask_strategy='epi', memory_level=1,
                                      standardize=False)  
    #compute the mask and extracts the time series form the file(s)
    fmri_masked = nifti_masker.fit_transform(epi_file[0])
    #retrieve the nup array of the mask
    mask = nifti_masker.mask_img_.get_data().astype(bool)
    #compute the connectivity matrix for the mask
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)
    
    #pdb.set_trace()
    tic = datetime.now()
    #FeatureAgglomeration clustering algorithm from scikit-learn 
    n_clusters= 40
    ward = FeatureAgglomeration(n_clusters, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
    ward.fit(fmri_masked)
    print("Ward agglomeration %d clusters: " % n_clusters, "in time=", str(datetime.now()- tic))
    #visualioze the results
    labels = ward.labels_ + 1
    labels_img = nifti_masker.inverse_transform(labels)
    mean_func_img = mean_img(epi_file)
    #msgtitle ="Ward parcellation nclusters=%s, %s" % (n_clusters, os.path.split(os.path.dirname(epi_file))[1])
    msgtitle = 'Ward parcellation Converters' + 'nb clusters='+ str(n_clusters)
    first_plot = plot_roi(labels_img, mean_func_img, title=msgtitle,
                      display_mode='ortho', cut_coords=(0,-52,18))
    cut_coords = first_plot.cut_coords
    print cut_coords
    labels_img.to_filename('parcellation.nii')

    
def test_connectome_timeseries(epi_file=None, type_of_mask=None):
    """ connectome time series characterization
    Run tests for collinearity, run_autoregression_on_df, stationarity and calculates the connectome of the given coords
    Example: test_connectome_timeseries() 
    epi_file is a list of file paths epi_file = ['file1, 'file2' ...]
    test_connectome_timeseries(None, type_of_mask='spheres') 
    test_connectome_timeseries(None, type_of_mask='atlas')
    test_connectome_timeseries(None, type_of_mask='brain-wide') 
    """
    preprocessing = False
    if epi_file is None:
        [epi_file, epi_params]= load_fmri_image_name()
    else:
        [epi_file, epi_params]= load_fmri_image_name(epi_file, preprocessing=preprocessing)    
    print " EPI file:", epi_file, " EPI params:", epi_params    
    if type_of_mask is None:
        #if not mask specified used harvard cortical atlas
        type_of_mask = ['brain-wide', 'atlas', 'spheres'][2]
    # for studies with a directory for each patient

    if type(epi_file) is list:
        subject_id = []
        for i in range(0, len(epi_file)):
            # for all subjects in one folder
            filename = os.path.split(os.path.basename(epi_file[i]))[1][:-9]
            # converter or control
            typeofsub = os.path.basename(os.path.dirname(epi_file[i]))
            fileandtype = typeofsub + ':' + filename
            subject_id.append(fileandtype)  
    else:
        subject_id = os.path.split(os.path.dirname(epi_file[0]))[1]
        
    label_coords = None
    epi_data = None
    dim_coords = None
    if type_of_mask == 'spheres':
        rs_network_name = 'DMN'
        dim_coords = get_MNI_coordinates(rs_network_name)  
        label_coords = dim_coords.keys()
    elif type_of_mask == 'atlas':
        rs_network_name = type_of_mask
        print("Using an Atlas as mask")
    elif type_of_mask == 'brain-wide':
        epi_data = epi_file[0]       
    #generate the mask, only if brain wide we need epi_data, if spheres dim_coords
    masker = generate_mask(type_of_mask, epi_data, dim_coords)
    # the same Mask for all subjects
    print("Masker parameters for subject:%s are:%s\n") % (subject_id[0], masker.get_params())
    print("Extracting the time series for subject:%s and the Mask") % subject_id[0] 

    time_series = []
    for i in range(0, len(epi_file)):
        ts = masker.fit_transform(epi_file[i])
        time_series.append(ts)         
        if time_series[i].shape[0] == epi_params['n']:
            #pdb.set_trace()
            warnings.warn("The time series number of points is 120, removing 4 initial dummy volumes", Warning)
            time_series[i] = time_series[i][4:120,:]
    print('Number of features:', len(time_series), 'Feature dimension:', time_series[0].shape)      
    #plot time series only if the mask is not the entire brain
    if type_of_mask != 'brain-wide':
        for i in range(0, len(epi_file)):
            print "Plotting time series for type_of_mask:", type_of_mask, " (txv)=", time_series[i].shape[0],  time_series[i].shape[1] 
            #plot_ts_network(time_series[i], label_coords, subject_id[i], rs_network_name)  
    
    #pdb.set_trace() 
    sparse_inverse_cov = True
    if sparse_inverse_cov is True:
        #print("Estimating the connectome from the Sparse Inverse Covariance for labels:", label_coords)
        #estimator, lw_cov_ = sparse_inverse_estimator(time_series, label_coords, dim_coords)
        # for n subjects
        print('Number of Subjects=' , len(time_series),' Estimating the connectome from the Sparse Inverse Covariance for mask:', label_coords)
        gsc = sparse_inverse_estimator_nsubjects(time_series,label_coords, dim_coords )
      
    print("END OF PROGRAM")
    return    
    
    non_lin_stats = False
    if non_lin_stats is True:    
        nonlinearresults = compute_nonlinear_statistics(time_series)
        print "Subject:", subject_id, ", Mean of nonlinear measures: corr_dim=", np.mean(nonlinearresults['corr_dim']),\
        " Sample Entropy", np.mean(nonlinearresults['sampen'])," Lyapunov exponent=", np.mean(nonlinearresults['lyap']),\
        " Hurst exponent", np.mean(nonlinearresults['hurst'])                                                                            
        #Save non linear measures in file"
        save_dict_in_file(nonlinearresults,type_of_mask)
    ts_nonlin_characterization = False
    if ts_nonlin_characterization is True:
        print "Testing for collinearity in the time series"
        res_lagged = run_collinearity_on_df(time_series, subject_id)
        print "Calling to run_autoregression_on_ts(y) to test for autocorrelation in the time series" 
        res_acf = run_autoregression_on_df(time_series, subject_id)      
        print "Calling to run_test_stationarity_adf(y) to test for stationarity using augmented Dickey Fuller test" 
        res_stationarity = run_test_stationarity_adf(time_series, subject_id)                                
        print "Calling to ARIMA model fit"
        order = None
        res_arima = run_arima_fit(time_series, order, subject_id)
       #print "Calling to run_forecasting to study predictability of the time series for the ARIMA"
       #res_forecasting = run_forecasting(res_arima)
       #return masker, time_series, res_lagged, res_acf, res_stationarity, res_arima 
    hetero_test = False
    if hetero_test is True:
        results_heterotest= []
        print "Running test for Heteroscedasticity:  Null assumption = residuals from a regression is homogeneous (constant volatility)"
        print "Performing autoregression in each time series, number of time series", time_series.shape[1]
        print("Running Test of heteroskedasticity. H0: Residuals of autoregression are Homogeneous")
        residuals_res =run_autoregression_on_df(time_series)
        for i in np.arange(0, time_series.shape[1]):
            #pdb.set_trace()
            reg_predicted = residuals_res[i].predict()
            reg_residuals = residuals_res[i].resid
            het_res = test_heteroscedasticity(reg_predicted, time_series[:,i])
            results_heterotest.append(het_res)
            print('Breusch Pagan test value, df, p-value')
            print subject_id , ' time series:', i, ' bp = ', het_res
            #het_res_white = test_heteroscedasticity(reg_residuals, time_series[:,i], 'white')
        #plot results of heteroscedasticity test
        rejecth0het = 0
        for ix in np.arange(0,len(results_heterotest)):
            if results_heterotest[ix][2] < 0.05:
                rejecth0het = rejecth0het + 1
                print 'Reject null hypothesis (homoscedasticity) for ts=', ix, ' p_value = ', results_heterotest[ix][2], \
                                         'bp = ', results_heterotest[ix][0]
        print subject_id, ' Number of time series with heteroscedasticity = ', rejecth0het, ' /', ix
    #plt.close('all')
    #plt.clf()
       
    display_connectome = True
    if display_connectome is True:
        kind_of_correlation = 'correlation'
        print "Displaying the connectome for corr_type:", kind_of_correlation
        if type_of_mask == 'atlas':
            label='cort-maxprob-thr25-2mm'
            label_map = get_atlas_labels(label)
        elif type_of_mask == 'spheres':
            label = rs_network_name
            label_map = dim_coords.keys()
        else:
            label = 'brain-wide'
        msgtitle =" Correlation:%s for %s, Type of Mask:%s" % ( kind_of_correlation, os.path.split(os.path.dirname(epi_file))[1], label)
        # Plotting time series and network in brain overlayed using  nilearn.connectome.ConnectivityMeasure
        corr_matrix = plot_ts_connectome(time_series, kind_of_correlation,label, subject_id)
        if type_of_mask == 'atlas':  
            #plotting kind_of_correlation as a heatmap from nitime
            fig_C_drawx = drawmatrix_channels(corr_matrix, label_map, size=[10., 10.], color_anchor=0, title= msgtitle)  
        elif type_of_mask == 'spheres': 
            #plotting kind_of_correlation as a network with nodes of variable size and label in it from nitime
            fig_C_drawg = drawgraph_channels(corr_matrix,label_map,title=msgtitle) 
    
    dynamictimewarping = False
    if dynamictimewarping is True:
        print "Computing the Dynamic time warping for ts similarity"
        onetoallcomparison =False
        if onetoallcomparison == True:
            ts1 = pd.Series(time_series[:,0])
            ts2 = pd.DataFrame(time_series[:,1:time_series.shape[1]])
        else:
            #do all to all rois calculus
            ts1 = pd.DataFrame(time_series)
            ts2 = ts1
        ts_similarity(ts1,ts2, subject_id)
  
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
        msgtitle ="Mean Coherence across frequencies %s" % (os.path.split(os.path.dirname(epi_file))[1])
        #T = ts.TimeSeries(np.vstack([ts_seed, ts_target]), sampling_interval=image_params['TR'])
        T = ts.TimeSeries(np.transpose(time_series), sampling_interval=epi_params['TR'])
        method=dict(this_method='welch', n_overlap= 20)
        Coh_az= CoherenceAnalyzer(T, method)
        print "Coherence Analyzer method used:", Coh_az
        #pdb.set_trace()
        freq_idx = np.where((Coh_az.frequencies > high_pass) * (Coh_az.frequencies < low_pass))[0]
        coh = np.mean(Coh_az.coherence[:, :, freq_idx], -1)  # Averaging on the last dimension
        #pdb.set_trace()
        fig04 = drawgraph_channels(coh, label_coords, title=msgtitle) 
        fig03 = drawmatrix_channels(coh, label_coords, size=[10., 10.], color_anchor=0)
        
                     
#        [coherence_mat, seeds, targets] = compute_coherence_pairs_of_ts(time_series)
#        print "Plotting functional connectivity of the Frequency spectrum"
#        fig_F_drawx = drawmatrix_channels(coherence_mat, dim_coords.keys(), size=[10., 10.], color_anchor=0, title= msgtitle)  
#        fig_F_drawg = drawgraph_channels(coherence_mat, dim_coords.keys(),title=msgtitle) 
        #plot using ploty
        #plot_coupling_heatmap(coherence_mat, seeds, targets, dim_coords.keys(), corr_type=msgtitle)
        #msgtitle = 'Coherency DMN (Cad)'
        #SeedCoherency
        #compute_seed_coherence(time_series)
                       
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
    
    seed_based_connectome = True
    if seed_based_connectome is True:
        seed_masker = masker
        if time_series is None:
            time_series = seed_masker.fit_transform(epi_file)
        seed_time_series = time_series    
        brain_masker = generate_mask('brain-wide', epi_file)
        brain_time_series = brain_masker.fit_transform(epi_file)
        print("seed time series shape: (%s, %s)" % seed_time_series.shape)
        print("brain time series shape: (%s, %s)" % brain_time_series.shape)
        #make sure time series have same number of time points
        if seed_time_series.shape[0] != brain_time_series.shape:
            brain_time_series = brain_time_series[4:brain_time_series.shape[0],:]
            print("The corrected time series dimension are:")
            print("seed time series shape: (%s, %s)" % seed_time_series.shape)
            print("brain time series shape: (%s, %s)" % brain_time_series.shape)
        #select the seed
        seed_time_series = seed_time_series[:,0]
        seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / \
                          seed_time_series.shape[0]
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
        #MNI152Template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_symmetric.nii.gz'
        #remove out-of brain functional connectivity using a mask
        icbms = datasets.fetch_icbm152_2009()
        masker_mni = NiftiMasker(mask_img=icbms.mask)
        data = masker_mni.fit_transform('sbc_z.nii.gz')
        masked_sbc_z_img = masker_mni.inverse_transform(data)
        #pdb_set_tarce()
        display = plotting.plot_stat_map(masked_sbc_z_img , cut_coords=pcc_coords, \
                                         threshold=0.6, title= 'PCC-based corr. V-A', dim='auto', display_mode='ortho')
        
        #Coherency

        #display.add_markers(marker_coords=pcc_coords, marker_color='g', marker_size=300)
        # At last, we save the plot as pdf.
        #display.savefig('sbc_z.pdf')
        #correlate the seed signal with the signal of each voxel. \
        #The dot product of the two arrays will give us this correlation. 
        #Note that the signals have been variance-standardized during extraction. 
        #To have them standardized to norm unit, we further have to divide the 
        #result by the length of the time series.
        
def  load_fmri_image_name(image_file=None, preprocessing=False):
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
        dir_name = '/Users/jaime/vallecas/data/surrogate/bcpa0650_0'
        dir_name = '/Users/jaime/vallecas/data/surrogate/rf_off'
        dir_name = '/Users/jaime/vallecas/data/surrogate/bcpa0650_1'
        f_name = 'bold_data_mcf2standard.nii'
        f_name = 'wbold_data.nii'
        image_file = os.path.join(dir_name, f_name)
    #preprocessing = False
    
        
    if preprocessing == True:
        print("Preprocessing the bold_data (MCFLIRT and SliceTimer)")
        if type(image_file) is list:
            for i in range(0, len(image_file)):
                pre_processing_bold(image_file[i])
        else:
            #pdb.set_trace()
            pre_processing_bold(image_file)
    return image_file, image_params

def pre_processing_bold(image_file, dir_name=None):
    """Preprocessing bold_data using the FSL wrapper
    MCFLIRT and SliceTimer
    """
    from nipype.interfaces import fsl
    slicetimer = True
    if  slicetimer == True: 
        print "slicetimer interleave for bold:", image_file, " original bold will be overwritten"
        st = fsl.SliceTimer()
        st.inputs.in_file = image_file
        st.inputs.interleaved = True
        st.inputs.out_file = image_file
        st.inputs.output_type = 'NIFTI'
        result = st.run()
    motion_correction = True
    if motion_correction == True:
        print "Motion correction for bold:", image_file
        mcflt = fsl.MCFLIRT()
        mcflt.inputs.in_file = image_file
        mcflt.inputs.cost = 'mutualinfo'
        mcflt.inputs.out_file = image_file
        mcflt.inputs.output_type = 'NIFTI'
        res = mcflt.run()

def plot_ts_network(ts, labels=None, subject_id=None,rs_net_name=None):
    """plot time series ts 
    
    ts: time series object
    labels: list of labels
    """
    # distinguish between coordinates and atlas to do not have a lgend when an atlas
    figsize=(10, 8)
    fig = plt.figure(figsize=figsize) 
    if labels is None:
        msgtitle = 'Atlas Time Series for %s standarize is %s' % (subject_id, standarize)
        for ts_ix in np.arange(0,  ts.shape[1]):
            plt.plot(np.arange(0,ts.shape[0]), ts[:,ts_ix])       
    else:
        msgtitle = '%s %s Time Series' % (subject_id, rs_net_name)
        for ts_ix, label in zip(ts.T, labels):
            plt.plot(np.arange(0,len(ts_ix)), ts_ix, label=label)
        
    plt.title(msgtitle)
    plt.xlabel('Time points (TR=2.5s)')
    plt.ylabel('Signal Intensity')
    plt.legend()
    #plt.tight_layout()
    
def plot_ts_connectome(ts, kind_of_corr=None, labelnet=None, subject_id=None):
    """plot correlation netween time series using nilearn.plotting.plot_connectome
    
    ts: time series at least 2
    kind_of_corr:  {“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
    labelnet: label on top ledt of the overlayed brain figure. If none DMN   
    
    """
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    if kind_of_corr is None:
        kind_of_corr='correlation'
    connectivity_measure = ConnectivityMeasure(EmpiricalCovariance(assume_centered=True), kind=kind_of_corr) 
    correlation_matrix = connectivity_measure.fit_transform([ts])[0] 
    #if the ts belong to an atlas 
    msgtitle = `subject_id`+ " " + ` labelnet`
    if labelnet[0:4] == 'cort':
        #obtaining the atlas areas
        legend_values = get_atlas_labels(labelnet)
        #we need the coordinates of each region . dont know how to get those
        #http://nilearn.github.io/modules/reference.html#module-nilearn.datasets
    else:
        #ts are from spheres coordinates        
        coords_dict = get_MNI_coordinates(labelnet)
        legend_values = coords_dict.values()   
        plotting.plot_connectome(correlation_matrix, legend_values,edge_threshold='05%',
                         title=msgtitle,display_mode="ortho",edge_vmax=.5, edge_vmin=-.5)
    return correlation_matrix    

def get_atlas_labels(label):
    """return the labels of an atlas
    """
    if label[0:4] == 'cort':
            dataset = datasets.fetch_atlas_harvard_oxford(label)
            legend_values = dataset.labels[1:]
            return legend_values
    
def generate_mask(type_of_mask, epi_filename=None, sphere_coords=None):
    """generate a mask calling to NiftiSpheresMasker, NiftiLabelsMasker or NiftiMasker
    
    type_of_mask: 'atlas'
    type_of_masl: 'spheres' 
    type_of_mask: 'brain-wide'
    epi_filename: bold_ image necessary only for brain-wide to build the raw mask
    """
    import nilearn.image as image
    from nilearn.plotting import plot_roi, show
    #mask parameters
    #high_pass, low_pass,smoothing_fwhm,standardize,t_r,verbose,detrend
    masker = []
    if type_of_mask == 'atlas':
        #Load Harvard-Oxford parcellation from FSL if installed, If not, it \
        #downloads it and stores it in NILEARN_DATA directory
        #Name of atlas to load. Can be: cort-maxprob-thr0-1mm, cort-maxprob-thr0-2mm, 
        #cort-maxprob-thr25-1mm, cort-maxprob-thr25-2mm, cort-maxprob-thr50-1mm, 
        #cort-maxprob-thr50-2mm, sub-maxprob-thr0-1mm, sub-maxprob-thr0-2mm, 
        #sub-maxprob-thr25-1mm, sub-maxprob-thr25-2mm, sub-maxprob-thr50-1mm, 
        #sub-maxprob-thr50-2mm, cort-prob-1mm, cort-prob-2mm, sub-prob-1mm, sub-prob-2mm
        atlas_name ='cort-maxprob-thr25-2mm'
        # The mask is an Atlas
        dataset = datasets.fetch_atlas_harvard_oxford(atlas_name)
        atlas_filename = dataset.maps
        masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=standarize,detrend=detrend, \
                                   smoothing_fwhm=smoothing_fwhm,low_pass=low_pass,\
                                   high_pass=high_pass,t_r=t_r,verbose=verbose)
        print "Generated mask for Atlas:", atlas_name 
        plotting_atlas = True
        if plotting_atlas is True:
            plotting.plot_roi(atlas_filename, title=atlas_name)
    elif type_of_mask == 'spheres':
        #to generate a mask from coordinates load_spheres_mask([x,y,z],[i,j,k]]])
        masker = load_spheres_mask(sphere_coords)
        print "Generated mask for Spheres", masker 
    elif type_of_mask == 'brain-wide':
        #extract from  lready masked data or from raw EPI data
        extract_from_raw_epi_data = True
        if extract_from_raw_epi_data is True:
            print "Extracting mask for raw EPI data:", epi_filename
            try:
                mean_img = image.mean_img(epi_filename)
                # Simple mask extraction from EPI images
                # We need to specify an 'epi' mask_strategy, as this is raw EPI data
                masker = NiftiMasker(mask_strategy='epi',standardize=standarize,detrend=detrend,smoothing_fwhm=smoothing_fwhm,low_pass=low_pass,\
                                     high_pass=high_pass,t_r=t_r,verbose=verbose)
                masker.fit(epi_filename)
                plot_roi(masker.mask_img_, mean_img, title='EPI automatic mask')
                # Generate mask with strong opening
                #masker = NiftiMasker(mask_strategy='epi', mask_args=dict(opening=10))
                #masker.fit(epi_filename)
                #plot_roi(masker.mask_img_, mean_img, title='EPI Mask with strong opening')
            except:
                warnings.warn('ERROR loading EPI data for Raw mask, check that the EPI data exists and is passed as an argument',Warning)
        else:
            # extract from already existing image, MNI 2mm
            icbms = datasets.fetch_icbm152_2009()
            masker = NiftiMasker(mask_img=icbms.mask)
            print "Generated mask for brain-wide MNI 152 mask" 
    else:
        masker = []
        warnings.warn("ERROR calling generate_mask. Use generate_mask('atlas'|'spheres'|brain-wide', epi_filename=None), Warning")
    return masker    

     
def load_spheres_mask(dim_coords=None):   
    """Returns the Masker object for a mask type shperes
    
    dim_coords: dictionary of a network of interest eg DMN.
    dim_coords: list of coordinates of voxels.    

    Example: 
    load_masker(get_MNI_coordinates('MNI'))  #returns masker for the the MNI coordinates
    load_masker([(0, -52, 18),(10, -52, 18)]) #returns masker for a list of coordinates                                             
    """
    radius = 8 #in mm. Default is None (signal is extracted on a single voxel
    #smoothing_fwhm = None # If smoothing_fwhm is not None, it gives the full-width half maximum in millimeters of the spatial smoothing to apply to the signal.

    if dim_coords is None or len(dim_coords) == 0:    
        print "No mask used, process the entire brain"
        return None
    elif type(dim_coords) is dict or type(dim_coords) is OrderedDict:  
        # Extract the coordinates from the dictionary
        print " The mask is the list of voxels:", dim_coords.keys(), "in MNI space:", dim_coords.values()
        masker = input_data.NiftiSpheresMasker(dim_coords.values(), radius=radius,
                                               detrend=detrend, smoothing_fwhm=smoothing_fwhm,standardize=standarize,
                                               low_pass=low_pass, high_pass=high_pass, 
                                               t_r=t_r,memory='nilearn_cache', 
                                               memory_level=1, verbose=verbose, allow_overlap=False)
        print masker
    elif type(dim_coords) is list:  
        # Extract the coordinates from the dictionary
        print " The mask is the list of voxels:", dim_coords
        masker = input_data.NiftiSpheresMasker(dim_coords, radius=radius,
                                               detrend=detrend, smoothing_fwhm=smoothing_fwhm,standardize=standarize,
                                               low_pass=low_pass, high_pass=high_pass, 
                                               t_r=t_r,memory='nilearn_cache', 
                                               memory_level=1, verbose=verbose) 
    return masker
        
def compute_seed_coherence(time_series, image_file=None, image_params=None):
    """compute seed based coherency analysis for one seed and the brain
    
    time_series: time series of theseeds
    """
    import nitime.fmri.io as io
    #f_lb = 0
    #f_ub = 0.2
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
                                filter=dict(lb=high_pass,
                                            ub=low_pass,
                                            method='boxcar'))
    time_series_target = io.time_series_from_file(image_file,
                                          coords_target,
                                          TR=image_params['TR'],
                                          normalize='percent',
                                          filter=dict(lb=high_pass,
                                                      ub=low_pass,
                                                    method='boxcar'))
    #remove nan
    [nn, pp] = time_series_target.shape
    for i in range(0,nn):
        time_series_target.data[i] = nitime.utils.thresholded_arr(time_series_target.data[i], fill_val=0.)
    print "Calculating  SeedCoherenceAnalyzer"
    A = SeedCoherenceAnalyzer(time_series_seed, time_series_target)
    freq_idx = np.where((A.frequencies > high_pass) * (A.frequencies < low_pass))[0]
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
    freq_idx_C1 = np.where((C1.frequencies > high_pass) * (C1.frequencies < low_pass))[0]
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
    """Calculate the Granger causality for pairs of time series using nitime.analysis.coherence
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
         warnings.warn("ERROR: We cant calculate Granger for only 1 time series", Warning)
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
    
def run_collinearity_on_df(y, subject_id=None):
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
            res_lagged = run_collinearity_on_ts(pd.Series(y[:,index]),index,subject_id)
            summary_list.append(res_lagged)
    return summary_list           

def run_collinearity_on_ts(y, index=None, subject_id=None):
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
        plot_collinearity_on_ts(res_lagged, df,"%s timeseries:%d" % (subject_id, index))
    return res_lagged

def plot_collinearity_on_ts(res_lagged,df,titlemsg):
        plt.figure()
        plt.subplot(2,1,1)
        sns.heatmap(df.corr(),vmin=-1.0, vmax=1.0) 
        titlemsg2 = " Shifted " + titlemsg 
        plt.title(titlemsg2)      
        plt.subplot(2,1,2)
        #axarr[0,1] = plt.axes()
        #ax2.set_title('Correlation of the lagged coefficients: y ~ b(0) + ...+ b(5)')
        res_lagged.params.drop(['Intercept', 'trend']).plot.bar(rot=0)
        plt.ylim(-1,1)
        plt.ylabel('Coefficient')
        titlemsg = ' Correlation of the lagged coefficients: y ~ b(0) + ...+ b(5)' 
        plt.title(titlemsg)
        sns.despine()  
        #plt.show() 
        
def run_autoregression_on_df(y, subject_id=None):
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
            res_trend  = run_autoregression_on_ts(pd.Series(y[:,index]), index, subject_id)                                                                
            summary_list.append(res_trend)                                                                
            print res_trend.summary()
                                                                            
    return summary_list        
                                                                
def run_autoregression_on_ts(y,index=None,subject_id=None):  
    """Run autoregression on a timeseries (ndarray)
    """
    if index is None: index=0
    mod_trend = sm.OLS.from_formula('y ~ trend', data=y.to_frame(name='y').assign(trend=np.arange(len(y))))
    res_trend = mod_trend.fit()
    # Residuals (the observed minus the expected, or $\hat{e_t} = y_t - \hat{y_t}$) are supposed to be white noise. 
    # Plot the residuals time series, and some diagnostics about them
    plotacfpcf = True
    if plotacfpcf is True:    
        plot_autoregression_on_ts(res_trend.resid,msgtitle = "%s timeseries:%d" % (subject_id, index), lags=36)
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

def test_heteroscedasticity(y,x, typeoftest=None):
    """test for heteroskedasticity
    
    x,y: ndarray for time series to run a regression
    typeoftest: None, Breusch, and Pagan , other are White in %http://www.statsmodels.org/stable/diagnostic.html?highlight=heteroscedasticity

    """
    import statsmodels.stats.diagnostic
    if typeoftest is None or typeoftest is 'bp':
        
        results = sm.OLS(y,x).fit()
        resid = results.resid
        sigma2 = sum(resid**2)/len(resid)
        f = resid**2/sigma2 - 1
        results2 = sm.OLS(f,x).fit()
        fv = results2.fittedvalues
        bp = 0.5 * sum(fv**2)            
        df = results2.df_model
        p_value = 1-sp.stats.chi.cdf(bp,df)
        return round(bp,6), df, round(p_value,7)
    elif typeoftest is 'white':
        print("TO DO Running Test of heteroskedasticity, Breusch, and Pagan (1979)")
        #statsmodels.stats.diagnostic.het_white(y, x)
        
    
        
def run_test_stationarity_adf(y,run_test_stationarity_adf=None, subject_id=None):
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
           res_trend  = run_test_stationarity_adf_ts(pd.Series(y[:,index]), index,subject_id)   
           summary_list.append(res_trend)                                                                          
    return summary_list                                                                        
                                                                             
def run_test_stationarity_adf_ts(timeseries, index=None, subject_id=None):
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
        msgtitle = `subject_id`+ " Rolling (window:" + ` window`+ ")" +" mean and std deviation time sries: " + `index` 
        plt.title(msgtitle)
    #Perform Dickey-Fuller test
    autolag = 'BIC'
    #print 'Results of Dickey Fuller test with ', autolag
    dftest = adfuller(timeseries,maxlag=None, regression='c', autolag=autolag, store=False, regresults=False)
    # print dftest
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print "Time series:", index, "H0: ts has unit root (non stationary). p-value:", dfoutput['p-value'] 
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value                
    return dfoutput

def run_arima_fit(y, order=None,subject_id=None):
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
                res_trend  = run_arima_fit_ts(y[:,ind], ind, order,subject_id)
            except:
                warnings.warn("ERROR: The ARIMA model is not invertible!!!", Warning)
            summary_list.append(res_trend) 
    print "ARIMA per time series objects created:", summary_list        
    return summary_list                                                          
        
def run_arima_fit_ts(y, indexts=None, order=None,subject_id=None):
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
            msgtitle= `subject_id` +' for Timeseries:' + `indexts`+ ' using ARIMA(p,d,q):'+ `pe` + `de` + `qu`
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
        
def compute_nonlinear_statistics(timeseries):
    """Calculate a battery of significant statistics (non linear measures) for a time structure. 
    The argument "timeseries" can be a Pandas dataframe, Pandas time series or a ndarray.
    The measures that are calculated are: correlation dimension, sample entropy,
    Lyapunov exponents. It calls to compute_nonlinear_statistics_ts for each time series
    We use the numpy-based library NonLinear measures 
    for Dynamical Systems (nolds) 
    https://pypi.python.org/pypi/nolds/0.2.0
    """

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
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
    corr_dim_list = []
    ap_entropy_list =[]       
    lyap_list =[]
    hurst_list = []
    
    #print "\t Calculating the correlation dimension (degrees of freedom) using Grassberger-Procaccia_algorithm ..."
    # http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm
    #emb_dim=1 because we are dealing with one dimensional time series
    timepoints, voxels = timeseries.shape
    emb_dim = 2 
    print "Computing corr_dimension for time series shape:...", timeseries.shape 
    #with rvals by default (iterable of float): rvals=logarithmic_r(0.1 * std, 0.5 * std, 1.03))
    for i in range(0,voxels):
        corr_dim = nolds.corr_dim(timeseries[:,i], emb_dim=emb_dim)
        corr_dim_list.insert(i,corr_dim)
    print "correlation dimension all voxels:", corr_dim_list, " mean is:", np.mean(corr_dim_list), " emb_dim:", emb_dim
    
    print "\t Calculating the Lyapunov exponents(initial conditions sensibility)) of the time series..."
    for i in range(0,voxels):
        lyap = nolds.lyap_r(timeseries[:,i], emb_dim=emb_dim) # lyap_r for largest exponent, lyap_e for the whole spectrum
        lyap_list.insert(i,lyap)
    #print "Lyapunov largest exponent all voxels:", lyap_list, " mean is:", np.mean(lyap_list), " emb_dim:", emb_dim    
    
    print "\t Calculating the sample entropy(complexity of time-series) based on approximate entropy..."
    for i in range(0,voxels):
        ap_entropy = nolds.sampen(timeseries[:,i], emb_dim=emb_dim)
        ap_entropy_list.insert(i,ap_entropy)
    print "\t Calculating  Hurst exponent for long-term memory of the time series "
    for i in range(0,voxels):
        hurst = nolds.hurst_rs(timeseries[:,i])
        hurst_list.insert(i,hurst)
        #(if K = 0.5 there are no long-range correlations in the data,
        #if K < 0.5 there are negative long-range correlations, 
        #if K > 0.5 there are positive long-range correlations
    #print "Sample entropy all voxels:", ap_entropy_list, " mean is:", np.mean(ap_entropy_list), " emb_dim:", emb_dim    
    nonlmeasures = OrderedDict([('corr_dim',corr_dim_list), ('sampen', ap_entropy_list), ('lyap', lyap_list), ('hurst',hurst_list)])
    #print "emb_dim:",emb_dim , " Mean Correlation dimension:", nonlmeasures['corr_dim'], " Mean Sample entropy:", nonlmeasures['sampen'], " Mean Lyapunov exponents",nonlmeasures['lyap'] 
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

def ts_similarity(df1, df2, subject_id=None):    
    """Calculate dynamic time warping between time series
    return a matrix nxn fn is the number of time series Mij is the DTW distamce between two time series
    """
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
        ax.set(xlabel='null model ts voxel', ylabel='observed ts voxel')
        msgtitle = 'Dynamic Time Warping distance', subject_id
        ax.set_title(msgtitle)
        KS_array = np.asarray(KStest)
        #KS_array = KS_array.reshape(cols1, cols2)
        plt.figure()
        ax = sns.heatmap([KS_array], xticklabels=5, yticklabels=False)
        msgtitle = '%s Kolmogorov SmirnoffS ', subject_id
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
    elif label is 'SN':
        # Salience network
        dim_coords = []    
    else: 
        print " ERROR: label:", label, " do not found, returning empty list of coordinates!"   
    return dim_coords   

def save_dict_in_file(nonlinearresults,type_of_mask=None):
    """save dictionary into a csv file
    
    nonlinearresults: non linear measures
    """
    import csv
    filename = '%s_dictionaryoutput' % type_of_mask
    filename = filename + '.csv'
    w = csv.writer(open(filename, "w"))
    for key, val in nonlinearresults.items():
        w.writerow([key, val])
    print "Saved dict with nonlinar measures in file:", filename
    

def sparse_inverse_estimator(time_series=None, labels=None, coords=None):
    """sparse_inverse_estimator estimate sparse inverse covariance matrix (partial covariance)
    to construct a functional connectome using the sparse inverse covariance.
    
    time_series: extracted time series from which to perform the estimator
    """
    print("Estimating the sparse inverse covariance with GraphLassoCV....")

    estimator = GraphLassoCV()
    estimator.fit(time_series)
    print("Displaying the connectome matrix from the covariance")
    plt.figure(figsize=(10,10))
    cov_= estimator.covariance_
    numrows = cov_.shape[0]
    numcols = cov_.shape[1]
    plt.imshow(cov_, interpolation="nearest", vmax=cov_.max(), vmin=-cov_.max(), cmap=plt.cm.RdBu_r)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title('Graph Lasso covariance') 
    print('Displaying the Graph structure from the sparse covariance matrix')
    plotting.plot_connectome(cov_, coords.values(), title='covariance')
    
    print("Displaying the connectome matrix from the precision")
    plt.figure(figsize=(10,10))   
    prec_ = estimator.precision_
    # Normalize the precision matrix
    norm_prec_ = np.zeros(shape=(numrows,numcols))
    for row in range(0, numrows):
        for col in range(0, numcols):
           norm_prec_[row,col] = prec_[row,col]/(np.sqrt(prec_[row,row]*prec_[col,col]))
      
    norm_prec_ = np.abs(norm_prec_)
    vmax = .9 * prec_.max()
    #plt.imshow(-prec_, interpolation="nearest", vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r)
    plt.imshow(norm_prec_, interpolation="nearest", vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title('Graph Lasso precision matrix (inv cov)') 
    print('displaying the Graph structure from the sparse covariance matrix')
    #plotting.plot_connectome(-estimator.precision_, coords.values(), title='Sparse inverse covariance') 
    plotting.plot_connectome(norm_prec_, coords.values(), title='Sparse inverse covariance') 
    plotting.show()
    
    print('Displaying the connectome matrix from LeDoit shrinkage')
    plt.figure(figsize=(10,10))
    lw_cov_, _ = ledoit_wolf(time_series)
    lw_prec_ = linalg.inv(lw_cov_)
    # Normalize the precision matrix
    norm_lw_prec_ = normalize_prec_matrix(lw_prec_) 
 
    vmax=  .9 * lw_prec_.max()
    plt.imshow(norm_lw_prec_, interpolation="nearest", vmax=lw_prec_.max(), vmin=-lw_prec_.max(), cmap=plt.cm.RdBu_r)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title('Ledoit normalized precision (sparse invserse covariance)') 
    plotting.plot_connectome(norm_lw_prec_, coords.values(), title='Ledoit inverse covariance')    
    plotting.show()
    return estimator, norm_lw_prec_
 
def sparse_inverse_estimator_nsubjects(subject_time_series=None, labels=None, coords=None):  
    """plot connectome and matrix for group of subjects sparse covariance and orecision
    subject_time_series ndarray
    """
    edge_threshold = '60%' # 0.6 #'60%'
    
    print('Calling to nilearn.connectome.GroupSparseCovarianceCV \
        Sparse inverse covariance w/ cross-validated choice of the parameter')
    gsc = GroupSparseCovarianceCV(verbose=2)
    gsc.fit(subject_time_series)
    #edge_threshold = np.mean(gsc.covariances_[...,0]) + np.std(gsc.covariances_[...,0]) -4
    plotting.plot_connectome(-gsc.precisions_[...,0], coords.values(), edge_threshold=edge_threshold,
                             title=str(edge_threshold)+'-GroupSparseCovariancePrec', display_mode='lzr')
    plotting.plot_connectome(gsc.covariances_[...,0], coords.values(), edge_threshold=edge_threshold,
                             title=str(edge_threshold)+'-GroupSparseCovariance', display_mode='lzr')
    plot_covariance_matrix(gsc.covariances_[..., 0],gsc.precisions_[..., 0], labels, title = str(edge_threshold)+"-GroupSparseCovariance")
    plotting.show()
    # persistent homology analysis
    persistent_homology(gsc.covariances_[..., 0], coords)
    return
    #persistent_homology(-gsc.precisions_[..., 0], coords)
    #pdb.set_trace()
    
    #gl = GraphLassoCV(verbose=2)
    #print('gl object created!')
    #gl.fit(np.concatenate(subject_time_series))
    #print('gl fit performed')
    # plotting connectome
    #norm_gl_prec_ = normalize_prec_matrix(gl.precision_)
    #print('Calling to GL plot_connectome for covariance')
    #plotting.plot_connectome(gl.covariance_, coords.values(), edge_threshold=edge_threshold,title=str(edge_threshold)+'-Covariance', display_mode='lzr')
    #print('Calling to GL plot_connectome for precision')
    #plotting.plot_connectome(-gl.precision_, coords.values(), edge_threshold=edge_threshold, title='Sparse inverse covariance (GraphLasso)', display_mode='lzr')
    #plotting.plot_connectome(norm_gl_prec_, coords.values(), edge_threshold=edge_threshold, title=str(edge_threshold)+'-Norm Sparse inverse covariance (GraphLasso)', display_mode='lzr')   
    #plot_covariance_matrix(gl.covariance_, gl.precision_, title = "GraphLasso with l1 penalty", labels)
    #plot_covariance_matrix(gl.covariance_, norm_gl_prec_, labels, title = str(edge_threshold)+"-Norm GraphLasso with l1 penalty")

    #return gsc, gl
    return gsc

def plot_covariance_matrix(cov, prec, labels, title):
    """Plot covariances and precision matrices for a given processing"""
    prec = prec.copy()  # avoid side effects    
     # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    #prec[list(range(size)), list(range(size))] = 0
    span = max(abs(prec.min()), abs(prec.max()))
    
    # Display covariance matrix
    plt.figure()
    #plt.imshow(cov, interpolation='nearest', vmin=-1, vmax=1, cmap=plotting.cm.bwr)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)

    plt.imshow(cov, interpolation='nearest', vmin=0, vmax=1, cmap=plotting.cm.bwr)
    plt.colorbar()
    plt.title("%s / covariance" % title)
     # Display precision matrix
    plt.figure()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.imshow(prec, interpolation="nearest",
               vmin=-span, vmax=span,
               cmap=plotting.cm.bwr)
    plt.colorbar()
    plt.title("%s / precision" % title)

def normalize_prec_matrix(lw_prec_):
    """ Normalize precision matrix in absolute value """
    # Normalize the precision matrix
    numrows = lw_prec_.shape[0]
    numcols = lw_prec_.shape[1]
    norm_lw_prec_ =  np.zeros(shape=(numrows,numcols))
    for row in range(0, numrows):
        for col in range(0, numcols):
           norm_lw_prec_[row,col] = lw_prec_[row,col]/(np.sqrt(lw_prec_[row,row]*lw_prec_[col,col]))    
    
    return  np.abs(norm_lw_prec_)

def persistent_homology(matrix, coords=None, outputdir=None):
    """persistent_homology computes PH for a mtrix
    computes a matrix for every threshold (defined locally)
    compurtes netork and algebraic topology distances between matrices
    """
    #pdb.set_trace()
    thresholds = np.linspace(0.0,1.0,11) 
    listofmatrices = []
    outputdir = '/Users/jaime/vallecas/data/converters_y1/controls/figure_results/'
    group = 'controls'
    #outputdir = '/Users/jaime/vallecas/data/converters_y1/converters/figures_results/'
    #group = 'converters'
    #if prec normalize
    matrix = normalize_prec_matrix(matrix)
    for idx,val in enumerate(thresholds):
        # convert boolean matrix in 0 and 1 
        p = np.percentile(matrix,val*100)
        print('Building connectivity for percentile=', p, ' threshold=', val)                   
        mat = matrix > val
        mat = mat + 0
        listofmatrices.append(mat)
        #filename = 'prec_thr' + str(val) + '.png'
        filename = 'cov_thr' + str(val) + '.png'
        output_file  = os.path.join(outputdir, filename)
        # plot the connectomes for each thrreshold
        #title='thr=' + str(val) + ' ' + group
        plotting.plot_connectome(mat, coords.values(), edge_cmap='Blues',\
                                 output_file=output_file, display_mode='z')
    #save the mnd objects
    matricesfile = os.path.join(outputdir,'listofmatrices')
    np.save(matricesfile,listofmatrices)
    network_metrics(matricesfile + '.npy', outputdir)
    #magick montage -title 'ppp' -geometry +10+0 thr*.png out.png
    
def network_metrics(listfmatrix=None, outputdir=None):
        """compute network metrics for a list of matrices"
        np.load(listfmatrix)
        """
        if type(listfmatrix) is str:
           listfmatrix = np.load(listfmatrix) 
           max_cliques = []
           clustering = []
           triangles = []
           conn_components = []
           avg_shortestpath = []
           density = []
           x=np.asarray(range(0,len(listfmatrix)))
           for i in range(0,len(listfmatrix)):
               mat = listfmatrix[i]
               G=nx.from_numpy_matrix(mat)
               max_cliques.append(nx.graph_clique_number(G))
               clustering.append(nx.average_clustering(G))
               triangles.append(nx.triangles(G,0))
               density.append(nx.density(G))
               conn_components.append(nx.number_connected_components(G))
               # for disconnected graphs dont work
               #avg_shortestpath.append(nx.average_shortest_path_length(G))
           #plot the metrics
           plt.figure()
           plt.plot(x,max_cliques,'r',label=['max size cliques'])
           plt.plot(x, clustering,'b',label=['clustering'])
           plt.plot(x,density,'g',label=['density'])
           plt.plot(x, conn_components,'k',label=['conn_components'])
           #, conn_components,'g',label=['conn_cmp'], triangles,'k', label=['triangles'])
           plt.legend(loc='best')
           msgtitle = 'Network metrics. Controls' 
           plt.title(msgtitle)
           plt.xticks(x)
           #plt.ylabel('Correlation ts~ts shifted')
           plt.xlabel('threshold percentile')
           plt.legend()
           plt.show()
           outputfile = os.path.join(outputdir,'netmetrics')
           print('Saving the network metrics at:', outputfile)
           np.save(outputfile,[max_cliques,clustering,triangles,conn_components])