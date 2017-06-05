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
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker
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
from nitime.timeseries import TimeSeries
# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer, GrangerAnalyzer
import plotly
plotly.tools.set_credentials_file(username='borriquear', api_key='ZGW7Nrb6GptJzSV7W6VY')
import plotly.plotly as py
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import warnings


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
    print "Plotting time series"
    plot_ts_network(time_series, dim_coords.keys())
#    if time_series.shape[0] != epi_params:
#        warnings.warn("The time series number of points is !=116 check your bold_data.nii !!", Warning)
#    print "Masker is:", masker, " \n \t and time series shape", time_series.shape #, ", time_series", time_series
#    
#    # Characterize the time series
#    print "Testing for collinearity in the time series"
#    res_lagged = run_collinearity_on_df(time_series)
#    print "Calling to run_autoregression_on_ts(y) to test for autocorrelation in the time series" 
#    res_acf = run_autoregression_on_df(time_series)      
#    print "Calling to run_test_stationarity_adf(y) to test for stationarity using augmented Dickey Fuller test" 
#    res_stationarity = run_test_stationarity_adf(time_series)                                    
#    print "Calling to ARIMA model fit"
#    res_arima = run_arima_fit(time_series, order=None)
#    #print "Calling to run_forecasting to study predictability of the time series for the ARIMA"
#    #res_forecasting = run_forecasting(res_arima)
#    #return masker, time_series, res_lagged, res_acf, res_stationarity, res_arima 
#    print "Displaying the connectome"
#    kind_of_correlation = 'correlation'
#    corr_matrix = plot_ts_connectome(time_series, kind_of_correlation)
#    print corr_matrix
    #Granger causality of the time series
    grangerresult = test_of_grangercausality(time_series)
    # Calculate the G causality among pairs of time series  
    [corrm_xy, corrm_yx, listoffreqs]  = compute_granger_pair_of_ts(time_series)
    for ff in range(0, len(listoffreqs)):
        print "XY Granger at freq:", listoffreqs[ff], " =", corrm_xy[:,:,listoffreqs[ff]]
        print "YX Granger at freq:", listoffreqs[ff], " =", corrm_yx[:,:,listoffreqs[ff]]
        # average on last dimension (frequency and plot)
        #fig03 = drawmatrix_channels(g1, roi_names, size=[10., 10.], color_anchor=0)
    return 
    fourier_analysis = False
    if fourier_analysis is True:
        print "Calculating the PSD of the time series"
        psd_original = fourier_spectral_estimation(time_series)
        
    surrogate_analysis = True
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


def plot_ts_network(ts, lab):
    # plot_ts_network  Plot time series for a network 
    # Input: ts time series object
    # lab: list of labels
    plt.figure()
    for ts_ix, label in zip(ts.T, lab):
        #pdb.set_trace()
        plt.plot(np.arange(0,len(ts_ix)), ts_ix, label=label)
        
    plt.title('Default Mode Network Time Series')
    plt.xlabel('Time points (TR=2.5s)')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.tight_layout()
    
def plot_ts_connectome(ts, kind_of_corr=None, labelnet=None):
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
         
    
def  load_fmri_image_name(image_file=None):
    # Load fmri image (4D object)
    image_params = {'TR':2.5, 'n':120-4, 'fs': 0.4, 'nfft':129,'duration_in_s':120*2.5}
    image_params = {'TR':2.5, 'n':120, 'fs': 0.4, 'nfft':129,'duration_in_s':120*2.5}
    if image_file is None:
        dir_name = '/Users/jaime/vallecas/mario_fa/RF_off'
        dir_name = '/Users/jaime/vallecas/data/cadavers/nifti/bcpa_0537/session_1/PVALLECAS3/reg.feat'
        dir_name = '/Users/jaime/vallecas/data/surrogate/bcpa0517'
        dir_name = '/Users/jaime/vallecas/data/surrogate/bcpa0537_0/'
        #dir_name = '/Users/jaime/vallecas/data/cadavers/dicom/fromOsiriX_bcpa0650_DEAD_2016_ID7741/niftis'
        f_name = 'bold_data_mcf2standard.nii.gz'
        f_name = 'bold_data.nii'
        #f_name = '20170126_172022rsfMRIFA7s007a1001.nii.gz'
        #f_name = '20170126_172022rsfMRIFA4s005a1001.nii.gz'
        image_file = os.path.join(dir_name, f_name)
    return image_file, image_params
    
def load_masker(dim_coords=None):   
    """Returns the Masker object from an Atlas (mask_file) or a list of voxels
    
    Input: dim_coords None || [] returns None (in the future will return a list of brain voxels, the full brain).
    dim_coords: dictionary of a network of interest eg DMN.
    dim_coords: list of coordinates of voxels.    
    dim_coords: atlas_harvard_oxford, eg 'cort-maxprob-thr25-2mm'  

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
    elif type(dim_coords) is dict:  
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


def compute_granger_pair_of_ts(time_series):
    """Calculate the Granger causality for pairs of tiem series
    """
    
    image_params = load_fmri_image_name()[1]
    #frequencies = np.linspace(0,0.2,129)
    combinations = list(itertools.combinations(range(time_series.shape[1]),2)) 
    time_series = np.transpose(time_series)
    time_series = nitime.timeseries.TimeSeries(time_series, sampling_interval=image_params['TR'])
    
    granger_sys = GrangerAnalyzer(time_series, combinations)
    listoffreqs = granger_sys.frequencies[:]
    corr_mats_xy = granger_sys.causality_xy[:,:,:]
    corr_mats_yx = granger_sys.causality_yx[:,:,:]
    return [corr_mats_xy, corr_mats_yx, listoffreqs]

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
            #pdb.set_trace()
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
            #pdb.set_trace()
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
        label = [
                'Posterior Cingulate Cortex',
                'Left Temporoparietal junction',
                'Right Temporoparietal junction',
                'Medial prefrontal cortex'
                ] 
        # Dictionary of network with MNI components. 
        # http://nilearn.github.io/auto_examples/03_connectivity/plot_adhd_spheres.html#sphx-glr-auto-examples-03-connectivity-plot-adhd-spheres-py
        # dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)] 
        dim_coords = {label[0]:(0, -52, 18),label[1]:(-46, -68, 32), label[2]:(46, -68, 32),label[3]:(1, 50, -5)} #from nilearn page
        #dim_coords = {label[0]:(0, -52, 18),label[1]:(0, -52, 20), label[2]:(46, -68, 32),label[3]:(1, 50, -5)} #from HEDDEN ET AL (2009) PCC
        
        #dim_coords = [(-5, -53, 41), (0, 52, -6), (-48, -62, 36), (46, -62, 32)]
    elif label is 'SN':
        # Salience network
        dim_coords = []    
    else: 
        print " ERROR: label:", label, " do not found, returning empty list of coordinates!"   
    return dim_coords   