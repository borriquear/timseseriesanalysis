#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
time series characterization

@author: jaime
"""
import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from datetime import datetime, date, time
import statsmodels.tsa.stattools as tsa
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker
from nilearn import plotting
from nilearn import datasets
from nilearn import input_data
from nipy.labs.viz import plot_map, mni_sform, coord_transform
import seaborn as sns

def test():
    # Run tests for Collinearity, Autocorrelation, Stationarity
    # Redirect the output to file output.txt
    #f = open('output-bcpa0578ph635.txt','w')
    plt.rcParams.update({'figure.max_open_warning': 0})
    #f = open('output-session1-bcpa0537.txt','w')        
    #sys.stdout = f
    # Load Data Image
    image_file = load_fmri_image_name()
    # mask_file = None (entire brain), mask_file = 'coordinates' coordinates specified in load_masker
    #mask_file = None
    mask_file = 'coordinates'
    #mask_file= 'cort-maxprob-thr25-2mm'
    masker = load_masker(mask_file=mask_file)
    print " Calling to createDataFrame_from_image for Image:",image_file, "Mask:", mask_file
    timeseries = createDataFrame_from_image(image_file, masker)
    #return timeseries, mask_data
    print "Time series dimensions :", timeseries.shape
    # Check for Collinearity
    print " Calling to run_collinearity_on_ts(y) to test for collinearity in the time series"
    res_lagged = run_collinearity_on_df(timeseries)
    # Autocorrelation
    print "Calling to run_autoregression_on_ts(y) to test for autocorrelation in the time series" 
    res_acf = run_autoregression_on_df(timeseries)                                         
    # Stationarity
    print "Calling to run_test_stationarity_adf(y) to test for stationarity using augmented Dickey Fuller test" 
    res_stationarity = run_test_stationarity_adf(timeseries)
    # Fit the  model with ARIMA
    print "Calling to ARIMA model fit"
    res_arima = run_arima_fit(timeseries, order=None)
    # Forecasting analysis
    print "Studying predicatbility of the time series"
    res_forecasting = run_forecasting(res_arima)
    # res_lagged, res_acf, res_stationarity are list, to print the results: res_acf[volxel_index].summary() 
    
    return timeseries, masker, res_lagged, res_acf, res_stationarity 

def  load_fmri_image_name(image_file=None):
    # Load fmri image (4D object)
    if image_file is None:
        dir_name = '/Users/jaime/vallecas/mario_fa/FA_50'
        #dir_name = '/Users/jaime/vallecas/data/cadavers/nifti/bcpa_0537/session_1/PVALLECAS3/reg.feat'
        #dir_name = '/Users/jaime/vallecas/data/cadavers/nifti/bcpa_0537/session_0/reg.feat'
        #dir_name = '/Users/jaime/vallecas/data/cadavers/dicom/fromOsiriX_bcpa0650_DEAD_2016_ID7741/niftis'
        f_name = 'filtered_func_data.nii.gz'
        f_name = '20170126_172022rsfMRIFA7s007a1001.nii.gz'
        #f_name = '10000_fMRI_RESTING_S_20170221164921_8.nii'
        image_file = os.path.join(dir_name, f_name)
    return image_file
    
def load_masker(mask_file=None):   
    # Returns the Masker object from an Atlas (mask_file) or a list of voxels     
    if mask_file is None:    
        print "No mask used, process the entire brain"
        return None
    elif mask_file is 'coordinates':
        #
        #, (0, -52, 18),(-46, -68, 32), (46, -68, 32), (1, 50, -5)
        #http://sprout022.sprout.yale.edu/mni2tal/mni2tal.html
        dmn_coords = [(-29,-19,-15), (3, -1, -11), (-52, -19, 7), (29, -92, 2)]
        dmn_coords = [(-52, -19, 7)]
        # Left Hipp, right Hipp, Left Prim Auditory (41),Right visual Ass (18) 
        labels = [
                'Posterior Cingulate Cortex',
                'Left Temporoparietal junction',
                'Right Temporoparietal junction',
                'Medial prefrontal cortex'
                ]   
        print " The mask is list of voxels:", labels
        masker = input_data.NiftiSpheresMasker(dmn_coords, radius=8,
                                               detrend=True, standardize=True,
                                               low_pass=0.2, high_pass=0.001, 
                                               t_r=2.5,memory='nilearn_cache', 
                                               memory_level=1, verbose=2)
    else:
        # The mask is an Atlas
        dataset = datasets.fetch_atlas_harvard_oxford(mask_file)
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
        image_file = load_fmri_image_name(image_file=None)
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

def run_collinearity_on_df(y):
    # Test for multicollinearity check type of data first
    if type(y) is pd.core.series.Series:
        run_collinearity_on_ts(y)
    if type(y) is np.ndarray:
        # Convert df into timeseries and call each time
        summary_list = []
        #return mask
        for index in np.arange(0,y.shape[1]):
            print "===Estimating collinearity for ts ROI:", index, "/", y.shape[1]-1
            # print pd.Series(y[:,index])
            res_lagged = run_collinearity_on_ts(pd.Series(y[:,index]))
            summary_list.append(res_lagged)
     
    return summary_list
               
def run_collinearity_on_ts(y):
    # Test for multicollinearity for the timeseries considered as a regression problem.
    # Multicollinearity (collinearity) is a phenomenon in which two or more predictor variables in a multiple regression model are highly correlated
    # Shift the timeseries lag index times and calculate the correlation between the coefficients in the regression.
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
        plt.figure()
        sns.heatmap(df.corr(),vmin=-1.0, vmax=1.0)        
        plt.figure()
        ax = plt.axes()
        ax.set_title('Correlation of the lagged coefficients: y ~ b(0) + ...+ b(5)')
        ax = res_lagged.params.drop(['Intercept', 'trend']).plot.bar(rot=0)
        plt.ylim(-1,1)
        plt.ylabel('Coefficient')  
        ax = sns.despine()  
        plt.show()
    
    return res_lagged
    
def run_autoregression_on_df(y):
    if type(y) is pd.core.series.Series:
        run_autoregression_on_ts(y)
    if type(y) is np.ndarray:
        # Convert df (np.ndarray) into timeseries and call each time
        summary_list = []
        for index in np.arange(0,y.shape[1]):
            print "===Estimating autoregression for ts ROI:", index, "/", y.shape[1]-1 
            res_trend  = run_autoregression_on_ts(pd.Series(y[:,index]))                                                                
            summary_list.append(res_trend)                                                                
            print res_trend.summary()
                                                                            
    return summary_list        
                                                                
def run_autoregression_on_ts(y):  
    # Autocorrelation also called serial correlation is when there is a pttern between the observed and the predicted in the regression model.
    mod_trend = sm.OLS.from_formula('y ~ trend', data=y.to_frame(name='y').assign(trend=np.arange(len(y))))
    res_trend = mod_trend.fit()
    # Residuals (the observed minus the expected, or $\hat{e_t} = y_t - \hat{y_t}$) are supposed to be white noise. 
    # Plot the residuals time series, and some diagnostics about them
    plotacfpcf = True
    if plotacfpcf is True:    
        tsplot(res_trend.resid, lags=36)
    return res_trend

def run_test_stationarity_adf(y):
    if type(y) is pd.core.series.Series:
        run_test_stationarity_adf_ts(y)
    if type(y) is np.ndarray:    
        # Convert df (np.ndarray) into timeseries and call each time
        summary_list = []
        for index in np.arange(0,y.shape[1]):
           print "===Estimating stationarity for ts ROI:", index, "/", y.shape[1]-1                                                               
           res_trend  = run_test_stationarity_adf_ts(pd.Series(y[:,index]))   
           summary_list.append(res_trend)                                                                          
    return summary_list                                                                        
                                                                             
def run_test_stationarity_adf_ts(timeseries):
    #Determing rolling statistics
    #body = 'phantom'
    toplot = True
    if toplot is True:
        plt.figure()
        rolmean = pd.rolling_mean(timeseries, window = 10)
        rolstd = pd.rolling_std(timeseries, window = 10)
        #Plot rolling statistics
        orig = plt.plot(timeseries, color = 'blue', label = 'Original')
        mean = plt.plot(rolmean, color = 'red', label = 'Rolling mean')
        std = plt.plot(rolstd, color = 'black', label = 'Rolling std')
        plt.legend(loc='best')
        plt.title("Rolling mean and std deviation voxel max power ")
    #plt.show(block=False)
    #Perform Dickey-Fuller test
    autolag = 'BIC'
    #print 'Results of Dickey Fuller test with ', autolag
    #dftest = adfuller(timeseries, autolag)
    dftest = adfuller(timeseries,maxlag=None, regression='c', autolag=autolag, store=False, regresults=False)
    # print dftest
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print "H0: ts has unit root (non stationary). p-value:", dfoutput['p-value'] 
    for key,value in dftest[4].items():
        #print key, value
        dfoutput['Critical Value (%s)'%key] = value
                
    #print dfoutput
    return dfoutput

def run_arima_fit(y, order=None):
    # Fit the time seris with ARIMA model
    if type(y) is pd.core.series.Series:
        run_arima_fit_ts(y, order)
    if type(y) is np.ndarray:   
        # Convert df (np.ndarray) into timeseries and call each time
        summary_list = []
        for ind in np.arange(0,y.shape[1]):
            print "===Fitting ARIMA, ROI:", ind, "/", y.shape[1]-1   
            #res_trend  = run_arima_fit_ts(pd.Series(y[:,ind]),order) 
            #You need datetime index when you use pandas, but you can use a numpy ndarray for the series which does not need any index.
            res_trend  = run_arima_fit_ts(y[:,ind], order)
            summary_list.append(res_trend) 
    return summary_list                                                          
        
def run_arima_fit_ts(y, order=None):
    # Fit timeseries with a ARIMA (SRIMAX) sesonal autoregressive integrated moving average with exogenbeous regressors
    if order is None:
        # Assign order by default
        p=1
        d=1
        q=1
    else:
        p = order[0]
        d = order[1]
        q = order[2]
    #res_arima = sm.tsa.ARIMA(y, order=(p,d,q)).fit(method='mle', trend='nc')
    res_arima = sm.tsa.ARIMA(y, order=(p,d,q)).fit(method='mle', trend='nc')
    print "ARIMA summary"
    print res_arima.summary()
    #print "ARIMA AIC=", res_arima.aic
    arima_plot = True
    if arima_plot is True:
        tsplot(pd.Series(res_arima.resid[2:]), lags=36)
    return res_arima    
                                      
def run_forecasting(res_arima):
    pred = res_arima[0].predict()
    print "Prediction", pred
    # Pred is the predcited values ndarray
    #pred_ci = pred.conf_int()
    ax = res_arima[0].plot_predict()
    #pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
    #ax.fill_between(pred.index,pred.iloc[:, 0],res_arima, color='k', alpha=.2)
    plt.legend()
    sns.despine()

    
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
                df = pd.concat([df, timeseries], axis = 1 )
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


def tsplot(y, lags=None, figsize=(10, 8)):  
    fig = plt.figure(figsize=figsize)    
    layout = (2, 2)           
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)   
    acf_ax = plt.subplot2grid(layout, (1, 0))  
    pacf_ax = plt.subplot2grid(layout, (1, 1))   
    y.plot(ax=ts_ax, title='Regression residuals ts residuals  y_t - \hat{y_t}') 
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)        
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)    
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]] 
    sns.despine()                            
    plt.tight_layout() 
    return ts_ax, acf_ax, pacf_ax  