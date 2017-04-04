#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
time series characterization

@author: jaime
"""
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
from nilearn import plotting
from nilearn import datasets
from nipy.labs.viz import plot_map, mni_sform, coord_transform
import seaborn as sns

def test():
    # Load Data Image
    image_file = os.path.join('/Users/jaime/vallecas/mario_fa/mario_fa_dicoms/FA10_SIN_rf', '20170126_172022rsfMRIFA4s005a1001.nii.gz')
    mask = 'cort-maxprob-thr25-2mm'
    print " Calling to createDataFrame_from_image for Image:",image_file, "\nMask:", mask
    [timeseries, mask_data]= createDataFrame_from_image(image_file,mask)
    return
    print "Atlas name:", mask_data.maps, "\nROIs of the mask:", mask_data.labels
    print " DataFrame extracted dimensions:", timeseries.shape
    # Check for Collinearity
    print " Calling to run_collinearity_on_ts(y) to test for collinearity in the time series"
    res_lagged = run_collinearity_on_df(timeseries, mask_data)
    # Autocorrelation
    # Stationarity
    return timeseries, mask_data, res_lagged

def createDataFrame_from_image(image_file=None, mask=None):
    # createDataFrame to work with pandas time series from nifti image 
    # If no mask, the mask is the entire brain and return pvalues array[x,y,zpavlue] if mask is given return the time series for each 
    # ROI of the mask. nomask doesnt return time series but 1 if the voxel is non stationay 0 if it is.
    if image_file is None:
        # Load image
        data_path = '/Users/jaime/vallecas/mario_fa/mario_fa_dicoms/FA10_SIN_rf' 
        image_file = '20170126_172022rsfMRIFA4s005a1001.nii.gz'
        image_file = os.path.join(data_path, image_file)
    image_data = nib.load(image_file).get_data()
    dims_image = image_data.shape
    # Create DataFrame from ndarray
    # Generate the voxels_list All brain voxels or specific voxel coordinates that correspond to areas of interest
    if mask is None:
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
        # Specify the mask with particular rois
        # Get voxel location and labels from mask, for example mask = 'cort-maxprob-thr25-2mm'
        dataset = datasets.fetch_atlas_harvard_oxford(mask)
        atlas_filename = dataset.maps
        #labels = dataset.labels
        # Plot the atlas
        plotting.plot_roi(atlas_filename)
        # Build the mask
        masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
        time_series = masker.fit_transform(image_file)
        #df = extract_ts_from_mask(image_data, voxels_list=[], mask)
        print masker
        return time_series, dataset

def run_collinearity_on_df(y, mask=None):
    # Test for multicollinearity check type of data first
    print y.shape
    if type(y) is pd.core.series.Series:
        run_collinearity_on_ts(y)
    if type(y) is np.ndarray:
        # Convert df into timeseries and call each time
        print enumerate(mask.labels)
        #return mask
        for index, item in enumerate(mask.labels):
            print "Estimating collinearity for ts=", item, index
            print pd.Series(y[:,index])
            run_collinearity_on_ts(pd.Series(y[:,index]))
    
def run_collinearity_on_ts(y):
    # Test for multicollinearity for the timeseries considered as a regression problem.
    # Multicollinearity (collinearity) is a phenomenon in which two or more predictor variables in a multiple regression model are highly correlated
    # Shift the timeseries lag index times and calculate the correlation between the coefficients in the regression.
    lags = 6
    # Create a dataframe with lagged values of timeseries (y), shifts the index i periods
    df = (pd.concat([y.shift(i) for i in range(lags)], axis=1,keys=['y'] + ['y%s' % i for i in range(1, lags)]).dropna())
    df.head()
    return df
    # Fit the lagged model using statsmodels
    mod_lagged = smf.ols('y ~ trend + y1 + y2 + y3 + y4 + y5',data=df.assign(trend=np.arange(len(df))))
    res_lagged = mod_lagged.fit()
    res_lagged.summary()
    # If there is no multicollinearity, the coefficient estimation is max in abs value for \beta1 and min for \beta5
    # If  the lagged values are highly correlated with each other the estimates of the slopes are not reliable
    plotresults = False
    if plotresults:
        ax = plt.axes()
        # Correlation of the coefficients 
        sns.heatmap(df.corr(),ax = ax)
        ax.set_title('Correlation of the lagged coefficients: y ~ b(0) + ...+ b(5)')
        plt.show()
        # If there is no multicollinearity, we expect the coefficients gradually decline to zero. 
        plt.figure()
        ax2 = res_lagged.params.drop(['Intercept', 'trend']).plot.bar(rot=0)
        plt.ylabel('Coefficient')     
        sns.despine()    
        plt.show()  
    return res_lagged
    
def test_stationarity_adf(timeseries):
    #Determing rolling statistics
    #body = 'phantom'
    toplot = False
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
    autolag = 'AIC'
    #print 'Results of Dickey Fuller test with ', autolag
    #dftest = adfuller(timeseries, autolag)
    dftest = adfuller(timeseries,maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print dfoutput
    return dfoutput
    
def load_matfile(matfile='/Users/jaime/vallecas/sub_bcpa0537/session_0/func/session_f004/matlab_results/cadaverbcpa0537-0_81_ON.mat'):
    #/Users/jaime/vallecas/sub_bcpa0537/session_0/func/session_f004/matlab_results/cadaverbcpa0537-0_81_ON.mat
    #/Users/jaime/vallecas/sub_bcpa0537/session_1/func/session_f008/matlab_results/brainbcpa0537-1_81_ON.mat
    print 'Loading mat file---:', matfile
    lisofvoxels = [(55,112,16),(57,112,17),(56,112,17)] #cad
    lisofvoxels = [(55,112,16),(57,112,15),(56,112,16)] #cad
    #lisofvoxels = [(30,40,30),(32,40,31),(31,40,30)] #phantom
    lisofvoxels = [(60,71,2),(50,98,5),(50,97,5)] #brain
    lisofvoxels = [(58,71,2),(50,98,5),(50,97,5)] #brain
    mdata = loadmat(matfile)
    print mdata.keys()
    ts = mdata['ts']
    #save mat file -v7.3 and load in python with  HDF5
    #import h5py
    #f = h5py.File(file2save, 'r')
    #myts = f.get('ts')
    #myts[:,2,71,60] (time, z,y,x)
    #pws = mdata['pws']
    #amps = mdata['amps']
    #meanspsdperbin = mdata['meanspsdperbin']
    #imgmeanspwnnz = mdata['imgmeanspwnnz']
    #pw_timeseries = pws[30,50,12,:]
    #pw_timeseries = pws[30,50,12,:]
    #print pw_timeseries[lisofvoxels[1],:]
    # plot time series for each of the three maximapoints
    for p in range(0,lisofvoxels.__len__()):  
        print 'Plotting the stationary test for voxel: ', lisofvoxels[p]
        timeseries= ts[lisofvoxels[p][0],lisofvoxels[p][1],lisofvoxels[p][2],:]
        test_stationarity.test_stationarity(timeseries)
                
def var_modeling():
    # Load the data
    tsa = sm.tsa
    mdata = sm.datasets.macrodata.load().data
    mdata = mdata['realgdp','realcons','realinv'] 
    names = mdata.dtypes.names
    data = mdata.view((float,3))
    data = np.diff(np.log(data), axis = 0)
    # Build VAR model to fit the data
    model = tsa.VAR(data)
    res = model.fit(2)  
    # Plot results  
    print res.summary
    res.plot_sample_acorr()
    # Test for Granger Causality
    res.test_causality('realinv','realcons')

def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df    
# test_stationarity.shuffle(y)
    
def run_unit_root_test(df=None):
    # H0: of the Augmented Dickey-Fuller is that there is a unit root
    # HA: is stationary (there is no unit root)
    # If the pvalue is above a critical size, then we cannot reject H0 that there is a unit root (non statiuonary).
    # statistical significance for ADFuller test
    stat_threshold = 0.005 
    print "aDF test, H0: there is an unit root (non-stationary) HA: stationary"
    adf_results = {}
    if df is None:
        # Build the time series 
        print " Creating the data frame"
    for col in df.columns.values:
        ts_pervoxel = df[col]
        # Plot the original time series and the difference time series, to visually inspect if is stationary, prior to run the ADFuller test
        ts_pervoxel.to_frame(name='y').assign(deltay=lambda x: x.y.diff()).plot(subplots=True)
        #print ts_pervoxel
        adf_results[col] = tsa.adfuller(ts_pervoxel)  
        #print tsa.adfuller(ts_pervoxel)
        print "p-value H0: unique root (non stationary) %.5f", adf_results[col][1]   
        # If the timse series is not stationary -fail to reject the null hypothesis that the original series was non-stationary-  we difference it
        adf_res = tsa.adfuller(ts_pervoxel.diff().dropna()) 
        if adf_res > stat_threshold:
            # We fit a model of delta y
            data = (ts_pervoxel.to_frame(name='y').assign(deltay=lambda df: df.y.diff()).assign(Ldeltay=lambda df: df.deltay.shift()))                  
            mod_stationary = smf.ols('deltay ~ Ldeltay', data=data.dropna())
            res_stationary = mod_stationary.fit()
            test_stationarity.tsplot(res_stationary.resid, lags=24);
        
               


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
                print df[df.columns[-1]]               
                if np.mean(timeseries) > 10:
                    print " testing ADF for time series in index , mean",i,j,k, np.mean(timeseries)
                    dfoutput = test_stationarity_adf(timeseries)  
                    # print "H0: unit root non stationary, p value :", dfoutput[1]
                    if dfoutput[1] > threshold_pvalue:
                        counter_nonstat =  counter_nonstat + 1
                        pvalues[i,j,k] =  dfoutput[1]
                        print "We reject H0: unit root or non stationary, p value at x,y,z", dfoutput[1], label
    return pvalues    


def plot_brain_voxels_test(pvalues):
    # Activation map for a 3D image in MNI space 
    x, y, z = 0, 0, 0
    mni_sform_inv = np.linalg.inv(mni_sform)
    pv_dims = pvalues.shape
    pv_dims_a = [pv_dims[0], pv_dims[1], pv_dims[2]]
    map = np.zeros(pv_dims_a)
    map = np.zeros((182, 218, 182))
    for i in np.arange(0, pv_dims[0]-1):
        for j in np.arange(0, pv_dims[1]-1):
            for k in np.arange(0, pv_dims[2]-1):
                label = [i,j,k]
                if  pvalues[i,j,k] > 0:
                         map[i,j,k] = 1
    # We use a masked array to add transparency to the parts that we are
    # not interested in:
    thresholded_map = np.ma.masked_less(map, 0.5)
    # And now, visualize it:
    plot_map(thresholded_map, mni_sform, cut_coords=(x, y, z), vmin=0.5)                        
   
            
    
def run_autoregression_on_ts(y):  
    # Autocorrelation also called serial correlation is when there is a pttern between the observed and the predicted in the regression model.
    mod_trend = sm.OLS.from_formula('y ~ trend', data=y.to_frame(name='y').assign(trend=np.arange(len(y))))
    res_trend = mod_trend.fit()
    # Residuals (the observed minus the expected, or $\hat{e_t} = y_t - \hat{y_t}$) are supposed to be white noise. 
    # Plot the residuals time series, and some diagnostics about them
    tsplot(res_trend.resid, lags=36)
       

def tsplot(y, lags=None, figsize=(10, 8)):     
    fig = plt.figure(figsize=figsize)    
    layout = (2, 2)           
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)   
    acf_ax = plt.subplot2grid(layout, (1, 0))  
    pacf_ax = plt.subplot2grid(layout, (1, 1))   
    y.plot(ax=ts_ax, title='Regression residuals ts residuals  y_t - \hat{y_t}')  
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, title='autocorrelation of residuals (e_t, e_{t-k}} )')        
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, title='partial (ontrolling for shorter lags) autocorrelation of residuals')    
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]   
    sns.despine()                             
    plt.tight_layout() 
    return ts_ax, acf_ax, pacf_ax  