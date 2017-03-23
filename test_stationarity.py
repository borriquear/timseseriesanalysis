#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:56:06 2017

@author: jaime
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from datetime import datetime, date, time
import statsmodels.tsa.stattools as tsa
import nibabel as nib

def test_stationarity(timeseries):
    #Determing rolling statistics
    #body = 'phantom'
    plt.figure()
    rolmean = pd.rolling_mean(timeseries, window = 10)
    rolstd = pd.rolling_std(timeseries, window = 10)
    #Plot rolling statistics
    orig = plt.plot(timeseries, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling std')
    plt.legend(loc='best')
    plt.title("Rolling mean and std deviation voxel max power in cadaver")
    #plt.show(block=False)
    #Perform Dickey-Fuller test
    print 'Results of Dickey Fuller test:'
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

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
    
def unit_root_test(df=None):
    # H0: of the Augmented Dickey-Fuller is that there is a unit root
    # HA: is stationary (there is no unit root)
    # If the pvalue is above a critical size, then we cannot reject H0 that there is a unit root (non statiuonary).
    print "aDF test, H0: there is an unit root (non-stationary) HA: stationary"
    adf_results = {}
    if df is None:
        # Build the time series 
        print " Creating the data frame"
    for col in df.columns.values:
        ts_pervoxel = df[col]
        #print ts_pervoxel
        adf_results[col] = tsa.adfuller(ts_pervoxel)  
        #print tsa.adfuller(ts_pervoxel)
        print "p-value H0: unique root (non stationary) %.5f", adf_results[col][1]                          

def createDataFrame_from_image(image_data=None):
    if image_data is None:
        # Load image
        data_path = '/Users/jaime/vallecas/mario_fa/mario_fa_dicoms/FA10_SIN_rf' 
        image_file = '20170126_172022rsfMRIFA4s005a1001.nii.gz'
        image_file = os.path.join(data_path, image_file)
        # nibabel.nifti1.Nifti1Image
        image = nib.load(image_file)
        # Get # numpy.ndarray image_data
        image_data = image.get_data()
    # Create DataFrame from ndarray
    # Get voxel location and labels from mask
    df = extract_ts_from_mask(image_data, voxels_list = [[1,1,1],[30,20,20],[21,22,15]])
    return df
    #df = pd.DataFrame(image_data[1,2,3,:], columns=list('A'))                           

def extract_ts_from_mask(image_data, voxels_list=[], mask_id=None):
    df = []
    if len(voxels_list) > 0:
        df = pd.DataFrame(image_data[voxels_list[0][0], voxels_list[0][1], voxels_list[0][2]])
        for voxel_i in voxels_list[1:]:
            label = str(voxel_i[0]) +  str(voxel_i[1]) +  str(voxel_i[2]) 
            #print label
            df[label] = pd.Series(image_data[voxel_i[0],voxel_i[1],voxel_i[2]], index = df.index)
        return df    
                                          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            