# timseseriesanalysis
Time Series Characterization
# Load nifti image using nibabel
cd /Users/jaime/github/code/python_trn
import test_stationarity
# df contains the time series 
df = test_stationarity.createDataFrame_from_image()
image_data.shape
# Input image otherwise create in createDataFrame_from_image
# Define mask or list of voxels when call to extract_ts_from_mask
df = test_stationarity.createDataFrame_from_image()
# Performs a unit root test (adf)
unit_root_test(df)
