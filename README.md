Time series analysis of signal intensity in pixels.

cd /Users/jaime/github/code/python_trn
import test_stationarity

To run the test with image_file and masker.
test_stationarity.run_test()

This is what test does:
- The std output is set to a .txt file
- Load the image load_fmri_image() using Nibabel 
- load_masker(mask_file=mask_file)
- timeseries = test_stationarity.createDataFrame_from_image()
- Check for collinearity, res_lagged = run_collinearity_on_df(timeseries)
- Check for autoregression, res_acf = run_autoregression_on_df(timeseries)  
- Check for stationarity with ADF test, res_stationarity = run_test_stationarity_adf(timeseries)

Alternatively, to just load the image and the masker
- timeseries = test_stationarity.createDataFrame_from_image(image_file = '...',masker=None|'coordinates', 'cort-maxprob-thr0-2mm' )
The atlas name is from http://nilearn.github.io/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html
To use a different atlas change line ataset = datasets.fetch_atlas_harvard_oxford(mask_file) in load_masker
