# GLM example data

Example data resulting from GLM modeling is shown here in GLM_eg_data.npy \
GLM_eg_data.npy is a numpy python dictionary. here is an example of a single neuron fitted to a GLM model using parameters described in the manuscript. \
We first ran the GLM, then saved the data to avoid running the model again for further analysis. 

- alpha: best alpha value for Elasticnet regression fitting
- coef: individual model kernels for each task variable. Only fitted task variables are shown here
- init_score: initial r2 fitting score for each task variable
- L : lick rate (binned by 50ms window)
- r_onset : reward onset
- score : final score for the fully built model, validated 20 times
- stim_onset: stimulus onset
- theta : fully built model coefficient weights (validated 20 times)
- X4: fully built model kernel
- Y : Calcium trace (dF/F0) (binned by 50ms window)
- yhat : model fitted trace.  (binned by 50ms window)

# List of alpha values

Because it is extremely time consuming to fit alpha values for each neuron when fitting the neural activity to our GLM, we first found the best alpha values from a selection ranging from 1e-4 to 1e0 (as explained in the model) then saved this list for future model fittings. \
Each neuron therefore has a single alpha value.
