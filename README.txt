################## Method classes ##################
_______________Selection.py_______________

3 classes:
----------- Criteria_selection -----------
Make the spectroscopic selection as BTS and the galactic extinction cut
**** Input****
Initial data frame with simulated SNe Ia (For the different methods there other input)
****Output****
Different output depending on the method like create a flag for spectroscopic selection or return ra, dec and z selected value for the galactic extinction function

----------- Good_sampling -----------
Selection the good number of point in light curve to know if we can take this SN or not
****Input****
Simulation number, filename (name of SN) and a data frame that correspond with the Salt2 fit.
****Output****
Flag on the data frame

----------- salt_2_selection -----------
Selection on Salt2 parameters
****Input****
Data frame for Salt2 fit
****Output****
Data frame after selection


_______________SN_distribution.py_______________

2 classes:
----------- SkyMap -----------
Plot in mollweide view the position of Supernovae in equatorial, galactic or ecliptic coordinates and display the redshift of each Supernovae with a color bar; Projection of pixel or scatter plot
****Input****
Resolution (power of 2), longitude, latitude, variable, variable name and a title
****Output***
Sky map of the SNe Ia distribution

----------- Patching_Map(SkyMap) -----------
partitions the map into patches and make some Hubble diagram
****Input****
Same as SkyMap
****Output****
Different partition of the sky map and Hubble diagrams (Add a flag/number on the data frame that correspond to the number of the patch to identify which patch SNe Ia belongs to)

_______________Hubble.py_______________
3 classes:
----------- Hubble_fit -----------
Fitting Hubble diagram by a chi square minimization
****Input****
Hubble constant and data frame with SNe Ia
****Output****
Fitted parameters: M_b, alpha and beta

----------- Hubble_fit_patch -----------
Fitting Hubble diagram per patch
****Input****
Hubble constant, data frame with SNe Ia and number of patch
****Output****
Fitted parameters: alpha, beta and different variation of H_0 per patches

----------- dipole -----------
Dipole fit on the fitted delta_H_0, to retrieve initial dipole
****Input****
Hubble constants from patches, errors on H_0 and coordinates (ra, dec) in radian
****Output**** 
Fitted parameter:  H_0, ra_dip, dec_dip and the H_0 amplitude of the dipole


################## Applying classes ##################
These .py use the last 3 file and make their application
_______________applySelection.py_______________
----------- dipole -----------
Apply all selection to a data frame
****Input****
Initial data frame with all simulated SNe Ia
****Output****
2 file if clustering method: 1 with selected SNe and the other with the centroids
1 file if HEALPix method: with selected SNe

_______________applyFit.py_______________
----------- FIT -----------
Make the fit and plot all interesting graphs 
****Input****
Data frame of selected SNe Ia
****Output****
A file with the result of the global fit and an other with data frame after the global fit, a file with the result of the patching fit and an other with data frame after the patching fit, write on a csv file the mean value and standard deviation for delta_H0 and if there is anisotropies write on a csv file the different parameters (dipole, fitted values...)


################## Jupyter notebooks ##################
Amp_1_cl.ipynb, Amp_1_fix.ipynb, Amp_3_cl.ipynb , Amp_3_fix.ipynb, Amp_5_cl.ipynb, Amp_5_fix.ipynb: Make the analysis with anisotropies for 3 differents amplitudes for 10 directions and 10 different simulation and for the 2 different method.
!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!! 
Be careful because you have to create the csv files that you are going to fill in, so you have to be careful not to relaunch a cell with a write at the risk of overwriting everything.

analysis.ipynb : Make the analysis for ten simulation without anisotropies

anisotropies_analysis.ipynb: Make the analysis for ten simulation with anisotropies

dipole fit.ipynb: Make the analysis/plot after all the analysis on different dipole

test_new_class.ipynb: A test notebook on a single simulation, I invite you to take it in hand to see what it does.