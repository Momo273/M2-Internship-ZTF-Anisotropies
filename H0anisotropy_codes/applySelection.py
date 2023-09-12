import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pandas as pd
from astropy import units as u
import sklearn.cluster as sclust
import sklearn.metrics as smet
import scipy.optimize as optimize
from astropy.coordinates import SkyCoord
import seaborn
from scipy.integrate import quad
from scipy.integrate import romberg 
from iminuit import Minuit
from iminuit import minimize
from SN_distribution import SkyMap
from SN_distribution import Patching_Map
from Selections import Criteria_selection
from Selections import Good_sampling
from Selections import salt_2_selection
from Hubble import Hubble_fit
from Hubble import Hubble_fit_patch



class select_end:
    """ Apply all selection to a dataframe """
    def __init__(self, df):
        """
        Parameters
    ----------
    df: Dataframe
        Dataframe comme from pickle file
        """
        self.df = df
        
    def allSelct(self, n, method, N_patch, N_resol, H0):
        """
        Apply all selection to a dataframe and save it into a pickle file
        Also for centers that come from clustering method (in csv file)
        
        Parameters
    ----------
    n: an interger
        simulation number
    method: a string
        could be 'clustering' for the clustering method to patch the map, or 'fixe' the fixe one that use HEALPix
    N_patch: an integer
        number of patch, !!!!!!ATTENTION!!!!!!!!: for fixe method is the degree of pixelisation
    N_resol: an integer
        the degree of pixelisation, the resolution
    H0: a float
        the input value of the Hubble constant in km s^-1 Mpc^-1
        """
        
        # Make the cosmo flag in the sampling
        self.df['flag_cosmo'] = ''
        for i in range(len(self.df)):
            samp = Good_sampling(n, '{}'.format(self.df['name'][i]), self.df)
            samp.second_selection()
            
        # call of the class Criteria selection    
        crit = Criteria_selection(self.df)
        lon, lat, z = crit.gal_select('zcmb')
        
        # take into account the selections
        mask_flags = (self.df['flag_cosmo'] == 0)
        mask_ = np.in1d(self.df['ra'][mask_flags], lon)
        mask_av = self.df['a_v'][mask_flags] < 1
        # Make the Salt2 selection on cosmological parameters
        salt_2_selec = salt_2_selection(self.df[mask_flags][mask_ & mask_av])
        final_df = salt_2_selec.salt_2_criteria()
        
        # Call of the class Patching Map
        patch = Patching_Map(nside = 1024, lon = final_df['ra'], lat = final_df['dec'], var = final_df['zcmb'], var_name = 'redshift z', title = 'SNe Ia from simulated & selected ZTF survey')
        # Clustering
        if method == 'clustering':
            centers = patch.patch_clustering_survey(N_patch, final_df)
        if method == 'fixe':
            centers = patch.pixelisation(N_resol, final_df)
        
        # Reorganise data in ascending order of z
        final_df = final_df.sort_values(by = 'zcmb').reset_index()
        # Call of the Hubble fit patch class
        hubble_patch = Hubble_fit_patch(H0, final_df, N_patch)
        alpha, beta = 0.154, 3.69
        # Errors
        final_df['sig_i'] = ''
        for i in range(len(final_df)):
            final_df.loc[i, 'sig_i'] = hubble_patch.sig_i(alpha, beta, final_df['cov_mb'][i])
        
        # Save the data in a pickle file and centers in a csv file (for clustering method)
        if method == 'clustering':
            final_df.to_pickle('selection_data_3years(3000{})_clust.pkl'.format(n))
            np.savetxt('centers_clust(3000{}).csv'.format(n), centers, delimiter = ',')
        if method == 'fixe':
            final_df.to_pickle('selection_data_3years(3000{})_fix.pkl'.format(n))
         
        
                                      
        






























