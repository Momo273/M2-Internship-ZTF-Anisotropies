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


class Criteria_selection:
    """ Selects Supernovae according to specific criteria """
    def __init__(self, df):#, path):
        """  
        Parameters
    ----------
    df : panda dataframe
            dataframe to work on it
    """
        self.flag_list = []
        self.df = df
        #self.path = path
        
    def Sigmoid(self, x, a, b, c):
        """
        A Sigmoid function
        
        Parameters
    ----------
    x : float
            our variable
    a, b, c : float
            parameters of sigmoid function
        
        Return
    ----------
    A float value comes from Sigmoid function
        """
        return a / (1 + np.exp(b *(x-c)))
    
    def fit_Sigmoid(self, df_completeness):
        """
        Fit A Sigmoid function
        
        Parameters
    ----------
    df_completeness : pandas dataframe 
             A data frame where completeness, magnitudes and efficiencies are stored
        
        Return
    ----------
    popt : 3 float 
            Fitted parameters
        """
        popt, pcov = optimize.curve_fit(self.Sigmoid, df_completeness['Mag-'][:35], df_completeness['Completeness'][:35])
        return popt
    
    
    def gal_select(self, name_var):
        """
        Take into account galaxy extinction with the criterion selection from BTS modulus(b)<7°
        
        Parameters
    ----------
    name_var: str
            Name of the variable
            
        Return
    ----------
    lon: an array or a column from a dataframe
            Represents the longitude
    lat: an array or a column from a dataframe
            Represents the latitude
    z: an array or a column from a dataframe
            Represents the redshift
    For all SN who are not in a galactic plane range
        """
        data = SkyCoord(self.df['ra'], self.df['dec'], unit='deg', frame='icrs') # COORDINATES 
        l, b = data.galactic.l.value, data.galactic.b.value # Change equatorial coodinates into galactic coordinates
        mask_b = (b > -7) & (b < 7) # mask for the galactic extinction
        return self.df['ra'][~mask_b], self.df['dec'][~mask_b], self.df[name_var][~mask_b]
    
    def selection(self, df_completeness, n, name_var, name_ZTF):
        """
        Criteria selection, spectroscopic selection
        
        Parameters
    ----------
    df_completeness : pandas dataframe 
             A data frame where completeness, magnitudes and efficiencies are stored
    name_var: str
            Name of the variable
    name_ZTF: str
            Name of the ZTF simulated SN
            
        Return
    ----------
    lon: an array or a column from a dataframe
            Represents the longitude
    lat: an array or a column from a dataframe
            Represents the latitude
    z: an array or a column from a dataframe
            Represents the redshift
    But here the dataframe objects have been selctioned
        """
        #np.random.seed(1200)
        lon, lat, z = self.gal_select(name_var) # We make first the galactic selection
        self.df['flag'] = '' # define a new column
        popt = self.fit_Sigmoid(df_completeness) # we take the fit parameters from Sigmoid
        for i in range(len(self.df)):
            selct = np.random.uniform() # take a random number uniformly
            df_lc = pd.read_csv("lc_spec{}/{}.csv".format(n, name_ZTF[i])) # read LC file
            mask_zp = (df_lc['flux'] == np.max(df_lc['flux'])) # zero-point mask
            mag = - 2.5 * np.log10(np.max(df_lc['flux'])) + float(df_lc[mask_zp]['zp']) # magnitude
            epsilon = self.Sigmoid(mag, popt[0], popt[1], popt[2]) # magnitude with sigmoid
            if epsilon > selct:
                self.df.loc[i, 'flag'] = 0
            else:
                self.df.loc[i, 'flag'] = 1
        mask = (self.df['flag'] == 0) # mask for the selection
        return lon[mask], lat[mask], z[mask]
    
    
class Good_sampling:
    """ Selection the good number of point in light curve to know if we can take this SN or not """
    def __init__(self, n, filename, df_pickle):
        """
        Parameters
    ----------
    n: an interger
        simulation number
    filename: str
            Name of SN
    df_pickle: pandas dataframe
            Dataframe for Salt2 fit
        """
        self.df_pickle = df_pickle # the pickle file
        self.filename = filename
        self.df = pd.read_csv('sim3000{}/lc_spec/{}.csv'.format(n, filename))
        self.df['sig'] = self.df['flux'] / self.df['fluxerr']
        
   
    def first_selection(self):
        """
        Make the first cosmological selection 
        
        Return
    ----------
    Number of point that correspond at this selection on the light curve
    mask_sig
    t_0
        """
        # Define t_0 
        t_0 = self.df_pickle[(self.df_pickle['name'] == self.filename)]['t0'].iloc[0] 
        # t_0 range
        mask_t0 = (self.df['time'] < t_0 + 30) & (self.df['time'] > t_0 - 15)
        # 5 sigma
        mask_sig = self.df['sig'] >= 5
        #return len(self.df[mask_t0][mask_sig]), mask_sig, t_0
        return len(self.df.loc[mask_t0 & mask_sig]), mask_sig, t_0    
    
    def second_selection(self):
        """
        Make the second selection on cosmological criteria
        """
        N_point, mask_sig, t_0 = self.first_selection()
        #self.df_pickle['flag_cosmo'] = ''
        if (N_point >= 7):
            # before t_0
            mask_inf_t0 = self.df['time'] < t_0
            # after t_0
            mask_sup_t0 = self.df['time'] > t_0
            # Mask on band
            mask_band_r = (self.df['band'][mask_sig] == 'ztfr')
            mask_band_g = (self.df['band'][mask_sig] == 'ztfg')
            # all different selection to make
            requirement = (len(self.df[mask_inf_t0 & mask_band_r]) >= 1) & (len(self.df[mask_inf_t0 & mask_band_g]) >= 1) & (len(self.df[mask_sup_t0 & mask_band_r]) >= 1) & (len(self.df[mask_sup_t0 & mask_band_g]) >= 1)
            if requirement == True:
                #self.df_pickle['flag_cosmo'][(self.df_pickle['name'] == self.filename)] = 0
                self.df_pickle.loc[self.df_pickle['name'] == self.filename, 'flag_cosmo'] = 0
            else:
                #self.df_pickle['flag_cosmo'][(self.df_pickle['name'] == self.filename)] = 1
                self.df_pickle.loc[self.df_pickle['name'] == self.filename, 'flag_cosmo'] = 1
        else:
            #self.df_pickle['flag_cosmo'][(self.df_pickle['name'] == self.filename)] = 1
            self.df_pickle.loc[self.df_pickle['name'] == self.filename, 'flag_cosmo'] = 1

            
            
class salt_2_selection:
    """ Selection on Salt2 parameters """
    def __init__(self, df_pickle):
        """
        Parameters
    ----------
    df_pickle: pandas dataframe
            Dataframe for Salt2 fit
        """
        self.df_pickle = df_pickle
        
    def salt_2_criteria(self):
        """ Make the selection for salt2 parameters """
        # color mask
        mask_color = (self.df_pickle['c'] >= -0.3) & (self.df_pickle['c'] <= 0.8)
        # stretch mask
        mask_x1 = np.abs(self.df_pickle['x1']) <= 4
        mask_sigX1 = self.df_pickle['sig_x1'] < 1
        #return self.df_pickle[mask_color][mask_x1][mask_sigX1]
        return self.df_pickle.loc[mask_color & mask_x1 & mask_sigX1]

        