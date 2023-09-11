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
import iminuit


class Hubble_fit:
    """ Fitting Hubble diagram """
    def __init__(self, H_0, df_pickle):
        """
        Parameters
    ----------
    H_0: float
        Hubble constant
    df_pickle: Dataframe
        Dataframe comme from pickle file
        """
        self.H_0 = H_0
        self.df_pickle = df_pickle
        
    def Hubble_diagram(self, M_b, alpha, beta, mb, x1, c, z):
        """
        Makes the Hubble diagram
        
        Parameters
    ----------
    M_b: float
        Correction parameter of absolute magnitude
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
    mb: dataframe of float
        Apparente magnitude in B band
    x1: dataframe of float
        Stretch parameters
    c: dataframe of float
        Color parameters
    z: dataframe of float
        Redshifts
        
        Return
    ----------
    mu: float or dataframe
        Distance Modulus
        """
        # Distance modulus with corrections
        mu = mb - M_b + alpha * x1 - beta * c
        # Distance modulus without corrections
        mu_c = mb - M_b
        # Plot
        ax = seaborn.jointplot(z, mu, label = 'with correction')
        seaborn.scatterplot(x = z, y = mu_c, facecolors='none', edgecolor = 'red', linewidth = 1, alpha = 0.4, label = 'without correction', ax = ax.ax_joint)
        ax.set_axis_labels('Redshift', 'Distance Modulus');
        return mu
    
    def H_z(self, z, omega_m):
        """
        Compute 1/E(z)
        
        Parameters
    ----------
    z: float
        Redshift
    omega_m : float
        Matter density
        
        Return
    ----------
    1/E_z : float
        Invert Hubble parameters without Hubble constant
        """
        return 1 / (np.sqrt(((1 + z) ** 3 - 1) * omega_m + 1))
    
    def d_L(self, omega_m):
        """
        Compute d_L, Luminosity distance
        
        Parameters
    ----------
    omega_m : float
        Matter density
        
        Return
    ----------
    d_L: float, Series 
        Luminosity distance
        """
        L = []
        for i in range(len(self.df_pickle)):
            L.append(quad(self.H_z, 0, self.df_pickle['zcmb'][i], omega_m)[0]) # make the integral
        I = np.array(L)
        return (1 + self.df_pickle['zcmb']) * 2.99e5 / self.H_0 * I
    
    def mu_th(self, omega_m):
        """
        Compute mu_th, Distance modulus from lambda CDM model
        
        Parameters
    ----------
    omega_m : float
        Matter density
        
        Return
    ----------
    mu_th: float, Series 
        Distance modulus from lambda CDM model
        """
        return 5 * np.log10(self.d_L(omega_m)) + 25
    
    def mu_exp(self, M_b, alpha, beta):
        """
        Compute mu_exp, Distance modulus from simulations/cosmological parameters
        
        Parameters
    ----------
    M_b : float
        Absolute magnitude
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
        
        Return
    ----------
    mu_exp: float, Series 
        Distance modulus from simulations/cosmological parameters
        """
        return self.df_pickle['mb'] - M_b + alpha * self.df_pickle['x1'] - beta * self.df_pickle['c']
    
    def sig_i(self, alpha, beta, df):
        """
        Compute variance of Distance modulus from simulations/cosmological parameters
        
        Parameters
    ----------
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
    df: dataframe
        Contain covariance matrix
        
        Return
    ----------
    sig_i: float
        variance of Distance modulus from simulations/cosmological parameters
        """
        L_der = [1, alpha, -beta]
        sig_i = 0
        for i in range(3):
            for j in range(3):
                sig_i += L_der[i] * L_der[j] * df[i, j]
        return sig_i
   
    def sig_th(self, omega_m, z, sig_z):
        """
        Compute variance of Distance modulus Distance modulus from lambda CDM model
        
        Parameters
    ----------
    omega_m: float
        Matter density
    z: float
        Redshift
    sig_z: float
        z variance
        
        Return
    ----------
    sig_th: float, Series
        variance of Distance modulus Distance modulus from lambda CDM model
        """
        L = []
        for i in range(len(self.df_pickle)):
            L.append(quad(self.H_z, 0, self.df_pickle['zcmb'][i], omega_m)[0])
        return 5/np.log(10) * (1 / (1 + z) + self.H_z(z, omega_m) / L) * sig_z
    
    def chi2(self, M_b, alpha, beta, omega_m):
        """
        Compute chi square
        
        Parameters
    ----------
    M_b : float
        Absolute magnitude
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
    omega_m: float
        Matter density
        
        Return
    ----------
    chi2: float
        chi square
        """
        return ((self.mu_exp(M_b, alpha, beta) - self.mu_th(omega_m)) ** 2 / (self.df_pickle['sig_i'] + (self.df_pickle['sig_th_GF'])**2)).sum()
    
    
    
    def fit(self, M_b, alpha, beta, omega_m):
        """
        Make the fit on chi square
        
        Parameters
    ----------
    M_b : float
        Absolute magnitude
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
    omega_m: float
        Matter density
        
        Return
    ----------
    param: dataframe
        Fitted parameters, error and some other indication
    values: List
        Fitted values
    covariance: dataframe
        Covariance matrix with fitted parameters
        """        
        m = Minuit(self.chi2, M_b = M_b, alpha = alpha, beta = beta, omega_m = omega_m)
        #minimize(self.chi2, M_b, alpha, beta, omega_m)
        m.fixed[3] = True # fix a parameter
        m.migrad() #make minimization
        return m.params, m.values, m.covariance, iminuit.describe(self.chi2)
    
    
class Hubble_fit_patch:
    """ Fitting Hubble diagram per patch"""
    def __init__(self, H_0, df_pickle, N_patch):
        """
        Parameters
    ----------
    H_0: float
        Hubble constant
    df_pickle: Dataframe
        Dataframe comme from pickle file
        """
        self.H_0 = H_0
        self.df_pickle = df_pickle
        self.N_patch = N_patch
    
    def H_z(self, z):
        """
        Compute 1/E(z)
        
        Parameters
    ----------
    z: float
        Redshift
        
        Return
    ----------
    1/E_z : float
        Invert Hubble parameters without Hubble constant
        """
        return 1 / (np.sqrt(((1 + z) ** 3 - 1) * 0.3 + 1))
    
    def d_L(self, delta_H0):
        """
        Compute d_L, Luminosity distance
        
        Parameters
    ----------
    delta_H0 : float
        Hubble constant variations
        
        Return
    ----------
    d_L: float, Series 
        Luminosity distance
        """
        L = []
        for i in range(len(self.df_pickle)):
            L.append(quad(self.H_z, 0, self.df_pickle['zcmb'][i])[0]) # Make the integrals
        I = np.array(L)
        return (1 + self.df_pickle['zcmb']) * 2.99e5 / (self.H_0 + delta_H0) * I
    
    def mu_th(self, delta_H0):
        """
        Compute mu_th, Distance modulus from lambda CDM model
        
        Parameters
    ----------
    delta_H0: float
        Hubble constant variations
        
        Return
    ----------
    mu_th: float, Series 
        Distance modulus from lambda CDM model
        """
        return 5 * np.log10(self.d_L(delta_H0)) + 25

    
    def mu_exp(self, M_b, alpha, beta):
        """
        Compute mu_exp, Distance modulus from simulations/cosmological parameters
        
        Parameters
    ----------
    M_b : float
        Absolute magnitude
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
        
        Return
    ----------
    mu_exp: float, Series 
        Distance modulus from simulations/cosmological parameters
        """
        return self.df_pickle['mb'] - M_b + alpha * self.df_pickle['x1'] - beta * self.df_pickle['c']
    
    def sig_i(self, alpha, beta, df):
        """
        Compute variance of Distance modulus from simulations/cosmological parameters
        
        Parameters
    ----------
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
    df: dataframe
        Contain covariance matrix
        
        Return
    ----------
    sig_i: float
        variance of Distance modulus from simulations/cosmological parameters
        """        
        L_der = [1, alpha, -beta]
        sig_i = 0
        for i in range(3):
            for j in range(3):
                sig_i += L_der[i] * L_der[j] * df[i, j]
        return sig_i
   
    def sig_th(self, z, sig_z):
        """
        Compute variance of Distance modulus Distance modulus from lambda CDM model
        
        Parameters
    ----------
    omega_m: float
        Matter density
    z: float
        Redshift
    sig_z: float
        z variance
        
        Return
    ----------
    sig_th: float, Series
        variance of Distance modulus Distance modulus from lambda CDM model
        """
        L = []
        for i in range(len(self.df_pickle)):
            L.append(quad(self.H_z, 0, self.df_pickle['zcmb'][i])[0])
        return 5/np.log(10) * (1 / (1 + z) + self.H_z(z) / L) * sig_z
    
    def chi2(self, M_b, alpha, beta, delta_H0_0, delta_H0_1, delta_H0_2, delta_H0_3, delta_H0_4, delta_H0_5, delta_H0_6, delta_H0_7, delta_H0_8, delta_H0_9, delta_H0_10, delta_H0_11):#, delta_H0_12, delta_H0_13, delta_H0_14, delta_H0_15, delta_H0_16, delta_H0_17, delta_H0_18, delta_H0_19, delta_H0_20, delta_H0_21, delta_H0_22, delta_H0_23):#, delta_H0_24, delta_H0_25, delta_H0_26, delta_H0_27, delta_H0_28, delta_H0_29, delta_H0_30, delta_H0_31, delta_H0_32, delta_H0_33, delta_H0_34, delta_H0_35, delta_H0_36, delta_H0_37, delta_H0_38, delta_H0_39):
        """
        Compute chi square
        
        Parameters
    ----------
    M_b : float
        Absolute magnitude
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
    delta_H0_0: float
        Hubble constant variation for cluster nbr 0
    delta_H0_1: float
        Hubble constant variation for cluster nbr 1 
    delta_H0_2: float
        Hubble constant variation for cluster nbr 2
    delta_H0_3: float
        Hubble constant variation for cluster nbr 3 
    delta_H0_4: float
        Hubble constant variation for cluster nbr 4
    delta_H0_5: float
        Hubble constant variation for cluster nbr 5 
    delta_H0_6: float
        Hubble constant variation for cluster nbr 6
    delta_H0_7: float
        Hubble constant variation for cluster nbr 7 
    delta_H0_8: float
        Hubble constant variation for cluster nbr 8 
    delta_H0_9: float
        Hubble constant variation for cluster nbr 9 
    delta_H0_10: float
        Hubble constant variation for cluster nbr 10
        
        Return
    ----------
    chi2: float
        chi square
        """
        if self.N_patch == 11:
            delta_H0 = [delta_H0_0, delta_H0_1, delta_H0_2, delta_H0_3, delta_H0_4, delta_H0_5, delta_H0_6, delta_H0_7, delta_H0_8, delta_H0_9, delta_H0_10]
        if self.N_patch == 12:
            delta_H0 = [delta_H0_0, delta_H0_1, delta_H0_2, delta_H0_3, delta_H0_4, delta_H0_5, delta_H0_6, delta_H0_7, delta_H0_8, delta_H0_9, delta_H0_10, delta_H0_11]#, delta_H0_12, delta_H0_13, delta_H0_14, delta_H0_15, delta_H0_16, delta_H0_17, delta_H0_18, delta_H0_19, delta_H0_20, delta_H0_21, delta_H0_22, delta_H0_23]#, delta_H0_24, delta_H0_25, delta_H0_26, delta_H0_27, delta_H0_28, delta_H0_29, delta_H0_30, delta_H0_31, delta_H0_32, delta_H0_33, delta_H0_34, delta_H0_35, delta_H0_36, delta_H0_37, delta_H0_38, delta_H0_39]
        chi_tot = 0
        for i in range(len(delta_H0)):
            mask = self.df_pickle['ncluster'] == i
            chi_tot += ((self.mu_exp(M_b, alpha, beta)[mask].reset_index()[0] - self.mu_th(delta_H0[i])[mask].reset_index()['zcmb']) ** 2 / (self.df_pickle[mask].reset_index()['sig_i'] +  (self.df_pickle[mask].reset_index()['sig_th'])**2)).sum()#(self.df_pickle['sig_i'] + (self.df_pickle['sig_th_GF'])**2)
            #print(delta_H0_0, chi_tot)
        return chi_tot
    
    
    def fit(self, M_b, alpha, beta, delta_H0_0, delta_H0_1, delta_H0_2, delta_H0_3, delta_H0_4, delta_H0_5, delta_H0_6, delta_H0_7, delta_H0_8, delta_H0_9, delta_H0_10, delta_H0_11):#, delta_H0_12, delta_H0_13, delta_H0_14, delta_H0_15, delta_H0_16, delta_H0_17, delta_H0_18, delta_H0_19, delta_H0_20, delta_H0_21, delta_H0_22, delta_H0_23):#, delta_H0_24, delta_H0_25, delta_H0_26, delta_H0_27, delta_H0_28, delta_H0_29, delta_H0_30, delta_H0_31, delta_H0_32, delta_H0_33, delta_H0_34, delta_H0_35, delta_H0_36, delta_H0_37, delta_H0_38, delta_H0_39):
        """
        Make the fit on chi square
        
        Parameters
    ----------
    M_b : float
        Absolute magnitude
    alpha: float
        Stretch coefficient
    beta: float
        Color coefficient
    delta_H0_0: float
        Hubble constant variation for cluster nbr 0
    delta_H0_1: float
        Hubble constant variation for cluster nbr 1 
    delta_H0_2: float
        Hubble constant variation for cluster nbr 2
    delta_H0_3: float
        Hubble constant variation for cluster nbr 3 
    delta_H0_4: float
        Hubble constant variation for cluster nbr 4
    delta_H0_5: float
        Hubble constant variation for cluster nbr 5 
    delta_H0_6: float
        Hubble constant variation for cluster nbr 6
    delta_H0_7: float
        Hubble constant variation for cluster nbr 7 
    delta_H0_8: float
        Hubble constant variation for cluster nbr 8 
    delta_H0_9: float
        Hubble constant variation for cluster nbr 9 
    delta_H0_10: float
        Hubble constant variation for cluster nbr 10
        
        Return
    ----------
    param: dataframe
        Fitted parameters, error and some other indication
    values: List
        Fitted values
    covariance: dataframe
        Covariance matrix with fitted parameters
        """ 
        if self.N_patch == 11:
            m = Minuit(self.chi2, M_b = M_b, alpha = alpha, beta = beta, delta_H0_0 = delta_H0_0, delta_H0_1 = delta_H0_1, delta_H0_2 = delta_H0_2, delta_H0_3 = delta_H0_3, delta_H0_4 = delta_H0_4, delta_H0_5 = delta_H0_5, delta_H0_6 = delta_H0_6, delta_H0_7 = delta_H0_7, delta_H0_8 = delta_H0_8, delta_H0_9 = delta_H0_9, delta_H0_10 = delta_H0_10, delta_H0_11 = delta_H0_11)
            m.fixed[14] = True
        if self.N_patch == 12:
            m = Minuit(self.chi2, M_b = M_b, alpha = alpha, beta = beta, delta_H0_0 = delta_H0_0, delta_H0_1 = delta_H0_1, delta_H0_2 = delta_H0_2, delta_H0_3 = delta_H0_3, delta_H0_4 = delta_H0_4, delta_H0_5 = delta_H0_5, delta_H0_6 = delta_H0_6, delta_H0_7 = delta_H0_7, delta_H0_8 = delta_H0_8, delta_H0_9 = delta_H0_9, delta_H0_10 = delta_H0_10, delta_H0_11 = delta_H0_11)#, delta_H0_12 = delta_H0_12, delta_H0_13 = delta_H0_13, delta_H0_14 = delta_H0_14, delta_H0_15 = delta_H0_15, delta_H0_16 = delta_H0_16, delta_H0_17 = delta_H0_17, delta_H0_18 = delta_H0_18, delta_H0_19 = delta_H0_19, delta_H0_20 = delta_H0_20, delta_H0_21 = delta_H0_21, delta_H0_22 = delta_H0_22, delta_H0_23 = delta_H0_23)#, delta_H0_24 = delta_H0_24, delta_H0_25 = delta_H0_25, delta_H0_26 = delta_H0_26, delta_H0_27 = delta_H0_27, delta_H0_28 = delta_H0_28, delta_H0_29 = delta_H0_29, delta_H0_30 = delta_H0_30, delta_H0_31 = delta_H0_31, delta_H0_32 = delta_H0_32, delta_H0_33 = delta_H0_33, delta_H0_34 = delta_H0_34, delta_H0_35 = delta_H0_35, delta_H0_36 = delta_H0_36, delta_H0_37 = delta_H0_37, delta_H0_38 = delta_H0_38, delta_H0_39 = delta_H0_39)
        m.fixed[0] = True # fix a parameter
        m.migrad() # make the minimization
        return m.params, m.values, m.covariance, iminuit.describe(self.chi2)
    
    
class dipole:
    """ Dipole fit on the fitted delta_H_0 """
    def __init__(self, H_0_exp, s_H_0, ra, dec):
        """
        Parameters
    ----------
    H_0_exp: float
        Hubble constant from patches
    s_H_0: float
        errors on H_0
    ra: float
        coordinate (ra) in radian
    dec: float
        coordinate (dec) in radian
        """
        self.H_0_exp = H_0_exp
        self.s_H_0 = s_H_0
        self.ra = ra
        self.dec = dec
        
    def H0_prime_th(self, H_0, ra_dip, dec_dip, delta_H0):
        """
        Calculate H_0'_th
        
        Parameters
    ----------
    H_0: float
        Hubble constant 
    ra_dip: float
        input dipole (ra) coordinate 
    dec_dip: float
        input dipole coordinate (dec)
    delta_H0: float
        H_0 variations
        """
        x_d = np.sin(np.pi/2 - dec_dip) * np.cos(ra_dip)
        x = np.sin(np.pi/2 - self.dec) * np.cos(self.ra)
        y_d = np.sin(np.pi/2 - dec_dip) * np.sin(ra_dip)
        y = np.sin(np.pi/2 - self.dec) * np.sin(self.ra)
        z_d = np.cos(np.pi/2 - dec_dip)
        z = np.cos(np.pi/2 - self.dec)
        return H_0 + delta_H0 * (x_d * x + y_d * y + z_d * z)
    
    def chi2(self, H_0, ra_dip, dec_dip, delta_H0):
        """
        Make the chi square
        
        Parameters
    ----------
    H_0: float
        Hubble constant 
    ra_dip: float
        input dipole (ra) coordinate 
    dec_dip: float
        input dipole coordinate (dec)
    delta_H0: float
        H_0 variations
        """
        return (((self.H_0_exp - self.H0_prime_th(H_0, ra_dip, dec_dip, delta_H0))/self.s_H_0)**2).sum()
    
    def dipole_fit(self, H_0, ra_dip, dec_dip, delta_H0):
        """ 
        Make the minimisation (fit)
        Parameters
    ----------
    H_0: float
        Hubble constant 
    ra_dip: float
        input dipole (ra) coordinate 
    dec_dip: float
        input dipole coordinate (dec)
    delta_H0: float
        H_0 variations
        """
        m = Minuit(self.chi2, H_0 = H_0, ra_dip = ra_dip, dec_dip = dec_dip, delta_H0 = delta_H0)
        m.migrad()
        return m.params, m.values, m.covariance
    
    