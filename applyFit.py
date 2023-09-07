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
from matplotlib.gridspec import GridSpec
import matplotlib
import csv



class FIT:
    """ Make the fit and plot all interesting graphs """
    
    def __init__(self, df):
        """
        Parameters
    ----------
    df: Dataframe
        Dataframe of selected SNe Ia comme from pickle file
        """
        self.df = df
    
    def map_delta_H0(self, centers, val, N_clust, coord, method, N_resol):
        """
        Plot the variation map of delta H_0
        
        Parameters
    ----------
    centers: DataFrame
        DataFrame with centers from clustering method
    val: an array
        values that comes from the fit 
    N_clust: an integer
        number of patch
    coord: a list of string
        like: ['C'], ['C', 'G'] and ['C', 'E']
    method: a string
        could be 'clustering' for the clustering method to patch the map, or 'fixe' the fixe one that use HEALPix
    N_resol: an integer
        the degree of pixelisation, the resolution
        """
        if method == 'clustering':
            # number of pixel / pixels indices
            npix = np.arange(hp.nside2npix(64))
            # Turn npix in angles : theta, phi
            theta , phi = hp.pix2ang(64, npix)
            # Put in an array (make the transposition)
            ang = np.array([theta.tolist(), phi.tolist()]).T
            # Coordinates
            data = SkyCoord(np.rad2deg(phi), np.rad2deg(np.pi/2 - theta), unit='deg', frame='icrs')
            # Change equatorial coordinate into galactic coordinates
            l, b = data.galactic.l.value, data.galactic.b.value
            # Make the mask on the galactic extinction
            mask_b = (b > -7) & (b < 7)
            mask_clustering = (ang[:,0] < np.radians(120))
            # KMeans on pixel with 1 iteration
            res = sclust.KMeans(N_clust, init = (np.array(centers)), max_iter = 1).fit(ang[~mask_b & mask_clustering])
            # Fill pixel of zeros
            m_new = np.zeros(hp.nside2npix(64)) 
            # We says that all pixels is nan
            m_masked = hp.ma(m_new)
            mask = m_new
            m_masked.mask = np.logical_not(mask)
            # SNe pixel indices
            pixel_indices = hp.ang2pix(64, np.array(np.pi/2 - np.radians(self.df['dec'])), np.array(np.radians(self.df['ra'])))
            # Galactic plan
            plan_lon = np.linspace(0, 360, 10000)
            plan_lat = np.zeros(10000)
            gal = SkyCoord(plan_lon, plan_lat, frame = 'galactic', unit = 'degree')
            ra_gal, dec_gal = gal.icrs.ra.value, gal.icrs.dec.value
            theta_gal = np.radians(90. - dec_gal)
            phi_gal = np.radians(ra_gal)
            # Light on all patches with their Huubble constant variation value
            for i in range(N_clust):
                mask_clustering_bis = (res.labels_ == i)
                new_ipix = hp.ang2pix(64, theta[~mask_b & mask_clustering][mask_clustering_bis], phi[~mask_b & mask_clustering][mask_clustering_bis])
                idx = np.in1d(pixel_indices, new_ipix)
                mask = (idx == True)
                m_masked[new_ipix] = val[i + 3]

            hp.mollview(m_masked, title = 'Variations of $H_0$', unit = '$\delta H_0$', coord = coord)
            hp.projplot(theta_gal, phi_gal, color = 'red', label = 'Galactic', coord = coord)
            #hp.projscatter(np.pi/2 - np.radians(df['dec'][df['ncluster'] == 10]), np.radians(df['ra'][df['ncluster'] == 10]))
            hp.projscatter(res.cluster_centers_[:,0], res.cluster_centers_[:,1], marker = 'x', color = 'r', coord = coord)
            hp.graticule()
            # Cluster number projection
            for i in range(N_clust):
                hp.projtext(res.cluster_centers_[:,0][i] + np.radians(6), res.cluster_centers_[:,1][i] + np.radians(7), s = 'n°{}'.format(i), color = 'red', coord = coord)

            plt.legend()
        
        if method == 'fixe':
            # number of pixel / pixels indices
            npix = hp.nest2ring(128,np.arange(hp.nside2npix(128)))
            # Turn npix in angles : theta, phi
            theta , phi = hp.pix2ang(128, npix)
            # Put in an array (make the transposition)
            ang = np.array([theta.tolist(), phi.tolist()]).T
            # Coordinates
            data = SkyCoord(np.rad2deg(phi), np.rad2deg(np.pi/2 - theta), unit='deg', frame='icrs')
            # Change equatorial coordinate into galactic coordinates
            l, b = data.galactic.l.value, data.galactic.b.value
            # Make the mask on the galactic extinction
            mask_b = (b > -7) & (b < 7)
            mask_clustering = (ang[:,0] < np.radians(120)) 
            
            m = np.arange(hp.nside2npix(128))
            ring = hp.nest2ring(128, m)
            m_mask = np.arange(hp.nside2npix(N_resol))
            ang = hp.pix2ang(N_resol, m_mask)
            m_pix = np.zeros(hp.nside2npix(128))
            m_pix_ = hp.nest2ring(128, np.arange(hp.nside2npix(128)))
            L = []
            maskk = []
            m = 0
            for i in range(N_clust):
                L.append(m_pix_[0 + m :(len(~mask_b)/N_clust)+m])
                maskk.append([~mask_b[0 + m:(len(~mask_b)/N_clust)+m] & mask_clustering[0 + m :(len(~mask_b)/N_clust)+m]])
                m += (len(~mask_b)/N_clust)
            m = hp.ma(m_pix)
            mask = m_pix
            m.mask = np.logical_not(mask)
            plan_lon = np.linspace(0, 360, 10000)
            plan_lat = np.zeros(10000)
            gal = SkyCoord(plan_lon, plan_lat, frame = 'galactic', unit = 'degree')
            ra_gal, dec_gal = gal.icrs.ra.value, gal.icrs.dec.value
            theta_gal = np.radians(90. - dec_gal)
            phi_gal = np.radians(ra_gal)
            for i in range(N_clust):
                m[L[i][maskk[i]]] = val[3 + i]
            hp.mollview(m, title = 'Variations of $H_0$', unit = '$\delta H_0$', coord = coord)
            hp.projplot(theta_gal, phi_gal, color = 'red', label = 'Galactic', coord = coord)
            hp.projscatter(ang[0], ang[1], marker = 'x', color = 'r', coord = coord)
            hp.graticule()
            plt.legend()
    
    def plt_H0(self, cov, H0, val, N_clust, val_global, color):
        """
        Plot the variation H_0 per patch
        
        Parameters
    ----------
    val: an array
        values that comes from the fit 
    N_clust: an integer
        number of patch
    val_global: an array
        values that comes from the global fit 
    color: a list of string
        different color, its length is the same than the number of patch
        """
        # Mean value of the Hubble constant
        M_b = val_global[0]
        Mean_H0 = ((H0 + val[3:]).sum())/N_clust
        hubble_patch = Hubble_fit_patch(H0, self.df)
        fig = plt.figure(figsize=[10,8])
        gs = GridSpec(4,3, hspace=0, wspace=0)
        ax_joint = fig.add_subplot(gs[0:3,0:2]) # Hubble constant values
        ax_joint_x = fig.add_subplot(gs[3:4,0:2]) # Mean residue values 
        x_ticks_labels = []
        for i in range(N_clust):
            x_ticks_labels.append('cluster n°{}'.format(i))
        for j in range(N_clust):
            ax_joint.errorbar(x_ticks_labels[j], 70 + val[3+j], yerr = cov[3+j], marker='.', color=color[j], linestyle="None", ecolor=color[j], capsize = 2, zorder = 1, elinewidth = 0.5,label='mu_exp')
        ax_joint.text(0, 70.9, s = '$\delta H_0$ mean value = {0: .3f}'.format(np.array(val[3:]).mean()))
        ax_joint.text(0, 70.8, s = '$\delta H_0$ std value = {0: .2f}'.format(np.array(val[3:]).std()))
        ax_joint.axhline(y = Mean_H0, c='red', linestyle="--")
        ax_joint.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
        ax_joint.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
        ax_joint.set_yticks(np.arange(69, 71, 0.05), minor = True)
        #ax_joint.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(33, 41, 1)))
        ax_joint.set_xlabel('Clusters')
        ax_joint.set_ylabel(r'$H_0$')  
        #ax_joint.set_xscale('log')
        ax_joint.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)
        ax_joint.axes.get_xaxis().set_visible(False)
        plt.setp(ax_joint.get_yticklabels());
        #ax_joint.legend(loc='best', shadow=True, fontsize='x-large')
        
        alpha , beta = val[1], val[2]
        ax_joint_x.axhline(c='red', linestyle="--")
        a = ax_joint_x.errorbar(x_ticks_labels[0], (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == 0] - hubble_patch.mu_th(val[3])[self.df['ncluster'] == 0]).mean(), yerr = (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == 0] - hubble_patch.mu_th(val[3])[self.df['ncluster'] == 0]).std()/np.sqrt(len(hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == 0])), marker='.', color="blue", linestyle="None", ecolor="blue", capsize = 2, zorder = 1, elinewidth = 0.5,label='mu_exp')
        for k in range(1, N_clust):
            ax_joint_x.errorbar(x_ticks_labels[k], (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == k] - hubble_patch.mu_th(val[3+k])[self.df['ncluster'] == k]).mean(), yerr = (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == k] - hubble_patch.mu_th(val[3+k])[self.df['ncluster'] == k]).std()/np.sqrt(len(hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == k])), marker='.', color=color[k], linestyle="None", ecolor=color[k], capsize = 2, zorder = 1, elinewidth = 0.5,label='mu_exp')
        

        b = ax_joint_x.errorbar(0.2, (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == 0] - hubble_patch.mu_th(0)[self.df['ncluster'] == 0]).mean(), yerr = (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == 0] - hubble_patch.mu_th(0)[self.df['ncluster'] == 0]).std()/np.sqrt(len(hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == 0])), marker='^', color="blue", linestyle="None", ecolor="blue", capsize = 2, zorder = 1, elinewidth = 0.5,label='mu_exp', markersize = 5)
        for l in range(1, N_clust):
            ax_joint_x.errorbar(l+0.2, (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == l] - hubble_patch.mu_th(0)[self.df['ncluster'] == l]).mean(), yerr = (hubble_patch.mu_exp(M_b, val[1], val[2])[self.df['ncluster'] == l] - hubble_patch.mu_th(0)[self.df['ncluster'] == l]).std()/np.sqrt(len(hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == l])), marker='^', color=color[l], linestyle="None", ecolor=color[l], capsize = 2, zorder = 1, elinewidth = 0.5,label='mu_exp', markersize = 5)
        
        alpha , beta = val_global[1], val_global[2]
        c = ax_joint_x.errorbar(0.4, (hubble_patch.mu_exp(M_b, alpha , beta)[self.df['ncluster'] == 0] - hubble_patch.mu_th(0)[self.df['ncluster'] == 0]).mean(), yerr = (hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == 0] - hubble_patch.mu_th(0)[self.df['ncluster'] == 0]).std()/np.sqrt(len(hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == 0])), marker='+', color="blue", linestyle="None", ecolor="blue", capsize = 2, zorder = 1, elinewidth = 0.5,label='mu_exp')
        for m in range(1, N_clust):
            ax_joint_x.errorbar(m + 0.4, (hubble_patch.mu_exp(M_b, alpha , beta)[self.df['ncluster'] == m] - hubble_patch.mu_th(0)[self.df['ncluster'] == m]).mean(), yerr = (hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == m] - hubble_patch.mu_th(0)[self.df['ncluster'] == m]).std()/np.sqrt(len(hubble_patch.mu_exp(M_b, alpha, beta)[self.df['ncluster'] == m])), marker='+', color=color[m], linestyle="None", ecolor=color[m], capsize = 2, zorder = 1, elinewidth = 0.5,label='mu_exp')
        
        ax_joint.legend([a, b, c], ['Residual mean value', 'Residual mean value with $\delta H_0$=0', 'Residual mean value with $\delta H_0$=0 and global fit values'], loc = 'lower left', labelcolor='black')
        ax_joint_x.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
        ax_joint_x.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
        ax_joint_x.set_yticks(np.arange(-0.04, 0.04, 0.01), minor = True)
        ax_joint_x.set_xlabel('Clusters')
        ax_joint_x.set_ylabel(r'Mean value of $\mu - \mu_{\Lambda CDM}$')  
        #ax_joint_x.set_xscale('log')
        ax_joint_x.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)
        #ax_joint_x.set_yticks(np.arange(-0.07, 0.07).round(2).tolist(), minor = True)
        #ax_joint_x.set_yticklabels(np.arange(-0.07, 0.07).round(2).tolist(), minor = True)
        plt.setp(ax_joint_x.get_xticklabels(), fontsize=12)
    
    def HD_per_patch(self, df, H0, val_global, N_clust, color):
        """
        Plot the Hubble Diagram per patches
        
        Parameters
    ----------
    df: a DataFrame
        dataFrame saved by the class analysis (for global fit or for patch fit)
    val_global: an array
        values that comes from the global fit 
    N_clust: an integer
        number of patch
    color: a list of string
        different color, its length is the same than the number of patch
        """
        M_b, alpha , beta = val_global[0], val_global[1], val_global[2]
        hubble_patch = Hubble_fit_patch(H0, self.df)
        sig_res = np.sqrt(list(df['sig_mu'] + (hubble_patch.sig_th(df['zcmb'], 1/2 * 1e-3))**2))
        for i in range(N_clust):
            fig = plt.figure(figsize=[10,8])
            gs = GridSpec(4,3, hspace=0, wspace=0)
            ax_joint = fig.add_subplot(gs[0:3,0:2])
            ax_joint_x = fig.add_subplot(gs[3:4,0:2])
            ax_marg_y = fig.add_subplot(gs[3:4,2])
            ax_joint.plot(df['zcmb'], hubble_patch.mu_th(0), c="red", label="$\Lambda$CDM fit")
            ax_joint.errorbar(df['zcmb'][df['ncluster'] == i], hubble_patch.mu_exp(M_b, alpha, beta)[df['ncluster'] == i], marker='.', yerr = sig_res[df['ncluster'] == i], color=color[i], linestyle="None", ecolor="steelblue", capsize = 2, zorder = 1, elinewidth = 0.5, label='mu_exp')
            ax_joint.set_xlim(0, 0.16)
            ax_joint.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint.set_yticks(np.arange(31.1, 41, 0.1), minor = True)
            ax_joint.set_xticks(np.arange(0, 0.16, 0.01), minor = True)
            plt.title('Cluster n°{}'.format(i))
            ax_joint.axes.get_xaxis().set_visible(False)
            plt.setp(ax_joint.get_yticklabels(), fontsize=12);
            ax_joint_x.axhline(c='red', linestyle="--")
            ax_joint_x.errorbar(df['zcmb'][df['ncluster'] == i], hubble_patch.mu_exp(M_b, alpha, beta)[df['ncluster'] == i] - hubble_patch.mu_th(0)[df['ncluster'] == i], marker='.', yerr = sig_res[df['ncluster'] == i], color=color[i], linestyle="None", ecolor="steelblue", capsize = 2, zorder = 1, elinewidth = 0.5, label='mu_exp')
            ax_joint_x.set_xlim(0, 0.16)
            ax_joint_x.set_ylim(-0.5, 0.5)
            plt.setp(ax_joint_x.get_xticklabels(), fontsize=12)
            plt.setp(ax_joint_x.get_yticklabels(), fontsize=12)
            ax_joint_x.set_xlabel('Redshift z')
            ax_joint_x.set_ylabel(r'$\mu - \mu_{\Lambda CDM}$')
            ax_joint_x.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint_x.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint_x.set_yticks(np.arange(-0.5, 0.5, 0.1), minor = True)
            ax_joint_x.set_xticks(np.arange(0, 0.16, 0.01), minor = True)
            ax_marg_y.hist(hubble_patch.mu_exp(M_b, alpha, beta)[df['ncluster'] == i] - hubble_patch.mu_th(0)[df['ncluster'] == i], orientation='horizontal', color=color[i])
            ax_marg_y.set_ylim(-0.5, 0.5)
            ax_marg_y.set_axis_off()
    
    def plt_standardization(self, df):
        """
        Plot Hubble diagram with and without standization
        
        Parameter
    ----------
    df: a DataFrame
        dataFrame with salt2 cosmological parameters
   
        """
        mb = self.df['mb']
        x1 = self.df['x1']
        c = self.df['c']
        M_b, alpha, beta = -19.2, 0.154, 3.69
        mu = mb - M_b + alpha * x1 - beta * c
        mu_c = mb - M_b
        z = self.df['zcmb']
        plt.figure()
        ax = plt.gca()
        plt.errorbar(z, mu, marker = '.', linestyle = 'None', capsize = 2, label = 'with standardization')
        plt.errorbar(z, mu_c, marker = '.', linestyle = 'None', capsize = 2, alpha = 0.1, label = 'without standardization')
        plt.legend()
        ax.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
        ax.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
        ax.set_yticks(np.arange(31, 41, 0.5), minor = True)
        ax.set_xticks(np.arange(0, 0.15, 0.005), minor = True)
        plt.xlabel('Redshift')
        plt.ylabel('Distance modulus');
    
    def analysis(self, H0, delta_H0, method, n, n_dip, N_patch, N_resol, anisotropies, theta_dip, phi_dip, delta_H0_dip, H0_dip, filename, filename_residu):#, minuit, list_H0, delta_H0_F):
        """
        Make the full analysis with or without anisotropies for each methods, apply the global fit, the fit for each patch and make the calculation for uncertainties
        
        Parameters
    ----------
    H0: a float
        input value of the Hubble constant in km s^-1 Mpc^-1
    delta_H0: 0
        Value: 0, to *delta_H0, his length depend on N_patch
    method: a string
        could be 'clustering' for the clustering method to patch the map, or 'fixe' the fixe one that use HEALPix
    n: an interger
        simulation number
    n_dip: an integer
        dipole indices
    N_patch: an integer
        number of patch
    N_resol: an integer
        the degree of pixelisation, the resolution
    anisotropies: a boolean
        True if there are anisotropies
    theta_dip: a float
        theta value for the dipole direction
    phi_dip: a float
        phi value for the dipole direction
    delta_H0_dip: a float
        Value of the dipole amplitude (for H0 variations)
    H0_dip: a float
        input H0 value for the dipole
    filename: a string
        name/repertory of the file for sensitivity
        """
        
        centers = pd.read_csv('centers_clust(3000{}).csv'.format(n), header = None)
        # If there are anisotropies, put the dipole effect on z values
        if anisotropies == True:
            theta = np.radians(90 - self.df['dec'])
            phi = np.radians(self.df['ra'])

            x_d = np.sin(theta_dip) * np.cos(phi_dip)
            x = np.sin(theta) * np.cos(phi)
            y_d = np.sin(theta_dip) * np.sin(phi_dip)
            y = np.sin(theta) * np.sin(phi)
            z_d = np.cos(theta_dip)
            z = np.cos(theta)
            delta_theta = x_d * x + y_d * y + z_d * z
            z = (1 + delta_H0_dip/H0_dip * delta_theta) * self.df['zcmb']
            self.df['zcmb'] = z
            
        # Errors of the mu_th for the global fit
        hubble = Hubble_fit(H0, self.df)
        self.df['sig_th_GF'] = ''
        sth_GF = hubble.sig_th(0.3, self.df['zcmb'], 1/2 * 1e-3)
        for i in range(len(self.df)):
            self.df.loc[i, 'sig_th_GF'] = sth_GF[i]
        
        # Global fit
        M_b, alpha, beta, omega_m = -19.2, 0.154, 3.69, 0.3
        param_global, val_global, cov_global, names_global = hubble.fit(M_b, alpha, beta, omega_m)
        
        # Errors on mu_exp with global fit value for alpha and beta
        hubble_patch = Hubble_fit_patch(H0, self.df, N_patch)
        self.df['sig_mu'] = ''
        for i in range(len(self.df)):
            self.df.loc[i, 'sig_mu'] = hubble_patch.sig_i(val_global[1], val_global[2], self.df['cov_mb'][i])
        
        # Save dataframe with new errors and fit results (GF)
        if anisotropies == True:
            if method == 'clustering':
                self.df.to_pickle('df_GF(3000{})_clust_aniso.pkl'.format(n))
                covbis = []
                for i in range(0, 4, 1):
                    covbis.append(np.sqrt(cov_global[i,i]))
                tab = np.array([names_global, val_global, covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('GF_clust_{}_aniso.csv'.format(n))
            if method == 'fixe':
                self.df.to_pickle('df_GF(3000{})_fix_aniso.pkl'.format(n))
                covbis = []
                for i in range(0, 4, 1):
                    covbis.append(np.sqrt(cov_global[i,i]))
                tab = np.array([names_global, val_global, covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('GF_pix_{}_aniso.csv'.format(n))
        else:
            if method == 'clustering':
                self.df.to_pickle('df_GF(3000{})_clust.pkl'.format(n))
                covbis = []
                for i in range(0, 4, 1):
                    covbis.append(np.sqrt(cov_global[i,i]))
                tab = np.array([names_global, val_global, covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('GF_clust_{}.csv'.format(n))
            if method == 'fixe':
                self.df.to_pickle('df_GF(3000{})_fix.pkl'.format(n))
                covbis = []
                for i in range(0, 4, 1):
                    covbis.append(np.sqrt(cov_global[i,i]))
                tab = np.array([names_global, val_global, covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('GF_pix_{}.csv'.format(n))
        
        
        # Errors on mu_th for patching fit
        sth = hubble_patch.sig_th(self.df['zcmb'], 1/2 * 1e-3)
        for i in range(len(self.df)):
            self.df.loc[i, 'sig_th'] = sth[i]
        # Fit
        M_b, alpha, beta = val_global[0], 0.154, 3.69
        #param, val, cov, names = hubble_patch.fit(M_b, alpha, beta, minuit, eval(delta_H0_F))
        param, val, cov, names = hubble_patch.fit(M_b, alpha, beta, *delta_H0)

        # Errors on mu_exp with global fit value for alpha and beta
        self.df['sig_mu'] = ''
        for i in range(len(self.df)):
            self.df.loc[i, 'sig_mu'] = hubble_patch.sig_i(val[1], val[2], self.df['cov_mb'][i])
    
        # Save dataframe with new errors and fit results
        if anisotropies == True:
            if method == 'clustering':
                self.df.to_pickle('df_F(3000{})_clust_aniso.pkl'.format(n))
                covbis = []
                for i in range(0, N_patch+3, 1):
                    covbis.append(np.sqrt(cov[i,i]))
                tab = np.array([names[:N_patch+3], val[:N_patch+3], covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('Fit_clust_{}_aniso.csv'.format(n))
            if method == 'fixe':
                self.df.to_pickle('df_F(3000{})_fix_aniso.pkl'.format(n))
                covbis = []
                for i in range(0, N_patch+3, 1):
                    covbis.append(np.sqrt(cov[i,i]))
                tab = np.array([names, val, covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('Fit_pix_{}_aniso.csv'.format(n))
        else:
            if method == 'clustering':
                self.df.to_pickle('df_F(3000{})_clust.pkl'.format(n))
                covbis = []
                for i in range(0, N_patch+3, 1):
                    covbis.append(np.sqrt(cov[i,i]))
                tab = np.array([names, val, covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('Fit_clust_{}.csv'.format(n))
            if method == 'fixe':
                self.df.to_pickle('df_F(3000{})_fix.pkl'.format(n))
                covbis = []
                for i in range(0, N_patch+3, 1):
                    covbis.append(np.sqrt(cov[i,i]))
                tab = np.array([names, val, covbis]).T
                df_fit = pd.DataFrame(tab, columns = ['Name', 'values', 'errors'])
                df_fit.to_csv('Fit_pix_{}.csv'.format(n))
            
        # Write an a csv file the mean value and standard deviation for delta_H0    
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            data = [n, method, np.array(val[3:]).mean(), np.array(val[3:]).std()]
            writer.writerow(data)
    
        # If there are anisotropies save delta_H0 values with errors and RaDec coordinates
        if anisotropies == True:
            covbis = []
            for i in range(3, N_patch+3, 1):
                covbis.append(np.sqrt(cov[i,i]))
            ra_dip, dec_dip = np.degrees(phi_dip), np.degrees(np.pi/2 - theta_dip)
            if method == 'clustering':
                ra, dec = centers[1], np.pi/2 - centers[0]
                for j in range(0, N_patch):
                    with open(filename_residu, 'a') as f:
                        writer = csv.writer(f)
                        data = [n_dip, delta_H0_dip, method, n, ra_dip, dec_dip, val[3:][j], covbis[j], ra[j], dec[j]]
                        writer.writerow(data)
            if method == 'fixe':
                m_mask = np.arange(hp.nside2npix(N_resol))
                ang = hp.pix2ang(N_resol, m_mask)
                ra, dec = ang[1], np.pi/2 - ang[0]
                for j in range(0, N_patch):
                    with open(filename_residu, 'a') as f:
                        writer = csv.writer(f)
                        data = [n_dip, delta_H0_dip, method, n, ra_dip, dec_dip, val[3:][j], covbis[j], ra[j], dec[j]]
                        writer.writerow(data)
            #if method == 'clustering':
             #   ra, dec = centers[1], np.pi/2 - centers[0]
             #   tab = np.array([val[3:],covbis ,ra, dec]).T
              #  df_fit = pd.DataFrame(tab, columns = ['$\delta H_0$', 'Error', 'ra', 'dec'])
              #  df_fit.to_csv('aniso_clust_{}_{}.csv'.format(n, n_dip))
           # if method == 'fixe':
            #    m_mask = np.arange(hp.nside2npix(N_resol))
            #    ang = hp.pix2ang(N_resol, m_mask)
             #   ra, dec = ang[1], np.pi/2 - ang[0]
             #   tab = np.array([val[3:],covbis ,ra, dec]).T
              #  df_fit = pd.DataFrame(tab, columns = ['$\delta H_0$', 'Error', 'ra', 'dec'])
              #  df_fit.to_csv('aniso_pix_{}_{}.csv'.format(n, n_dip))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        