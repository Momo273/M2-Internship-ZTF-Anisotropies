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
import matplotlib
from matplotlib.gridspec import GridSpec


class SkyMap:
    """ Plot in mollweide view the position of Supernovae in equatorial, galactic or ecliptic coordinates
    And display the redshift of each Supernovae with a color bar
    Projection of pixel or scatter plot """
    
    def __init__(self, nside, lon, lat, var, var_name, title):
        """
        Parameters
    ----------
    nside : int
            Resolution (power of 2)
    lon: an array or a column from a dataframe
            Represents the longitude
    lat: an array or a column from a dataframe
            Represents the latitude
    var: an array or a column from a dataframe
            Represents the variable that we want to represent
    var_name: str
            Name of the variable
    title: str
            Part of the final title to know what it's represented 
      """
            # Initialization
        self.lat = lat
        self.lon = lon
        self.var = var
        self.var_name = var_name
            # Plans coordinates
        self.plan_lon = np.linspace(0, 360, 10000)
        self.plan_lat = np.zeros(10000)
            # fill the fullsky map of a zero value 
        self.m = np.zeros(hp.nside2npix(nside)) # np.nside2npix() Give the number of pixels for the given nside
            # convert to HEALPix indices
        self.pixel_indices = hp.ang2pix(nside, np.array(np.pi/2 - np.radians(self.lat)), np.array(np.radians(self.lon)))
            # Apply at our full sky the value for HEALPix indices: here with the associated redshift z
        self.m[self.pixel_indices] = self.var
        self.title = title
        
        
    def pixel_eq(self):
        """ Pixel representation in equatorial coordinates"""
            # Ecliptic
        ecl = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame = 'geocentrictrueecliptic') # use astropy for plan coordinates
        ra_ecl, dec_ecl = ecl.icrs.ra.value, ecl.icrs.dec.value # Transforms ecliptic coordinates into radec
            # Transforms degree into radian
        theta_ecl = np.radians(90. - dec_ecl)
        phi_ecl = np.radians(ra_ecl)
            # Galaxy line
        gal = SkyCoord(self.plan_lon, self.plan_lat, frame = 'galactic', unit = 'degree') # use astropy for plan coordinates
        ra_gal, dec_gal = gal.icrs.ra.value, gal.icrs.dec.value # Transforms galactic coordinates into radec
            # Transforms degree into radian
        theta_gal = np.radians(90. - dec_gal)
        phi_gal = np.radians(ra_gal)
            # Plot in Mollweide view
        hp.mollview(self.m, title = 'Equatorial coordinates representation of {}'.format(self.title), unit = self.var_name, coord=['C'])
            # Projection of the different plans
        hp.projplot(theta_ecl, phi_ecl, color = 'blue', label = 'Ecliptic')
        hp.projplot(theta_gal, phi_gal, color = 'red', label = 'Galactic')
        plt.legend()
        hp.graticule()
    
    
    def pixel_gal(self):
        """ Pixel representation in galactic coordinates"""
            # Ecliptic
        ecl = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame = 'geocentrictrueecliptic') # use astropy for plan coordinates
        l_ecl, b_ecl = ecl.galactic.l.value, ecl.galactic.b.value # Transforms ecliptic coordinates into galactic
            # Transforms degree into radian
        theta_ecl = np.radians(90. - b_ecl)
        phi_ecl = np.radians(l_ecl)
            # Equatorial
        eq = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame='icrs') # use astropy for plan coordinates
        l_eq, b_eq = eq.galactic.l.value, eq.galactic.b.value # Transforms equatorial coordinates into ecliptic
            # Transforms degree into radian
        theta_eq = np.radians(90. - b_eq)
        phi_eq = np.radians(l_eq)
            # Plot in Mollweide view
        hp.mollview(self.m, title = 'Galactic Coordinates representation of {}'.format(self.title), unit = self.var_name, coord=['C','G'])
            # Projection of the different plans
        hp.projplot(theta_ecl, phi_ecl, color = 'red', label = 'Ecliptic')
        hp.projplot(theta_eq, phi_eq, color = 'blue', label = 'Equatorial')
        plt.legend()
        hp.graticule()
        
        
    def pixel_ecl(self):
        """ Pixel representation in ecliptic coordinates"""
            # Galactic
        gal = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame = 'galactic')# use astropy for plan coordinates
        lon_gal, lat_gal = gal.geocentrictrueecliptic.lon.value, gal.geocentrictrueecliptic.lat.value # Transforms galactic coordinates into ecliptic
            # Transforms degree into radian
        theta_gal = np.radians(90. - lat_gal)
        phi_gal = np.radians(lon_gal)
            # Equatorial
        eq = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame='icrs') # use astropy for plan coordinates
        lon_eq, lat_eq = eq.geocentrictrueecliptic.lon.value, eq.geocentrictrueecliptic.lat.value # Transforms equatorial coordinates into ecliptic
            # Transforms degree into radian
        theta_eq = np.radians(90. - lat_eq)
        phi_eq = np.radians(lon_eq)
            # Projections
        hp.mollview(self.m, title = 'Ecliptic coordinates representation of {}'.format(self.title), unit = self.var_name, coord = ['C','E'])
        hp.graticule()
            # Projection of the different plans
        hp.projplot(theta_gal, phi_gal, color = 'red', label = 'Galactic')
        hp.projplot(theta_eq, phi_eq, color = 'blue', label = 'Equatorial')
        plt.legend()
        
        
    def scatter_eq(self):
        """ Scatter representation in equatorial coordinates"""
            # Ecliptic
        ecl = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame = 'geocentrictrueecliptic') # use astropy for plan coordinates
        ra_ecl, dec_ecl = ecl.icrs.ra.value, ecl.icrs.dec.value # Transforms ecliptic coordinates into equatorial
            # Transforms degree into radian
        theta_ecl = np.radians(90. - dec_ecl)
        phi_ecl = np.radians(ra_ecl)
            # Galactic plan
        gal = SkyCoord(self.plan_lon, self.plan_lat, frame = 'galactic', unit = 'degree') # use astropy for plan coordinates
        ra_gal, dec_gal = gal.icrs.ra.value, gal.icrs.dec.value # Transforms galactic coordinates into equatorial
            # Transforms degree into radian
        theta_gal = np.radians(90. - dec_gal)
        phi_gal = np.radians(ra_gal)
            # Mollweide view
        hp.mollview(title = 'Equatorial coordinates representation of {}'.format(self.title))
        hp.graticule()
        psc = hp.projscatter(np.pi/2 - np.radians(self.lat), np.radians(self.lon), c = self.var, s = 3)
        plt.colorbar(psc, label = self.var_name)
            # Projection of the different plans
        hp.projplot(theta_gal, phi_gal, color = 'red', label = 'Galactic')
        hp.projplot(theta_ecl, phi_ecl, color = 'blue', label = 'Ecliptic')
        plt.legend()
        
        
    def scatter_gal(self):
        """ Scatter representation in galactic coordinates"""
            # Change Equatorial coordinates into galactic coordinates then in spherical coordinates
        data = SkyCoord(self.lon, self.lat, unit='deg', frame='icrs')
        l, b = data.galactic.l.value, data.galactic.b.value 
            # Transforms degree into radian
        b = np.radians(90. - b)
        l = np.radians(l)
            # Ecliptic
        ecl = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame = 'geocentrictrueecliptic') # use astropy for plan coordinates
        l_ecl, b_ecl = ecl.galactic.l.value, ecl.galactic.b.value # Transforms ecliptic coordinates into galactic
            # Transforms degree into radian
        theta_ecl = np.radians(90. - b_ecl)
        phi_ecl = np.radians(l_ecl)
            # Equatorial
        eq = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg') # use astropy for plan coordinates
        l_eq, b_eq = eq.galactic.l.value, eq.galactic.b.value # Transforms equatorial coordinates into galactic
            # Transforms degree into radian
        theta_eq = np.radians(90. - b_eq)
        phi_eq = np.radians(l_eq)
            # Projections
        hp.mollview(title = 'Galactic coordinates representation of {}'.format(self.title))
        hp.graticule()
        psc = hp.projscatter(b, l, c = self.var)
            # Projection of the different plans
        hp.projplot(theta_ecl, phi_ecl, color = 'red', label = 'Ecliptic')
        hp.projplot(theta_eq, phi_eq, color = 'blue', label = 'Equatorial')
        plt.colorbar(psc, label = self.var_name)
        plt.legend()
        
        
    def scatter_ecl(self):
        """ Scatter representation in ecliptic coordinates"""
            # Change Equatorial coordinates into galactic coordinates then in spherical coordinates
        data = SkyCoord(self.lon, self.lat, unit='deg') 
        l, b = data.geocentrictrueecliptic.lon.value, data.geocentrictrueecliptic.lat.value
            # Transforms degree into radian
        b = np.radians(90. - b)
        l = np.radians(l)
            # Galactic
        gal = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame = 'galactic') # use astropy for plan coordinates
        lon_gal, lat_gal = gal.geocentrictrueecliptic.lon.value, gal.geocentrictrueecliptic.lat.value # Transforms galactic coordinates into ecliptic
            # Transforms degree into radian
        theta_gal = np.radians(90. - lat_gal)
        phi_gal = np.radians(lon_gal)
            # Equatorial
        eq = SkyCoord(self.plan_lon, self.plan_lat, unit = 'deg', frame='icrs') # use astropy for plan coordinates
        lon_eq, lat_eq = eq.geocentrictrueecliptic.lon.value, eq.geocentrictrueecliptic.lat.value # Transforms equatorial coordinates into ecliptic
            # Transforms degree into radian
        theta_eq = np.radians(90. - lat_eq)
        phi_eq = np.radians(lon_eq)
            # Projections
        hp.mollview(title = 'Ecliptic coordinates representation of {}'.format(self.title))
        hp.graticule()
        psc = hp.projscatter(b, l, c = self.var)
            # Projection of the different plans
        hp.projplot(theta_gal, phi_gal, color = 'red', label = 'Galactic')
        hp.projplot(theta_eq, phi_eq, color = 'blue', label = 'Equatorial')
        plt.colorbar(psc, label = self.var_name)
        plt.legend()
        
    #def coord_(self):
     #   if self.syst_coord == 'equatorial':
      #      lon_eq, lat_eq = self.lon, self.lat
       #     eq = SkyCoord(lon_eq, lat_eq, unit = 'deg', frame='icrs')
        #    lon_gal, lat_gal = eq.galactic.l.value, eq.galactic.b.value
         #   lon_ecl, lat_ecl = eq.geocentrictrueecliptic.lon.value, eq.geocentrictrueecliptic.lat.value
             
        #if self.syst_coord == 'galactic':
         #   lon_gal, lat_gal = self.lon, self.lat
          #  gal = SkyCoord(lon_gal, lat_gal, unit = 'deg', frame='galactic')
           # lon_eq, lat_eq = gal.icrs.ra.value, gal.icrs.dec.value
            #lon_ecl, lat_ecl = gal.geocentrictrueecliptic.lon.value, gal.geocentrictrueecliptic.lat.value
        
        #if self.syst_coord == 'ecliptic':
         #   lon_ecl, lat_ecl = self.lon, self.lat
          #  ecl = SkyCoord(lon_ecl, lat_ecl, unit = 'deg', frame = 'geocentrictrueecliptic')
           # lon_gal, lat_gal = ecl.galactic.l.value, ecl.galactic.b.value
            #lon_eq, lat_eq = ecl.icrs.ra.value, ecl.icrs.dec.value

        #return lon_eq, lat_eq, lon_gal, lat_gal, lon_ecl, lat_ecl

        
class Patching_Map(SkyMap):
    """ partitions the map into patches """
    
    def __init__(self, nside, lon, lat, var, var_name, title):
        """ 
            Parameters
    ----------
    lon: an array or a column from a dataframe
            Represents the longitude
    lat: an array or a column from a dataframe
            Represents the latitude
    var: an array or a column from a dataframe
            Represents the variable
    var_name: str
            Name of the variable
    nside : int
            Resolution (power of 2)
    title: str
            Part of the final title to know what it's represented
        """
        SkyMap.__init__(self, nside, lon, lat, var, var_name, title) # we get the instance, methods and parameters of the class SkyMap
        self.nside = nside # Resolution
        self.npix = np.arange(hp.nside2npix(self.nside)) # number of pixel
        self.theta, self.phi = hp.pix2ang(self.nside, self.npix) # pixel angles: theta, phi 
        self.ang = np.array([self.theta.tolist(), self.phi.tolist()]).T # angles in an array
        self.mask_clustering = (self.ang[:,0] < np.radians(120)) # mask used for the clustering on pixel for didn't have any pixel under -30 ° dec
        
    def histo_redshift(self, m, label):
        """
        Plot the distribution of Supernovae according to the redshift
        
        Parameters
    ----------
    m : an array
            Contains the value of each pixels on the map
    label : str
            The label for each histogram
        """
        plt.subplot(1, 2, 1)
        plt.hist(m, bins = 15, density = True, label = label, range = [0, 0.15], histtype = 'step')
    
    def histo_count(self, m, label):
        """
        Plot distribution of each patch
        
        Parameters
    ----------
    m : an array
            Contains the value of each pixels on the map
    label : str
            The label for each histogram
        """
        plt.bar(label, len(m))
        plt.xlabel('Patch')
        plt.ylabel('Number of Supernovae $\mathrm{1}$a')
        plt.title('Repartition in each patch');
    
    def legend(self):
        """
        Display labels on axis, the legend and the title
        """
        plt.xlabel(self.var_name)
        plt.ylabel('Number of Supernovae $\mathrm{1}$a')
        plt.legend(loc = 'upper left')
        plt.title('Distribution from differents patch');
    
    
    def patch_polygon(self, ra, dec, df):
        """ Partition the map with query_polygon 
        
        Parameters
    ----------
     ra : an array
            Right ascension for each vertices of polygons
     dec : an array
            Declination for each vertices of polygons
     df: Pandas Dataframe
     
         Return
    ----------
    clust : list of dataframe
        """
        clust = []
        plt.figure(figsize = (10, 5))
        for i in range(len(ra)):
            # Transforms degree into radian
            theta = np.pi/2 - np.radians(dec[i])
            phi = np.radians(ra[i])
            # Define polygon vertices
            vec = hp.ang2vec(theta, phi)
            # make the query_polygon to return all pixel in
            ipix = hp.query_polygon(self.nside, vec)
            # find index for pixels_indices of SNe correspond with ipix
            idx = np.in1d(self.pixel_indices, ipix)
            mask = (idx == True)
            # Make histo
            self.m[self.pixel_indices[idx]] = self.var[mask]
            plt.subplot(1, 2, 1)
            self.histo_redshift(self.m[self.pixel_indices[idx]], 'Patch {}: {}'.format(i, len(self.m[self.pixel_indices[idx]])))
            self.legend()
            plt.subplot(1, 2, 2)
            self.histo_count(self.m[self.pixel_indices[idx]], '{}'.format(i))
            mask_clust = np.in1d(df['zcmb'], self.m[self.pixel_indices[idx]])
            clust.append(df[mask_clust])
        return clust
    
    def n_cluster(self, range_n_cluster):
        """
        Allows to find the right number of clusters with the Elbow method
        
        Parameters
    ----------
    range_n_cluster: tuple
                    range of numbers of cluster
    ang : an array
                    angles in the map 
        """
        # range to the clustering
        c_range = range_n_cluster
        # Survey angles in an array
        ang_survey = np.array(hp.pix2ang(self.nside, self.pixel_indices)).T
        # K_Means
        res_k = [sclust.KMeans(k, n_init = 5).fit(ang_survey) for k in c_range]
        # Elbow method - inertia
        meas_sil_k = [smet.silhouette_score(ang_survey, resk.labels_) for resk in res_k]
        J_k = [resk.inertia_ for resk in res_k]
        #plt.plot(c_range, meas_sil_k, '.-k')
        plt.plot(c_range, J_k, '+-k')
        plt.xlabel('n_cluster')
        plt.ylabel('Sum of squared errors');
        
        
        
    def patch_clustering_map(self, n_cluster, df):
        """ 
        partition the map thanks to clustering method on the map's pixels 
        
        Parameters
    ----------
    n_cluster : int
                    number of clusters
    df: Pandas Dataframe
        """
        plt.figure(figsize = (10, 5))
        # KMeans
        res = sclust.KMeans(n_cluster).fit(self.ang[self.mask_clustering])
        clust = []
        for i in range(n_cluster):
            mask_clustering = (res.labels_ == i) # mask to find which SNe are in a define cluster
            # New pixels indices with the previous mask
            new_ipix = hp.ang2pix(self.nside, self.theta[self.mask_clustering][mask_clustering], self.phi[self.mask_clustering][mask_clustering])
            idx = np.in1d(self.pixel_indices, new_ipix)
            mask = (idx == True)
            # Make Histo
            self.m[self.pixel_indices[idx]] = self.var[mask]
            plt.subplot(1, 2, 1)
            self.histo_redshift(self.m[self.pixel_indices[idx]], 'Patch {}'.format(i, len(self.m[self.pixel_indices[idx]])))
            self.legend()
            plt.subplot(1, 2, 2)
            self.histo_count(self.m[self.pixel_indices[idx]], '{}'.format(i))
            mask_clust = np.in1d(df['zcmb'], self.m[self.pixel_indices[idx]])
            clust.append(df[mask_clust])
        return clust

        
        
    def patch_clustering_survey(self, n_cluster, df):
        """ 
        partition the map thanks to clustering method on survey
        
        Parameters
    ----------
    n_cluster : int
                    number of clusters
    df: Pandas Dataframe
    
        Return
    ----------
     CLuster's Centers
        """
        # Survey angles in an array
        ang_survey = np.array(hp.pix2ang(self.nside, self.pixel_indices)).T
        # KMeans
        res_survey = sclust.KMeans(n_cluster).fit(ang_survey)
        #df['ncluster'] = ''
        df.loc[:, 'ncluster'] = ''
        plt.figure(figsize = (10, 5))
        for i in range(n_cluster):
            mask_clustering = (res_survey.labels_ == i) # mask to find which SNe are in a define cluster
            # New pixels indices with the previous mask
            new_ipix = hp.ang2pix(self.nside, ang_survey[:,0][mask_clustering], ang_survey[:,1][mask_clustering])
            idx = np.in1d(self.pixel_indices, new_ipix)
            mask = (idx == True)
            # Make Histo
            self.m[new_ipix] = self.var[mask_clustering]
            plt.subplot(1, 2, 1)            
            self.histo_redshift(self.m[new_ipix], 'Patch {}'.format(i))
            self.legend()
            plt.subplot(1, 2, 2)
            self.histo_count(self.m[new_ipix], '{}'.format(i))
            mask_clust = np.in1d(df['zcmb'], self.m[new_ipix])
            # Add a flag in the data frame to know to which cluster each data belongs
            for j in list(df[mask_clust].reset_index()['index']):
                df.loc[j, 'ncluster'] = i
        return res_survey.cluster_centers_


    def pixelisation(self, nside, df):
        """
        Make the partionning with HEALPix
        
        Parameters
    ----------
    nside: int
            Resolution (power of 2)
    df: DataFrame
        a dataframe with selected SNe
        """
        ipix = hp.ang2pix(nside, np.array(np.pi/2 - np.radians(self.lat)), np.array(np.radians(self.lon)))
        df['ncluster'] = ipix
    
    
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
    ---------
    mu: float , array 
        Distance modulus
        """
        # Distance modulus with corrections
        mu = mb - M_b + alpha * x1 - beta * c 
        # Distance modulus without corrections
        mu_c = mb - M_b
        #Plot
        seaborn.scatterplot(x = z, y = mu_c, facecolors='none', edgecolor = 'red', linewidth = 1, alpha = 0.4, label = 'without correction', ax = ax.ax_joint)
        ax = seaborn.jointplot(z, mu, label = 'with correction')
        ax.set_axis_labels('Redshift', 'Distance Modulus');
        return mu
    
    
    def Hubble_diagram2(self, M_b, alpha, beta, mb, x1, c, z, y):
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
    y: dataframe of float
        fitted mu_th
        """
        mu = mb - M_b + alpha * x1 - beta * c
        ax = seaborn.jointplot(z, mu, label = 'with correction')
        ax.ax_joint.plot(z, y, color = 'red')
        #seaborn.scatterplot(z, y, facecolors='none', edgecolor = 'red', linewidth = 1, alpha = 0.4, ax = ax.ax_joint)
        ax.set_axis_labels('Redshift', 'Distance Modulus');
        
    def Hubble_diagram3(self, M_b, alpha, beta, mb, x1, c, z, y, y2):
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
    y: dataframe of float
        fitted mu_th
    y2: dataframe of float
        fitted mu_exp
        """
        # Distance modulus with corrections
        mu = mb - M_b + alpha * x1 - beta * c
        # Plot
        ax = seaborn.jointplot(z, mu, label = 'with correction')
        ax.ax_joint.plot(z, y, color = 'red') # fitted plot
        seaborn.scatterplot(z, y2, facecolors='none',edgecolor = 'green', linewidth = 1, alpha = 0.4, ax = ax.ax_joint)
        ax.set_axis_labels('Redshift', 'Distance Modulus');
    
    def diff_exp_fit(self, z, mu_exp, mu_fit, mu_res, sig_exp, sig_res, chi2, log):
        """
        Makes the Hubble diagram, with residue and histogram of residue
        
        Parameters
    ----------
    z: dataframe of float
        Redshifts
    mu_exp: dataframe of float
        Distance modulus from simulations fitted
    mu_fit: dataframe of float
        Distance modulus fitted
    mu_res: dataframe of float
        Residue
    sig_exp: dataframe of float
        uncertainty on mu_exp
    sig_res: dataframe of float
        uncertainty on mu_res
        """
        if log == True:
            fig = plt.figure(figsize=[10,8])
            # Use GridSpec to partition into 3 figure as we want
            gs = GridSpec(4,3, hspace=0, wspace=0)
            ax_joint = fig.add_subplot(gs[0:3,0:2]) # Hubble diagram
            ax_joint_x = fig.add_subplot(gs[3:4,0:2]) # Residue
            ax_marg_y = fig.add_subplot(gs[3:4,2]) # Residue's Histogram 
            ax_joint.errorbar(z, mu_exp, marker='.', yerr = sig_exp, color="dodgerblue", linestyle="None", ecolor="steelblue", capsize = 2, zorder = 1, elinewidth = 0.5, label='mu_exp')
            ax_joint.plot(z, mu_fit, c="red", label="$\Lambda$CDM fit")
            ax_joint.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(31, 41, 1)))
            ax_joint.set_xlabel('Redshift z')
            ax_joint.set_ylabel(r'Distance modulus $\mu = m_B - M_B + \alpha x_1 - \beta c$')  
            ax_joint.set_xscale('log')
            ax_joint.set_xlim(0, 0.16)
            #ax_joint.set_ylim(32, 40)
            ax_joint.axes.get_xaxis().set_visible(False)
            plt.setp(ax_joint.get_yticklabels(), fontsize=12)
            ax_joint.legend(loc='best', shadow=True, fontsize='x-large', title = '$\chi ^2 /n_{}$ = {}'.format('{dof}', chi2))
            ax_joint.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint.set_yticks(np.arange(31.1, 41, 0.1), minor = True)
            #ax_joint.set_xticks(np.arange(0, 0.16, 0.01), minor = True)
            ax_joint_x.errorbar(z, mu_res, yerr = sig_res, marker='.', color="dodgerblue", linestyle="None", capsize = 2, zorder = 1, ecolor="steelblue", label='mu_res', elinewidth = 0.5)
            ax_joint_x.axhline(c='red', linestyle="--")
            #ax_joint_x.axhline(c='pink', linestyle="--")
            ax_joint_x.set_xlabel('Redshift z')
            ax_joint_x.set_ylabel(r'$\mu - \mu_{\Lambda CDM}$')  
            ax_joint_x.set_xscale('log')
            ax_joint_x.set_xlim(0, 0.16)
            ax_joint_x.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint_x.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint_x.set_yticks(np.arange(-0.6, 0.7, 0.1), minor = True)
            #ax_joint_x.set_xticks(np.arange(0, 0.16), minor = True)
            #ax_joint_x.set_ylim(-0.5, 0.5)
            plt.setp(ax_joint_x.get_xticklabels(), fontsize=12)
            plt.setp(ax_joint_x.get_yticklabels(), fontsize=12)
            ax_marg_y.hist(mu_res, orientation='horizontal', color="dodgerblue")
            ax_marg_y.legend(loc='best', shadow=True, fontsize='x-large', title = '$\sigma$ = {0: .4f}'.format(np.std(mu_res)))
            #ax_marg_y.set_ylim(-0.5, 0.5)
            ax_marg_y.set_axis_off()
    
    
        else:
            fig = plt.figure(figsize=[10,8])
            # Use GridSpec to partition into 3 figure as we want
            gs = GridSpec(4,3, hspace=0, wspace=0)
            ax_joint = fig.add_subplot(gs[0:3,0:2]) # Hubble diagram
            ax_joint_x = fig.add_subplot(gs[3:4,0:2]) # Residue
            ax_marg_y = fig.add_subplot(gs[3:4,2]) # Residue's Histogram 
            ax_joint.errorbar(z, mu_exp, marker='.', yerr = sig_exp, color="dodgerblue", linestyle="None", ecolor="steelblue", capsize = 2, zorder = 1, elinewidth = 0.5, label='mu_exp')
            ax_joint.plot(z, mu_fit, c="red", label="$\Lambda$CDM fit")
            ax_joint.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(31, 41, 1)))
            ax_joint.set_xlabel('Redshift z')
            ax_joint.set_ylabel(r'Distance modulus $\mu = m_B - M_B + \alpha x_1 - \beta c$')  
            ax_joint.set_xlim(0, 0.16)
            #ax_joint.set_ylim(32, 40)
            ax_joint.axes.get_xaxis().set_visible(False)
            plt.setp(ax_joint.get_yticklabels(), fontsize=12)
            ax_joint.legend(loc='best', shadow=True, fontsize='x-large', title = '$\chi ^2 /n_{}$ = {}'.format('{dof}', chi2))
            ax_joint.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint.set_yticks(np.arange(31.1, 41, 0.1), minor = True)
            #ax_joint.set_xticks(np.arange(0, 0.16, 0.01), minor = True)
            ax_joint_x.errorbar(z, mu_res, yerr = sig_res, marker='.', color="dodgerblue", linestyle="None", capsize = 2, zorder = 1, ecolor="steelblue", label='mu_res', elinewidth = 0.5)
            ax_joint_x.axhline(c='red', linestyle="--")
            #ax_joint_x.axhline(c='pink', linestyle="--")
            ax_joint_x.set_xlabel('Redshift z')
            ax_joint_x.set_ylabel(r'$\mu - \mu_{\Lambda CDM}$')  
            ax_joint_x.set_xscale('log')
            ax_joint_x.set_xlim(0, 0.16)
            ax_joint_x.xaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint_x.yaxis.set_tick_params(which = 'both', right = True, left = True, direction = 'in')
            ax_joint_x.set_yticks(np.arange(-0.6, 0.7, 0.1), minor = True)
            #ax_joint_x.set_xticks(np.arange(0, 0.16), minor = True)
            #ax_joint_x.set_ylim(-0.5, 0.5)
            plt.setp(ax_joint_x.get_xticklabels(), fontsize=12)
            plt.setp(ax_joint_x.get_yticklabels(), fontsize=12)
            ax_marg_y.hist(mu_res, orientation='horizontal', color="dodgerblue")
            ax_marg_y.legend(loc='best', shadow=True, fontsize='x-large', title = '$\sigma$ = {0: .4f}'.format(np.std(mu_res)))
            #ax_marg_y.set_ylim(-0.5, 0.5)
            ax_marg_y.set_axis_off()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    