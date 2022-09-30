#from pathlib import Path  #Object-oriented filesystem paths
#import pandas as pd       #Python Data Analysis Library
#import wget               #Package to retrieve content from web servers
#import time               #Time access and conversions module
import xarray as xr
#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt
#import matplotlib.path as mpath
#from matplotlib.animation import FuncAnimation
import numpy as np
#import itertools
#from scipy.stats import kurtosis


def distance(lon1,lat1,lon2,lat2):
    """computes the great circle distance between 
        two points on a sphere (the Earth)"""
    r=6371 # Earth's radius in km
    hav = lambda x :  (1-np.cos(np.radians(x)))/2 
    h = hav(lat2 - lat1) + (1-hav(lat1-lat2)-hav(lat1+lat2))*hav(lon2-lon1)
    return 2*r*np.arcsin(np.sqrt(h))

def dists_to_c(points, centers,T,n):
    """calculates the distance from the borderpoints to
        the sea ice center"""
    dists_c=np.zeros((T,n))
    for i in range(T):
        Blo=points[i,0,:]
        Bla=points[i,1,:]
        clo=centers[i,0]
        cla=centers[i,1]
        for k,lo in enumerate(Blo):
            la=Bla[k]
            dists_c[i,k] = distance(lo,la,clo,cla)
    return dists_c

def calc_geo_dists(borders):
    """ calculates the geographical distance between border points"""
    geo_dists = np.zeros((borders.shape[0], borders.shape[2], borders.shape[2]))
    for tt,t in enumerate(borders):
        for ii,i in enumerate(t.T):
            for jj,j in enumerate(t.T):
                geo_dists[tt,ii,jj] = distance(i[0], i[1], j[0], j[1])
    return geo_dists


def calc_shifted_rows(covmat):
    """shift the rows of the covariance matrix, so that the variance is in column 0 and not on the diagonal"""
    shifted_rows = []
    for i,row in enumerate(covmat):
        new_row = np.concatenate((row[i:],row[:i]))
        shifted_rows.append(new_row)
    shifted_rows= np.array(shifted_rows)
    return shifted_rows

def binning_geo_cov(shifted_geo_dists, shifted_row, stepsize = 50, geo_to = "midpoint"):
    """function to take the mean covariance within a bin of size stepsize of the grographical distance"""
    bin_range = np.arange(stepsize,np.max(shifted_geo_dists), step = stepsize)
    geos = np.zeros_like(bin_range)
    covs = np.zeros_like(bin_range)
    for i,s in enumerate(bin_range):
        idx = np.where((shifted_geo_dists.flatten() < s) & (shifted_geo_dists.flatten() >= s-stepsize)) #point where dist in [s-50,s)
        geo = shifted_geo_dists.flatten()[idx]
        co = shifted_row.flatten()[idx]
        co_mean = np.mean(co)
        covs[i] = co_mean
        if geo_to == "mean":
            geo_midpoint = np.mean(geo)
            geos[i] = geo_midpoint
        elif geo_to == "start":
            geo_midpoint = s-stepsize
            geos[i] = geo_midpoint
        elif geo_to == "midpoint":
            geo_midpoint = s-stepsize/2
            geos[i] = geo_midpoint
        elif geo_to == "end":
            geo_midpoint = s
            geos[i] = geo_midpoint  
    
    if geo_to in ["mean", "start", "midpoint", "end"]:
        return geos, covs
    else:
        print("Invalid geo_to type, select mean, start, midpoint or end") 


def get_errorbars_geo_cov(shifted_geo_dists, detrended_dist, n_shuffles = 100, 
                        bin_steps = 50, geo_set_to = "midpoint"):
    """function to get std errorbars on the mean 
        covariance vs ge distance relatinship (binned)"""
    num_bins = len(np.arange(bin_steps,np.max(shifted_geo_dists), step = bin_steps))
    n_points = detrended_dist.shape[0]
    covs_shuffled = np.zeros((n_shuffles, num_bins))
    geos_shuffled = np.zeros((n_shuffles, num_bins))
    #shuffle
    for i in range(n_shuffles):
        idx = np.arange(n_points)
        np.random.shuffle(idx)
        shuf_dist_c = detrended_dist[idx]
        #calc covariace
        cov_mat = np.cov(shuf_dist_c, bias = True)
        #shift the rows
        shifted_shuffled = calc_shifted_rows(cov_mat)
        shuf_geo_dist = shifted_geo_dists[idx][idx]
        
        #calculate the binned covariance as function of geographical distance
        geo, cov = binning_geo_cov(shuf_geo_dist, shifted_shuffled, stepsize = bin_steps, 
                                    geo_to = geo_set_to)
        geos_shuffled[i] = geo
        covs_shuffled[i] = cov

    #take errorbars as std
    err_bars = np.std(covs_shuffled,axis=0)
    return err_bars

def get_spatial_scale(binned_geo, binned_cov,
        err_binned_geo_cov, until = 1000, 
        n_scale_shuffles = 200, get_fit = False):
    """calculates the spatial scale (for the binned or mean) relationship between mean covariance and geographical
        distance by a OLS linear fit until a certain distance. Errorbars (binned case) are taken from the std of shuffling in space"""
    #calc spatial scale
    lim = np.where(binned_geo<=until)
    y = np.log(binned_cov[lim])
    #N = y.shape[0]
    x= binned_geo[lim]
    beta= np.polyfit(x,y,1)
    y_est= beta[1] + beta[0]*x
    s = -1/beta[0]

    #do errorbars
    scales = np.zeros(n_scale_shuffles)
    for i in range(n_scale_shuffles):
        y_pert = np.log(binned_cov[lim] + np.random.normal(0, err_binned_geo_cov[lim]) )
        beta_pert = np.polyfit(x,y_pert,1)
        scales[i] = -1/beta_pert[0]
    std_scale = np.std(scales)
    if get_fit:
            return s, std_scale, x, y_est
    else:
        return s, std_scale

    