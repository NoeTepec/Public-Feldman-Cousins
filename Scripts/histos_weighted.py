import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#####################tag################### WEIGHTED 1D HISTOGRAMS  ########################################
#def mask_inBin(data, bin_edges, index):
#    events_in = (data>= bin_edges[index])  & (data< bin_edges[index+1])
#    return events_in

def chi2_histogram(data1, data2, weights1=None, weights2=None, 
                   ensure_positive_counts=True, return_histos=False, 
                   min_prob=0,
                   ignore_bins_wzero_counts=True, **kwargs):
    """Evaluate the chi2 2 sample test by binning the histograms in the same way
    If data is weighted the uncertainty is taken as sqrt(sum(w**2))
    If ensure_positive_counts, reduce the number of bins by 1 if there is a bin with negative counts.
    Dof is the number of non-empty bins (in both histos) -1 """
    #Get initial number of weighted counts and corresponding uncertainty
    h1 = histogram_weighted(data1, weights=weights1, density=True, **kwargs)
    h2 = histogram_weighted(data2, weights=weights2, density=True, bins=h1[1])
    bin_size = np.mean(h1[1][1:]-h1[1][:-1])
    rng = [h1[1][0], h1[1][-1]]
    nbins = len(h1[0])
    
    # If number of counts is negative, reduce the number of bins by 1 
    # repeat until only positive counts
    while ( (h1[0]<0).any() or (h2[0]<0).any() ) and ensure_positive_counts:
        nbins-=1
        h1 = histogram_weighted(data1, weights=weights1, density=True, range=rng, bins=nbins)
        h2 = histogram_weighted(data2, weights=weights2, density=True, bins=h1[1] )
    
    #Evaluate numerator and denominator, and the chi2 per bin
    difference = h1[0]-h2[0]
    error_no_corr = np.hypot(h1[2], h2[2])
    if ignore_bins_wzero_counts:
        mask = np.bitwise_and(h1[0]>min_prob, h2[0]>min_prob)
        difference=difference[mask]
        error_no_corr=error_no_corr[mask]
        
    ratio = difference/error_no_corr
    chi2_list = np.power(ratio,2)
    
    #Remove nans (should occur only when there are 0 counts)
    #chi2_list = chi2_list[~np.isnan(chi2_list)]
    
    chi2 = chi2_list.sum()
    dofs = len(chi2_list)-1
    p_val = 1 - stats.chi2.cdf(chi2, dofs)
    
    
    if return_histos:
        return chi2, dofs, p_val, h1, h2

    return chi2, dofs, p_val



def mask_underflow(data, bin_edges):
    events_in = (data< bin_edges[0])
    return events_in

def mask_overflow(data, bin_edges):
    events_in = (data>= bin_edges[-1])
    return events_in

def mask_inBin(data, bin_edges, index):
    return (data>= bin_edges[index])  & (data< bin_edges[index+1])

def histogram_weighted(data, bins, weights=None,density=False,**kwargs):
    
    supported_types = [np.ndarray, pd.Series]    
    if not type(weights) in supported_types:
        if weights==None:
            weights = np.ones_like(data)
            
    
    counts, bin_edges = np.histogram(data, bins=bins, **kwargs)
    bin_size = bin_edges[1]-bin_edges[0]
    bins = len(counts)
    
    counts_weighted = np.zeros_like(counts, dtype=float)
    errors_weighted = np.zeros_like(counts, dtype=float)
    
    for i in range(bins):
        events_in = mask_inBin(data, bin_edges, i)
        #print(events_in)
        #print(weights)
        counts_weighted[i] = np.sum(weights[events_in])
        errors_weighted[i] = np.sqrt(np.sum(np.power(weights[events_in], 2)))
    
    if density:
        sum_w            = np.sum(counts_weighted)
        counts_weighted /= (sum_w*bin_size)
        errors_weighted /= (sum_w*bin_size)
    
    return (counts_weighted, bin_edges, errors_weighted)





def hist_weighted(data, bins, weights=None, axis=None, only_pos=False, density=False, **kwargs):    
    
    supported_types = [np.ndarray, pd.Series]    
    if not type(weights) in supported_types:
        if weights==None:
            weights = np.ones_like(data)
    
    
    if 'range' in kwargs:
        range_ = kwargs['range']
        del kwargs['range']
    else:
        range_ = None
    
    histo_www = histogram_weighted(data,
                                             bins,
                                             weights=weights,
                                             density=density, 
                                             range=range_)
    counts_weighted, bin_edges, errors_weighted = histo_www
    
    
    events_under = mask_underflow(data, bin_edges)
    events_over  = mask_overflow(data,  bin_edges)
    
    bin_mean = (bin_edges[1:]+bin_edges[:-1])/2 
    bin_size = bin_edges[1]-bin_edges[0]
    
    if  only_pos:
        non_zero        = counts_weighted>0
        bin_mean        = bin_mean[non_zero] 
        counts_weighted = counts_weighted[non_zero]
        errors_weighted = errors_weighted[non_zero]
    
    
    hist_type = kwargs.get('hist_type', 'error')
    if 'hist_type' in kwargs:
        del kwargs['hist_type']
    if axis:
        line_style = kwargs.get('ls', 'none')
        if line_style!= 'none': del kwargs['ls']
            
        if hist_type=='bar':
            axis.bar(bin_mean, 
                    counts_weighted, 
                    width=bin_size, 
                    **kwargs)
        elif hist_type=='step':
            axis.step(bin_mean, 
                    counts_weighted, 
                    where='mid',
                    #width=bin_size, 
                    **kwargs)
        else:
            axis.errorbar(x = bin_mean,  xerr=bin_size/2,
                      y = counts_weighted, yerr=errors_weighted,
                      ls = line_style,
                      **kwargs)
    else:
        if 'ls' in kwargs: line_style = kwargs['ls'];del kwargs['ls']
        else: line_style = 'none'
                     
                     
        if hist_type=='bar':
            plt.bar(bin_mean, 
                    counts_weighted, 
                    width=bin_size, 
                   **kwargs)
        elif hist_type=='step':
            plt.step(bin_mean, 
                    counts_weighted, 
                    where='mid',
                    #width=bin_size, 
                    **kwargs)
        else:        
            plt.errorbar(x  = bin_mean,        xerr = bin_size/2,
                     y  = counts_weighted, yerr = errors_weighted,
                     ls = line_style,
                      **kwargs)
        
    if any(events_under):
        under_cou = np.sum(weights[events_under])
        under_err = np.sqrt(np.sum(np.power(weights[events_under], 2)))
        print(f'Underflow (<{np.round(bin_edges[0],3)})')
        print('\t', round(under_cou, 2), '+-', round(under_err,2) )
        print('\tUnweighted ', len(weights[events_under]), '\n' )
        
    if any(events_over):
        over_cou = np.sum(weights[events_over])
        over_err = np.sqrt(np.sum(np.power(weights[events_over], 2)))
        print(f'Overflow  (>={np.round(bin_edges[-1],3)})')
        print('\t', round(over_cou, 2), '+-', round(over_err,2) )
        print('\tUnweighted ', len(weights[events_over]), '\n' )
    
        
    return counts_weighted, bin_edges, errors_weighted
######################################## WEIGHTED 1D HISTOGRAMS  ########################################
