"""
Module for custom data functions for use in dashboard
"""

from display.engine import bq, bql, ResponseError
import math
import numpy as np
from numpy import linalg as linalg
import pandas as pd
import matplotlib.pyplot as plt

def get_members(univ):
    """
    @returns: list of tickers of assets in fund
    """
    req = bql.Request(univ, bq.data.id()['value'])
    res = bq.execute(req)
    return res[0].df().values.T[0].tolist()

def get_NAV_chg(tck, startdate, enddate):
    """
    @returns: dataframe of NAV change (%) over specified date range
    """
    req = bql.Request(tck, bq.data.fund_net_asset_val(dates=bq.func.range(start=startdate , end= enddate)).pct_diff())
    res = bq.execute(req)
    return res[0].df().fillna(method="ffill").fillna(method="bfill")

def get_pct_chg(ticker, startdate, enddate):
    """
    @returns: DATAFRAME of numpy array (n x 1) of daily % change price of input ticker (n = # of datapts)
    """
    req = bql.Request(ticker, bq.func.pct_diff(bq.data.px_last(dates=bq.func.range(startdate, enddate), fill='prev')))
    res = bq.execute(req)
    data = res[0].df().fillna(method="ffill").fillna(method="bfill")
    return data

def all_changes(ticker_list, dfdict, correct_len, startdate, enddate, plot_data = False):
    """
    @returns DICT, modifies dictionary with %chg data for each ticker in asset
    """
    str_rep = "PCT_DIFF(PX_LAST(fill='prev',dates=RANGE(" + startdate +"," + enddate +")))"
    
    
    if plot_data: # set plot display settings
        plt.rcParams.update({'font.size': 4})
        x = math.ceil(len(ticker_list)**.5)
        i=1
        plt.figure(figsize=(x*2,x*2))
    
    for tck in ticker_list:
        df = get_pct_chg(tck, startdate, enddate)
        pct_chg = df[str_rep].values
        if pct_chg.shape[0] != correct_len:
            continue
        if plot_data:
            plt.subplot(x,x,i)
            i+=1
            plt.plot(df['DATE'], pct_chg)
            plt.tick_params(axis='x', rotation=70)
            plt.title(tck)
        dfdict[tck] = pct_chg
        
    if plot_data: #reset plot display settings
        plt.show()
        plt.rcParams.update({'font.size': 12})
    return dfdict

def corr_matrix(df, cef):
    """ 
    @returns DF correlation matrix from input data dataframe (sorted in max -> min order)
    """
    corr = df.corr().abs()
    corr.fillna(0, inplace= True)
    return corr.sort_values(by=cef,ascending=False)

def corr_list_sorted(sorted_corr_df):
    """
    @returns sorted max->min of correlations with fund
    """
    return sorted_corr_df.values[0].tolist()

def get_top_corr(sorted_corr_df, num, cef):
    """
    @returns DICT of top (num) correlated assets in fund
    """
    top = {}
    i=0
    for row in sorted_corr_df.itertuples():
        if (row.Index == cef):
            continue
        i+=1
        top[row.Index] = row[1] #CEF column
        if i==num: return top
    
def get_data(tup):
    """
    @returns np.array of price changes given tuple of the form: (ticker string, correlation value, dictionary ticker is in)
    """
    dic = tup[2]
    tck = tup[0]
    return dic[tck]

""" LINEAR REGRESSION FUNCTIONS """

def lin_reg(X, Y, lam=None):
    d, n = X.shape #Y = (n,1)
    
    aug = np.array([1]*n).T # (n,)
    Xaug = np.vstack((X, aug)) # (topnum+1, n)
    
    addon = 0
    if lam is not None:
        addon = n*lam*np.identity(d+1)
    a = np.linalg.inv( np.dot(Xaug, Xaug.T)  + addon)
    b = np.dot(Xaug, Y)
    th = np.dot(a,b) #d+1 x1
    return th[:d], th[d][0]

def mse(X_test, Y_test, th, th0, lam):
    Y_hat = th.T@X_test + th0 # 1 x n
    return  np.mean((Y_test.T-Y_hat)**2) + lam*(np.linalg.norm(th))

def make_splits(X, Y, n_splits):
    """
    X: (d, n)
    Y: (n, 1)
    """
    d, n = X.shape
    subsetsize = math.ceil(n/n_splits)
    splits = []
    for i in range(n_splits):
        X_test = X[: , i*subsetsize : (i+1)*subsetsize] # (d, s)
        Y_test = Y[i*subsetsize : (i+1)*subsetsize, :] # (s, 1)
        
        X_sub = np.concatenate((X[:,:i*subsetsize], X[:,(i+1)*subsetsize:]), axis=1) #(d, n-s)
        Y_sub = np.concatenate((Y[:i*subsetsize,:], Y[(i+1)*subsetsize:,:]), axis=0) #(n-s, 1)
        splits.append((X_sub, Y_sub, X_test, Y_test))
    return splits

def cross_validate(X, Y, n_splits, lam,
                   learning_algorithm, loss_function):
    splits = make_splits(X, Y, n_splits)
    total_e = 0
    for i in range(n_splits):
        X_subset = splits[i][0]
        Y_subset = splits[i][1]
        X_test = splits[i][2]
        Y_test = splits[i][3]
        th, th0 = learning_algorithm(X_subset, Y_subset, lam)
        l = loss_function(X_test, Y_test, th, th0, lam)
        total_e += l
    return total_e/n_splits