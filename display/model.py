import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from display.engine import bq
from display.universe import (eq_tck_list,)

""" HELPER FUNCTIONS """
get_pct_chg = bq.func.pct_diff(bq.data.px_last(dates=bq.func.range('-3y', '0d'), fill='prev')) #get request
str_rep = "PCT_DIFF(PX_LAST(fill='prev',dates=RANGE(-3Y,0D)))"

def get_members(univ):
    """
    @returns: list of tickers of assets in fund
    """
    req = bql.Request(univ, bq.data.id()['value'])
    res = bq.execute(req)
    return res[0].df().values.T[0].tolist()

def get_pct_chg(ticker):
    """
    @returns: numpy array (n x 1) of daily % change price of input ticker (n = # of datapts)
    """
    req = bql.Request(ticker, get_pct_chg)
    res = bq.execute(req)
    data = res[0].df()
    data.fillna(0, inplace = True)
    return data[str_rep].values

def all_changes(ticker_list, dfdict, correct_len):
    """
    @returns DICT, modifies dictionary with %chg data for each ticker in asset
    """
    for tck in ticker_list:
        pct_chg = get_pct_chg(tck)[str_rep].values
        if pct_chg.shape[0] == correct_len:
            dfdict[tck] = pct_chg
    return dfdict

def corr_matrix(df):
    """ 
    @returns DF correlation matrix from input data dataframe (sorted in max -> min order)
    """
    corr = df.corr()
    corr.fillna(0, inplace= True)
    return corr.sort_values(by=self.cef,ascending=False)

def corr_list_sorted(sorted_corr_df):
    """
    @returns sorted max->min of correlations with fund
    """
    return sorted_corr_df.values[0].tolist()

def get_top_corr(sorted_corr_df, num):
    """
    @returns DICT of top (num) correlated assets in fund
    """
    top = {}
    i=0
    for row in sorted_corr_df.itertuples():
        i+=1
        top[row.Index] = row[1] #CEF column
        if i==num: return top
    
# def get_top_num_tickers(all_top, topnum):
#     sortedtop = [(k,v) for k, v in sorted(all_top.items(), key=lambda item: item[1], reverse=True)]
#     X = chg_dict[sortedtop[0][0]] #n x topnum
#     Y = cef_pct_chg #nx1
#     for i in range(1,TOPNUM):
#         X.vstack(chg_dict[sortedtop[i][0]])
    
    
def run_lin_reg(X, Y, reg = None):
    """
    @returns #topnum x1 matrix of linear regression coefficients (th)
    """
    if reg == None:
        addon = 0
    else: 
        addon = X.shape[0]*reg*np.identity(X.shape[1])
        
    a = np.linalg.inv(np.dot(X.T, X) + addon) 
    b = np.dot(X.T, Y)
    th = np.dot(a,b) #topnum x1
    return th
    
def calc_mse(th, X, Y, reg = None):
    """
    @returns mean squared error given coeff from linear regression
    """
    if reg==None:
        addon = 0
    else:
        addon = np.linalg.norm(th)*reg
    
    Y_hat = th.T@X #1 x n
    mse = np.mean((Y.T-Y_hat)**2) + addon
    return mse
    
    


""" CLASSES """

class CEFAssets():
    """
    construct values for given CEF TICKER: members list, price change arr, members' price change arr, correlation dataframe
    """
    def __init__(self, cef):
        self.cef = cef
        self.members = get_members(bq.univ.holdings(cef))
        self.pctChgArr = get_pct_chg(cef)
        self.len = self.pctChgArr.shape[0]
        self.allChg = all_changes(self.members, {cef: self.pctChgArr, }, self.len)
        self.corrDF = corr_matrix(pd.DataFrame(data=self.allChg))

        

class Equities():
    def __init__(self):
        self.members = eq_tck_list
        
    def get_segment_corr(segment_num):
        
            
            
    
        
        