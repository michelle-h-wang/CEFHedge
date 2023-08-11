from display.engine import bq, bql, ResponseError
import ipywidgets as widgets
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy import linalg as linalg

from display.constants import WINDOW, TOPNUM
from display.factor import (
    get_members,
    get_NAV_chg,
    get_pct_chg,
    all_changes,
    corr_matrix,
    corr_list_sorted,
    get_top_corr,
    get_data,
    get_daily_px,
    lin_reg,
    mse,
    make_splits,
    cross_validate
)

plt.rcParams.update({'font.size': 20})
plt.style.use('dark_background')

class Fund():
    def __init__(self, CEF_TCK, start, end):
        self.CEF_TCK = CEF_TCK
        self.startdate = start
        self.enddate = end
        
        self.cef_list = None
        self.cef_pct_chg_df = None
        self.cef_pct_chg = None
        self.chg_dict = None
        self.corr_df= None 
        self.correct_len = None
        self.top_h = None
        
        
    def _cef_list(self): 
        """
        @return & initializes list of holdings in CEF, as bbg tickers
        """
        if self.cef_list is None:
            self.cef_list = get_members(bq.univ.holdings(self.CEF_TCK))
        return self.cef_list
    
    def _cef_pct_chg_df(self):
        """
        @return & initializes pd.Dataframe of CEF NAV % daily change values
        """
        if self.cef_pct_chg_df is None:
            self.cef_pct_chg_df = get_NAV_chg(self.CEF_TCK, self.startdate, self.enddate)
        return self.cef_pct_chg_df
        
    def _cef_pct_chg(self):
        """
        @return & initializes np.array of CEF NAV % daily change values
        """
        if self.cef_pct_chg is None:
            df = self._cef_pct_chg_df()
            self.cef_pct_chg = df["PCT_DIFF(FUND_NET_ASSET_VAL(dates=RANGE(start=" + self.startdate + ",end=" + self.enddate + "),currency='USD'))"].values
            self.correct_len = self.cef_pct_chg.shape[0]
        return self.cef_pct_chg
    
    def length(self):
        """
        @return & initializes int: correct len data should be
        """
        if self.correct_len is None:
            self.correct_len = self._cef_pct_chg().shape[0]
        return self.correct_len
    
    def _chg_dict(self): #dict
        """
        @return & initializes dict of each holding and its corresponding price changes
        """
        if self.chg_dict is None:
            cef_list = self._cef_list()
            cef_pct_chg = self._cef_pct_chg()
            self.chg_dict = all_changes(cef_list, {self.CEF_TCK: cef_pct_chg}, self.length(), self.startdate, self.enddate)
        return self.chg_dict
    
    def _corr_df(self):
        """
        @return & initializes correlation matrix between CEF and holdings
        """
        if self.corr_df is None:
            chg = self._chg_dict()
            chg_df = pd.DataFrame(data=chg)
            self.corr_df = corr_matrix(chg_df, self.CEF_TCK)
        return self.corr_df
    
    def _top_corr(self, topnum):
        """
        @return & initializes {topnum} correlated holdings as dict
        """
        if self.top_h is None:
            self.top_h = get_top_corr(self._corr_df(), topnum, self.CEF_TCK)
        return self.top_h
              
    def display_fund_assets(self):
        """
        @returns displayable section of datafram
        """
        s = 'FUND HOLDINGS \n'
        for h in self._cef_list():
            s += h + '\n'
        s += ' \n total ' + str(len(self.cef_list))
        return s

    
    def plot_nav_chg(self):
        """
        @returns plt figure of CEF NAV change plotted
        """
        fig = plt.figure(figsize=(WINDOW,WINDOW))
        plt.plot(self._cef_pct_chg_df()['DATE'], self._cef_pct_chg())
        plt.xlabel('dates')
        plt.tick_params(axis='x', rotation=70)
        plt.ylabel('daily NAV change (%)')
        plt.title(self.CEF_TCK + " daily change in NAV")
        plt.show()
        return fig
        
        
    def plot_indiv_price_chg(self):
        """
        @returns plt figure of each holding's price changes plotted
        """
        plt.rcParams.update({'font.size': 4})
        x = math.ceil(len(self._cef_list())**.5)
        fig = plt.figure(figsize=(x*4,x*4))
        i=1
        for tck in self._chg_dict().keys():
            if self._chg_dict()[tck].shape[0] != self.correct_len:
                continue
            plt.subplot(x,x,i)
            i +=1
            plt.plot(self._cef_pct_chg_df()['DATE'] ,self._chg_dict()[tck])
            plt.tick_params(axis='x', rotation=70)
            plt.title(tck)
            
        plt.show()
        plt.rcParams.update({'font.size': 12})
        return fig
    
    def plot_heatmap(self):
        """
        plots heatmap viz of correlations
        """
        fig = plt.figure(figsize=(WINDOW, WINDOW)).set_dpi(150)
        sns.heatmap(self.corr_df)
        plt.show()
        return fig
    

        

""" US EQUITIES WITH LIQUIDITY > $1M """
class Eq():
    def __init__(self):
        eq = bq.univ.filter(bq.univ.filter(bq.univ.equitiesuniv(['PRIMARY', 'ACTIVE']), bq.data.cntry_of_risk() == 'US'), 
                    bq.func.avg(bq.func.dropna(bq.data.turnover(dates=bq.func.range('-20d', '0d')))) >= 1000000)
        self.univ = bq.univ.filter(eq, bq.data.security_typ() == 'Common Stock')
        self.eq = get_members(self.univ)
        
    def refresh(self):
        """ @returns NONE
        """
        self.eq = get_members(self.univ)
        
    def display(self):
        """
        @ returns displayable section of datafram
        """
        s = ''
        for eq in self.eq:
            s += '<div>' + eq + '</div>' 
        return s
        


""" US ETFs WITH TOTAL ASSETS > $150M """
class Etf():
    def __init__(self):
        self.univ1 = bq.univ.filter(bq.univ.filter(bq.univ.fundsuniv(['active', 'primary']), bq.data.fund_typ() == 'ETF'), bq.func.between(bq.data.fund_total_assets()['value'],'150M','500M'))
        self.univ2 =  bq.univ.filter(bq.univ.filter(bq.univ.fundsuniv(['active', 'primary']), bq.data.fund_typ() == 'ETF'), bq.data.fund_total_assets() > '500M')
        self.etfs1 = get_members(self.univ1)
        self.etfs2 = get_members(self.univ2)
        
        
    def refresh(self):
        """@returns NONE
        """
        self.etfs1 = get_members(self.univ1)
        self.etfs2 = get_members(self.univ2)
    
    def display(self):
        """
        @ returns displayable section of datafram
        """
        s = ''
        for etf in self.etfs1:
            s += '<div>' + etf + '</div>' 
        for etf in self.etfs2:
            s += '<div>' + etf + '</div>' 
        return s
                                         
                                         