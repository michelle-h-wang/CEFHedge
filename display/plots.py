import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from display.model import get_pct_chg

def corr_heat_map(df):
    return sns.heatmap(df, annot=True, cmap='RdBu')

def scatter_corr_plot(ticker):
    
    req = bql.Request(ticker, get_pct_chg)
    res = bq.execute(req)
    data = res[0].df()
    data.fillna(0, inplace = True)
    for row in data.itertuples():
        plt.scatter(row[0], row[1])
    