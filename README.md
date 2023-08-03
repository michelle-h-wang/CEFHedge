# CEFHedge

Main code is found in main.ipynb. This file searches through an established universe. 
UNIV ={ETFs in the US, Equities with market cap >150M in the US, and assets in the fund itself}.

ONLY CHANGE the following constants under "global constants" cell. 
    - "startdate", "enddate" should be in the following format: "-3M", "0D", "4Y" etc. The following will not work: "-3m" "0" "4years"
    - "CEF_TCK" can be changed to any bbg ticker, it must be a fund, and must be in string format (in quotations)
    - "TOPNUM" refers to how many assets you ultimately want in the model. If TOPNUM=3, the algo will find the top 3 tickers in the described universe above for the highest 3 correlations to the fund's NAV changes.
    - "WINDOW" refers to the window size of the plots (in inches). Ultimately affects how large the plot will be.
    - "lams" is a list of different regularization parameters to be used during linear regression. Feel free to add/remove as you see fit, but always keep 0 in there. Can only add positive numbers (int/float).
    - "n_splits" is how many different splits of the data to be used during cross-validation of the linear regression model. Any number works here, except 0, 1.
