import ipywidgets as widgets
from ipydatagrid import DataGrid
from IPython.display import display
from bqwidgets import TickerAutoComplete
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from numpy import linalg as linalg

from display.colors import BLUE, WHITE
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
from display.universe import Fund, Eq, Etf
from display.engine import bq, bql, ResponseError
from display.spinner import Spinner
from display.logger import ApplicationLogger

plt.rcParams.update({'font.size': 20})
plt.style.use('dark_background')

LAYOUT_DICT = {
    'display': 'flex',
    'max_height': '75px',
    'max_width': '600px',
    'overflow_y': 'auto',
    'margin': '20px 0px 20px 0px',
    'border': '1px solid #505050',
    'padding': '8px 5px 8px 8px'
}

MAIN = '800px'
BUTTON = '200px'
SMALLBUTTON = '120px'
RIGHTPANEL = '760px'
LEFTPANEL = '240px'


class FundAutoComplete(TickerAutoComplete):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            yellow_keys=['Fund'],
            continuous_update=False,
            max_results=5,
            **kwargs
        )
class FundSelector(FundAutoComplete):
    def __init__(self, cef=None):
        super().__init__(
            value=cef,
            layout={
                'margin': '2px 0px 0px 0px',
                'width': '175px',
            }
        )
        


class MainDashboard():
    
    def __init__(self):
        self.ETFS = None
        self.EQS = None
        self.cef = None
        self._build_widgets()
        self._build_control_panel()
        
        
        self.app = widgets.VBox(
            [
                widgets.HBox(
                [
                    self.widgets['left_panel'],
                    widgets.VBox([
                        self.widgets['input_fund_controls'],
                        self.widgets['fund_analysis']
                    ]),
                    self.widgets['right_panel']
                ], layout = {'width': '100%' }), 
                widgets.Label("Error log:"),
                self._logger.get_widget()
            ]
        )
        
    def _build_widgets(self, ):
        """
        Build all the application's component widgets.
        """
        self.widgets = {}
        
        self.widgets['fund_input'] = widgets.Text(
            value='BMEZ US EQUITY',
            placeholder='BMEZ US EQUITY',
            description= 'Enter Fund Ticker:',
            disabled =False,
            layout={
                'margin': '14px 0px 0px 0px',
                'width': '100%'
            },
            style = {'description_width': 'initial'}
        )
        # self.widgets['fund_input'].observe(self._update)
        
        self.widgets['start_date'] = widgets.Text(
            value='-3M',
            placeholder='ex: -3M',
            description= 'Enter Start Date:',
            disabled =False,
            layout={
                'margin': '14px 12px 0px 0px',
                'width': '50%'
            },
            style = {'description_width': 'initial'}
        )
        self.widgets['end_date'] = widgets.Text(
            value='0D',
            placeholder='ex: 0D',
            description= 'Enter End Date:',
            disabled =False,
            layout={
                'margin': '14px 0px 0px 12px',
                'width': '50%'
            },
            style = {'description_width': 'initial'}
        )
        
        self.widgets['TOPNUM'] = widgets.IntSlider(
            value = 3,
            min = 1,
            max = 15,
            step = 1,
            description='Number assets (model):',
            disabled = False,
            continuous_update = False,
            orientation='horizontal',
            readout = True,
            readout_format='d',
            layout={
                'margin': '14px 0px 0px 0px',
                'width': '100%'
            },
            style = {'description_width': 'initial'}
        )
        
        self.widgets['epsilon'] = widgets.IntText(
            value = 5,
            description= "Epsilon (as %)",
            disabled = False,
            layout={
                'margin': '14px 0px 0px 0px',
                'width': '100%'
            },
            style = {'description_width': 'initial'}
        )
        
        self.widgets['n_splits'] = widgets.IntSlider(
            value = 10,
            min = 5,
            max = 15,
            step = 1,
            description= "Number of splits:",
            disabled = False,
            continuous_update = False,
            orientation='horizontal',
            readout = True,
            readout_format='d',
            layout={
                'margin': '14px 0px 0px 0px',
                'width': '100%'
            },
            style = {'description_width': 'initial'}
        )
        
        # right panel
        self.widgets['update_corr_plot_holding'] = widgets.Button(
            description = 'Update top holdings',
            layout={
                'margin': '12px 24px 24px 12px',
                'width': BUTTON
            }
        )
        self.widgets['update_corr_plot_holding'].on_click(self._update_holding_corr_plot)
        self.widgets['update_corr_plot_holding'].style.button_color = BLUE
        
        self.widgets['update_corr_plot_eq'] = widgets.Button(
            description = 'Update top equities',
            layout={
                'margin': '12px 24px 24px 12px',
                'width': BUTTON
            }
        )
        self.widgets['update_corr_plot_eq'].on_click(self._update_eq_corr_plot)
        self.widgets['update_corr_plot_eq'].style.button_color = BLUE
        
        self.widgets['update_corr_plot_etf'] = widgets.Button(
            description = 'Update top ETFs',
            layout={
                'margin': '12px 24px 24px 12px',
                'width': BUTTON
            }
        )
        
        self.widgets['update_corr_plot_etf'].on_click(self._update_etf_corr_plot)
        self.widgets['update_corr_plot_etf'].style.button_color = BLUE
        # left panel
        self.widgets['get_etf_univ'] = widgets.Button(
            description = 'Pull ETFs Universe',
            layout={
                'margin': '12px 24px 24px 12px',
                'width': BUTTON
            }
        )
        self.widgets['get_etf_univ'].on_click(self._update_etf)
        self.widgets['get_etf_univ'].style.button_color = BLUE
        
        self.widgets['get_eq_univ'] = widgets.Button(
            description = 'Pull Equities Universe',
            layout={
                'margin': '12px 24px 24px 12px',
                'width': BUTTON
            }
        )
        self.widgets['get_eq_univ'].on_click(self._update_eq)
        self.widgets['get_eq_univ'].style.button_color = BLUE
        
        # main panel, plots
        self.widgets['fund_plots'] = widgets.VBox(layout = {'width': MAIN})
        # self.widgets['yhat-y-plot'] = widgets.VBox()
        # self.widgets['error-plot'] = widgets.VBox()
        self.widgets['model'] = widgets.VBox()
        
        # right panel
        self.widgets['top_holding_plot'] = widgets.VBox()
        self.widgets['top_eq_plot'] = widgets.Box()
        self.widgets['top_etf_plot']= widgets.Box()
        
        # left panel
        self.widgets['etf_dis'] = widgets.VBox(
            layout={'margin': '12px 24px 24px 12px','height': '500px'}
        )
        self.widgets['eq_dis'] = widgets.VBox(
            layout={'margin': '12px 24px 24px 12px','height': '500px'}
        )
        self.widgets['reg_plot'] = widgets.VBox()
        
        self.widgets['update_main_plot'] = widgets.Button(
            description = 'Update holdings',
            layout={
                'margin': '14px 0px 0px 0px',
                'width': SMALLBUTTON
            }
        )
        
        self.widgets['update_main_plot'].on_click(self._update_cef)
        self.widgets['update_main_plot'].style.button_color = BLUE
        
        self.widgets['get_model_button'] = widgets.Button(
            description = 'Calculate model',
            layout={
                'margin': '14px 0px 0px 0px',
                'width': SMALLBUTTON
            }
        )
        
        self.widgets['get_model_button'].on_click(self.get_model)
        self.widgets['get_model_button'].style.button_color = BLUE
        
        self._spinner = Spinner(
            display=False,
            size='16px',
            layout_dict={'margin': '0px 0px 0px 20px'}
        )
        
        self.eq_spinner = Spinner(
            display=False,
            size='16px',
            layout_dict={'margin': '0px 0px 0px 20px'}
        )
        
        self.etf_spinner = Spinner(
            display=False,
            size='16px',
            layout_dict={'margin': '0px 0px 0px 20px'}
        )
        
        self.toph_spinner = Spinner(
            display=False,
            size='16px',
            layout_dict={'margin': '0px 0px 0px 20px'}
        )
        
        self.topeq_spinner = Spinner(
            display=False,
            size='16px',
            layout_dict={'margin': '0px 0px 0px 20px'}
        )
        
        self.topetf_spinner = Spinner(
            display=False,
            size='16px',
            layout_dict={'margin': '0px 0px 0px 20px'}
        )
        
        self.model_spinner = Spinner(
            display=False,
            size='16px',
            layout_dict={'margin': '0px 0px 0px 20px'}
        )
        # to display messages to console        
        self._logger = ApplicationLogger()

        
    def _build_control_panel(self):
        #input control box
        self.widgets['input_fund_controls'] = widgets.Tab(
            (
                widgets.VBox(
                [
                    widgets.Label(value='INPUT FUND CONTROLS'),
                    self.widgets['fund_input'], 
                    widgets.HBox(
                        [self.widgets['start_date'], self.widgets['end_date']]),
                    self.widgets['TOPNUM'], 
                    self.widgets['n_splits'],
                    self.widgets['epsilon'],
                    widgets.HBox([self.widgets['update_main_plot'], self._spinner])
                ]),
                widgets.VBox([
                    widgets.HTML("UNIV ={ETFs in the US with total assets >= 150M, Equities with 20d traded value >= 1M in the US, and assets in the fund itself}."),
                    widgets.HTML("- \'start date\', \'end date\' should be in the following format: \"-3M\", \"0D\", \"2020-03-28\" etc. The following will not work: \"-3m\" \"0\" \"4years\". "),
                    widgets.HTML("- Fund Ticker can be changed to any bbg ticker, it must be a fund."),
                    widgets.HTML("- \"how many assets to use\" refers to how many assets you ultimately want in the model. If num=3, the algo will find the top 3 tickers in the universe above for the highest 3 correlations to the fund's NAV changes."),
                    widgets.HTML("- \"Number of cross-validation splits\" is how many different splits of the data to be used during cross-validation of the linear regression model. Any number works here, except 0, 1, default value 10.")
                ])
                
            ), layout={'margin': '0px 12px 12px 0px', 'width': MAIN}
        )
        
        self.widgets['input_fund_controls'].set_title(0, "Inputs")
        self.widgets['input_fund_controls'].set_title(1, "README")
        
        # right panel
        self.widgets['top_holding'] = widgets.VBox(
            [
                widgets.Label(value = f'Top {TOPNUM} correlations:', layout ={'margin': '0px 4px 0px 0px'}), 
                self.widgets['update_corr_plot_holding'], 
                self.toph_spinner, 
                self.widgets['top_holding_plot']
            ],
            layout={'margin': '0px 12px 0px 0px'}
        )
        self.widgets['top_eq'] = widgets.VBox(
            [
                widgets.Label(value = f'Top {TOPNUM} correlations:', layout ={'margin': '0px 4px 0px 0px'}), 
                self.widgets['update_corr_plot_eq'], 
                self.topeq_spinner, 
                self.widgets['top_eq_plot']
            ],
            layout={'margin': '0px 12px 0px 0px'}
        )
        self.widgets['top_etf'] = widgets.VBox(
            [
                widgets.Label(value = f'Top {TOPNUM} correlations:', layout ={'margin': '0px 4px 0px 0px'}), 
                self.widgets['update_corr_plot_etf'], 
                self.topetf_spinner, 
                self.widgets['top_etf_plot']
            ],
            layout={'margin': '0px 12px 0px 0px'}
        )
        
        self.widgets['right_panel'] = widgets.Tab(
            (
                self.widgets['top_holding'], 
                 self.widgets['top_eq'], 
                 self.widgets['top_etf']
            ),
            layout = {'margin': '0px 0px 0px 12px', 'width': RIGHTPANEL}
        )
        self.widgets['right_panel'].set_title(0, "Holdings")
        self.widgets['right_panel'].set_title(1, "Equities")
        self.widgets['right_panel'].set_title(2, "ETFs")
        
        # left_panel
        lams = widgets.HTML('λ = [0, 0.00001, 0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3]')
        self.widgets['eq_panel'] = widgets.VBox(
            [
                widgets.Label('EQUITIES IN US WITH LIQUIDITY OVER PAST 20 DAYS > 1M'),
                widgets.VBox([self.widgets['get_eq_univ'], self.eq_spinner]),
                self.widgets['eq_dis']
            ], layout={'margin': '0px 12px 0px 0px', 'height': '600px'})
        
        self.widgets['etf_panel'] = widgets.VBox(
            [
                widgets.Label('ETFs IN THE US WITH TOTAL ASSET VALUE > 150M'),
                widgets.VBox([self.widgets['get_etf_univ'], self.etf_spinner]),
                self.widgets['etf_dis']
            ], layout={'margin': '0px 12px 0px 0px', 'height': '600px'})
        
        
        self.widgets['left_panel'] = widgets.VBox(
            [
                self.widgets['eq_panel'],
                self.widgets['etf_panel'],
                widgets.Label(value = 'Regularization'),
                lams,
                self.widgets['reg_plot']
                
            ],
            layout={'margin': '0px 12px 0px 0px', 'width': LEFTPANEL }
        )
        
        # main panel, plots
        self.widgets['fund_analysis'] = widgets.VBox(
        [
            self.widgets['fund_plots'],
            self.widgets['model'],
        ], layout={'margin': '0px 12px 12px 0px', 'width':  MAIN}
        )
        
    """ MAIN PANEL"""
    def _update_cef(self, *args):
        try:
            self._spinner.text = "Fetching data..."
            self._spinner.start()
            
            self.start_date = self.widgets['start_date'].value
            self.end_date = self.widgets['end_date'].value
            self.cef = Fund(self.widgets['fund_input'].value, self.start_date, self.end_date)
            self.topnum = self.widgets['TOPNUM'].value
            self.n_splits = self.widgets['n_splits'].value
            self.eps = self.widgets['epsilon'].value/100
            
            # draw app
            self._spinner.text = "Building visualizations..."
            
            nav_plot = widgets.Output()
            with nav_plot:
                nav_plot.clear_output()
                self.cef.plot_nav_chg()
            hold_plot = widgets.Output()
            with hold_plot:
                hold_plot.clear_output()
                self.cef.plot_indiv_price_chg()
                
            self.widgets['fund_plots'].children = [
                widgets.Label("CEF NAV CHANGES PLOT:"),
                nav_plot,
                widgets.Label("HOLDINGS PRICE CHANGES:"),
                hold_plot,
                self.widgets['get_model_button'], 
                self.model_spinner
            ]
            self._spinner.stop()
            #clear right panel, model
            self.widgets['top_holding_plot'].children = []
            self.widgets['top_eq_plot'].children = []
            self.widgets['top_etf_plot'].children = []
            self.widgets['model'].children = []
            self.widgets['reg_plot'].children = []

        except ResponseError as e:
            self._logger.log_message('No Valid Data, check inputs', color='red')
            self._logger.log_message(e, color='red')
            self._spinner.stop()
    
    def gd_XY(self, sortedtop):
        """
        @returns X,Y matrices for gradient descent (model method 2)
        """
        X_gd = get_daily_px(sortedtop[0][0], self.start_date, self.end_date) #(d,n)
        seen = [sortedtop[0][0]]
        n = X_gd.shape[0]
        
        i=1
        for tup in sortedtop:
            if tup[0] in seen: continue
            else:
                i += 1
                arr = get_daily_px(tup[0], self.start_date, self.end_date)
                arr = np.resize(arr, (n,1)).T
                X_gd = np.vstack((arr, X_gd))
                seen.append(tup[0])
                if i == self.topnum:
                    break
           
        X2 = (X_gd[:,1:])

        X1 = X_gd *(1+ np.resize(self.cef._cef_pct_chg(), (n,1)).T)
        X1 = X1[:,:-1]        

        Xaug = X1 - X2
        
        # GET ORIGINAL FUND NAV VALUES
        req = bql.Request(self.cef.CEF_TCK, bq.data.fund_net_asset_val(dates=bq.func.range(start=self.start_date , end=self.end_date), fill='prev', currency='USD'))
        res = bq.execute(req)
        NAV = res[0].df()["FUND_NET_ASSET_VAL(fill='prev',dates=RANGE(start=" + self.start_date + ",end=" + self.end_date + "),currency='USD')"].values
        NAV = np.resize(NAV, (n, 1)) #nx1
        return Xaug, NAV, X_gd # (TOPNUM, n), (n, 1), (TOPNUM, n)
    
    def linreg_XY(self, sortedtop):
        """
        @returns X,Y matrices for linear regression model (model method 1)
        """
        X = get_data(sortedtop[0]) # (n,)
        n = X.shape[0]
        seen = [sortedtop[0][0]]
        i=1
        
        for tup in sortedtop:
            if tup[0] in seen:
                continue
            else:
                i+=1
                X = np.vstack((X, get_data(tup))) # (d, n)
                seen.append(tup[0])
                if i == self.topnum:
                    break
         
        Y = np.resize(self.cef._cef_pct_chg(), (n, 1)) #nx1
        return X,Y
    
    def grad_desc(self, X, lam, eps):
        """
        @returns: np.array of size(TOPNUM,1) containing optimal coefficients gradient descent initialized to (1,1,..), no offset.
        """
        thnew = np.ones((self.topnum,1)) #*(1/TOPNUM)
        obj = float('inf')
        
        i=0
        while obj > eps:
            i+=1
            th = thnew
            gd = 2/self.n * X@(th.T@X).T + 2*lam*th
            
            thnew = th - (1/i)*(gd/np.sum(gd))
            if np.any(thnew < 0):
                thnew = th
                break
            thnew /= (np.sum((thnew)))
            obj = (np.mean((th.T@X)**2) - np.mean((thnew.T@X)**2))/np.mean((th.T@X)**2)
            print(f'step {i}: \n th_init = {th.T} \n gradient descent = {gd.T/np.sum(gd)} \n error decreased by = {obj}')
        return thnew
    
    def get_model(self, *args):
        if not self.widgets['top_etf_plot'].children or not self.widgets['top_eq_plot'].children or not self.widgets['top_holding_plot'].children:
            self._logger.log_message('Pull top correlations first', color='red')
            return
        
        self.model_spinner.text = "Calculating model..."
        self.model_spinner.start()
        
        self.lams = [0, 0.00001, 0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3]
        alltop = [(k,v, self.chg_h) for k, v in self.top_h.items()]
        eqs = [(k,v, self.chg_eq) for k, v in self.top_eq.items()]
        etf = [(k,v, self.chg_etf) for k, v in self.top_etf.items()]
        alltop.extend(eqs)
        alltop.extend(etf)
        sortedtop = [tup for tup in sorted(alltop, key=lambda item: item[1], reverse=True)]
        
        ####################################################################
        # LINEAR REGRESSION MODEL
        # build X, Y matrices for linear regression
        self.lr_X, self.lr_Y = self.linreg_XY(sortedtop)
        self.n = self.lr_X.shape[0]
        # cross - validate, find optimal lambda, coefficients
        
        lr_errors = []
        self.lr_err, self.lr_lam, self.lr_th = float('inf'), None, None
        for lam in self.lams:
            th = lin_reg(self.lr_X, self.lr_Y, lam)
            err = cross_validate(self.lr_X, self.lr_Y, self.n_splits, lam, lin_reg, mse)
            lr_errors.append(err)
            if err < self.lr_err:
                self.lr_err, self.lr_lam, self.lr_th = err, lam, th
        
        opt_lr_reg = f'<div> optimal lambda: {str(self.lr_lam)} </div> <div> min testing error: {str(self.lr_err)} </div> <div> linear regression coefficients: </div> <div> {np.array2string(self.lr_th)} </div>'
        # model
        self.lr_Y_hat = self.lr_th.T@self.lr_X
        self.linreg_model = str(round(self.lr_th[0][0], 2)) + '  ' + sortedtop[0][0]
        for i in range(1,self.topnum):
            self.linreg_model +=  ' + ' + str(round(self.lr_th[i][0],2)) + '  ' + sortedtop[i][0] 
        
        #plots
        lr_reg_plot, lr_yhat_plot, lr_err_plot = widgets.Output(), widgets.Output(), widgets.Output()
        with lr_reg_plot:
            lr_reg_plot.clear_output()
            self.plot_cross_valid(self.lams, lr_errors)
            
        with lr_yhat_plot:
            lr_yhat_plot.clear_output()
            self._update_LR_yhat_plot()
            
        with lr_err_plot:
            lr_err_plot.clear_output()
            self._update_LR_err_plot()
        
        ####################################################################    
        # GRADIENT DESCENT
        self.X_gd, self.NAV, self.X_gd_orig = self.gd_XY(sortedtop)
        #choose reg
        self.gd_lam, self.gd_th, self.gd_err = None, None, float('inf')
        gd_errors = []
        for lam in self.lams:
            th = self.grad_desc(self.X_gd, lam, self.eps)
            err = np.mean((th.T@self.X_gd)**2)
            gd_errors.append(err)
            if err <= self.gd_err:
                self.gd_lam, self.gd_th, self.gd_err = lam, th, err
        opt_gd_reg = f'<div> optimal lambda: {str(self.gd_lam)} </div> <div> min testing error: {str(self.gd_err)} </div> <div> linear regression coefficients: </div> <div> {np.array2string(self.gd_th)} </div>'
        self.gd_Y_hat = self.gd_th.T@self.X_gd
        self.gd_Y_hat_chg = (self.gd_th.T@self.X_gd_orig[:,1:] - self.gd_th.T@self.X_gd_orig[:,:-1])/(self.gd_th.T@self.X_gd_orig[:,:-1])
        self.gd_model = str(round(self.gd_th[0][0], 2)) + '  ' + sortedtop[0][0]
        for i in range(1,self.topnum):
            self.gd_model +=  ' + ' + str(round(self.gd_th[i][0],2)) + '  ' + sortedtop[i][0] 
        #plots
        gd_reg_plot, gd_yhat_plot, gd_err_plot = widgets.Output(), widgets.Output(), widgets.Output()
        with gd_reg_plot:
            gd_reg_plot.clear_output()
            self.plot_gd_reg(self.lams, gd_errors)
            
        with gd_yhat_plot:
            gd_yhat_plot.clear_output()
            self._update_gd_yhat_plot()
            
        with gd_err_plot:
            gd_err_plot.clear_output()
            self._update_gd_err_plot()
        ########################################################################
        self.widgets['linreg-model'] = widgets.VBox(
            [widgets.Label('METHOD 1 HEDGE BASKET: '), widgets.HTML(self.linreg_model), lr_yhat_plot, lr_err_plot], layout ={'border': 'solid'})
        
        self.widgets['gd-model'] = widgets.VBox([
            widgets.Label('METHOD 2 HEDGE BASKET: '), widgets.HTML(self.gd_model), gd_yhat_plot, gd_err_plot
        ], layout = {'border': 'solid'})
        
        self.widgets['reg_plot'].children = [widgets.Label('LINEAR REGRESSION MODEL'), lr_reg_plot, widgets.HTML(opt_lr_reg), widgets.Label('GRADIENT DESCENT MODEL'), gd_reg_plot, widgets.HTML(opt_gd_reg)]
        self.widgets['model'].children = [self.widgets['linreg-model'], self.widgets['gd-model']]
        self.model_spinner.stop()

    def plot_cross_valid(self, lams, errors):
        fig = plt.figure(figsize=(WINDOW, WINDOW))
        plt.title("Cross-Validation Error from Linear Regression with Different Lambdas", fontweight="bold")
        plt.xlabel("lamda")
        plt.ylabel("Mean Squared Error")
        plt.plot(lams, errors)
        plt.show()
        return fig
        
    def _update_LR_yhat_plot(self):
        fig = plt.figure(figsize=(WINDOW, WINDOW))
        plt.title("Modeled NAV Changes vs. Actual NAV Changes", fontweight="bold")
        plt.text(.2, .6, 'MSE: ' + str(self.lr_err) + '\nRegularization: '+ str(self.lr_lam) + '\n \nmodel: \n' + self.linreg_model,transform=plt.gca().transAxes)
        plt.plot(self.cef._cef_pct_chg_df()['DATE'], self.lr_Y_hat.T, label = "Modeled NAV")
        plt.plot(self.cef._cef_pct_chg_df()['DATE'], self.lr_Y, label = "Actual NAV")
        plt.legend()
        plt.tick_params(axis='x', rotation=70)
        plt.show()
        return fig
    
    def _update_LR_err_plot(self):
        fig = plt.figure(figsize=(WINDOW, WINDOW))
        plt.title("Model Error", fontweight="bold")
        plt.text(.2, .6, 'MSE: ' + str(self.lr_err) + '\nRegularization: '+ str(self.lr_lam) + '\n \nmodel: \n' + self.linreg_model,transform=plt.gca().transAxes)
        e = self.lr_Y_hat.T-self.lr_Y
        plt.tick_params(axis='x', rotation=70)
        plt.plot(self.cef._cef_pct_chg_df()['DATE'], e)
        plt.show()
        return fig

    def plot_gd_reg(self, lams, errors):
        fig=plt.figure(figsize=(WINDOW, WINDOW))
        plt.title("Error from Gradient Descent for each Regularization Hyperparameter")
        plt.xlabel("lambda")
        plt.ylabel("Variance")
        plt.plot(lams, errors)
        plt.show()
        return fig
        
    def _update_gd_yhat_plot(self):
        fig=plt.figure(figsize=(WINDOW, WINDOW))
        plt.title("Modeled NAV Changes vs. Actual NAV Changes", fontweight="bold")
        plt.text(.2, .6, 'Variance: ' + str(self.gd_err) + '\n \nmodel: \n' + self.gd_model,transform=plt.gca().transAxes)
        plt.plot(self.cef._cef_pct_chg_df()['DATE'][:-1], self.gd_Y_hat_chg.T, label = 'Deviance from NAV')
        plt.plot(self.cef._cef_pct_chg_df()['DATE'][:-1], self.cef._cef_pct_chg()[:-1], label = 'Actual NAV')
        plt.legend()
        plt.tick_params(axis='x', rotation=70)
        plt.show()
        return fig
    
    def _update_gd_err_plot(self):
        fig = plt.figure(figsize=(WINDOW, WINDOW))
        plt.title("Model Error", fontweight="bold")
        plt.text(.2, .6, 'Variance: ' + str(self.gd_err) + '\n \nmodel: \n' + self.gd_model,transform=plt.gca().transAxes)
        x = self.gd_Y_hat_chg.shape[1]
        e = abs(self.gd_Y_hat_chg.T - np.resize(self.cef._cef_pct_chg()[1:], (x,1)))
        plt.tick_params(axis='x', rotation=70)
        plt.plot(self.cef._cef_pct_chg_df()['DATE'][:-1], e)
        plt.show()
        return fig
    
    """left panel"""
    def _update_etf(self, *args):
        # refresh etf universe
        self.etf_spinner.text = "Fetching ETFs..."
        self.etf_spinner.start()
            
        self.ETFS = Etf()
        self.widgets['etf_dis'].children = [widgets.HTML(self.ETFS.display(), layout={'overflow_y': 'scroll'})]
        self.etf_spinner.stop()
            
    def _update_eq(self, *args):
        self.eq_spinner.text = "Fetching equities..."
        self.eq_spinner.start()
        # refresh equity universe
        self.EQS = Eq()
        self.widgets['eq_dis'].children = [widgets.HTML(self.EQS.display(), layout={'overflow_y': 'scroll'})]
        
        self.eq_spinner.stop()
        
    
    """RIGHT PANEL"""
    def plot_top(self, top_dict, chg_dict):
        fig = plt.figure(figsize=(WINDOW, WINDOW))
        figures = []
        i=0
        for k in top_dict.keys():
            i+=1
            plt.subplot(self.topnum, 1, i)
            # fig = plt.figure(figsize=(WINDOW, WINDOW))
            plt.plot(self.cef._cef_pct_chg_df()['DATE'], chg_dict[k])
            plt.xlabel('dates')
            plt.tick_params(axis='x', rotation=70)
            plt.ylabel('daily price change (%)')
            plt.text(.2, .8, 'Correlation: ' + str(top_dict[k]),transform=plt.gca().transAxes)
            plt.title(k, fontweight='bold')
            
            # figures.append(top_h_plot)
            
        plt.tight_layout(h_pad=2)
        plt.show()
        return fig
    
    def _update_holding_corr_plot(self, *args):
        if not self.cef:
            self._logger.log_message('Cannot update holdings plot; Input fund controls first', color='red')
            return
        self.toph_spinner.text = 'Calculating top correlated holdings...'
        self.toph_spinner.start()
        
        self.chg_h = self.cef._chg_dict()
        self.top_h = self.cef._top_corr(self.topnum)

        top_h_plot = widgets.Output()
        with top_h_plot:
            top_h_plot.clear_output()
            self.plot_top(self.top_h, self.chg_h)

        self.widgets['top_holding_plot'].children = [top_h_plot]
        self.toph_spinner.stop()
        
    def _update_eq_corr_plot(self, *args):
        if not self.EQS:
            self._logger.log_message('Cannot update equities plot; Pull Equities Universe first', color='red')
            return
        self.topeq_spinner.text = 'Calculating top correlated equities...'
        self.topeq_spinner.start()
        
        self.chg_eq = all_changes(self.EQS.eq, {self.cef.CEF_TCK: self.cef._cef_pct_chg()}, self.cef.length(), self.start_date, self.end_date)
        self.eq_df = pd.DataFrame(data=self.chg_eq)
        self.eq_corr = corr_matrix(self.eq_df, self.cef.CEF_TCK)
        self.top_eq = get_top_corr(self.eq_corr, self.topnum, self.cef.CEF_TCK)
        
        top_eq_plot = widgets.Output()
        with top_eq_plot:
            top_eq_plot.clear_output()
            self.plot_top(self.top_eq, self.chg_eq)
        
        # get top corr from eqs
        self.widgets['top_eq_plot'].children = [top_eq_plot]
        self.topeq_spinner.stop()
        
        
    def _update_etf_corr_plot(self, *args):
        if not self.ETFS:
            self._logger.log_message('Cannot update ETFs plot; Pull ETFs Universe first', color='red')
            return
        self.topetf_spinner.text = 'Calculating top correlated ETFs...'
        self.topetf_spinner.start()
        # get top corr from etfs
        
        self.chg_etf1 = all_changes(self.ETFS.etfs1, {self.cef.CEF_TCK: self.cef._cef_pct_chg()}, self.cef.length(), self.start_date, self.end_date)
        self.chg_etf = all_changes(self.ETFS.etfs2, {self.cef.CEF_TCK: self.cef._cef_pct_chg()}, self.cef.length(), self.start_date, self.end_date)
        
        self.chg_etf.update(self.chg_etf1)
        self.etf_df = pd.DataFrame(data=self.chg_etf)
        self.etf_corr = corr_matrix(self.etf_df, self.cef.CEF_TCK)
        self.top_etf = get_top_corr(self.etf_corr, self.topnum, self.cef.CEF_TCK)

        top_etf_plot = widgets.Output()
        with top_etf_plot:
            top_etf_plot.clear_output()
            self.plot_top(self.top_etf, self.chg_etf)
            
        self.widgets['top_etf_plot'].children = [top_etf_plot]
        self.topetf_spinner.stop()
       