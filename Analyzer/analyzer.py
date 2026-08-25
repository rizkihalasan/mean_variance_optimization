import pandas as pd
import numpy as np
from scipy import optimize
from typing import Any
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

class Analyzer:
    def __init__(self):
        print('Analyzer opened')
        pass

    def filter_volatile_stock(self, stock_lists, start_date, end_date) -> list:
        dict_stdev_below_zero = {}
        res = stock_lists.copy()
        for s in stock_lists:
            df = pd.read_csv('../csv_data/stocks/'+s+'.csv')
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)
            df['pct_close'] = df['Close'].pct_change()*100
            stdev_below_zero = np.std(df[df['pct_close'] < 0]['pct_close'])
            dict_stdev_below_zero[s] = float(stdev_below_zero)

        for volatile_stock in sorted(dict_stdev_below_zero.items(), key=lambda x: x[1], reverse=True)[:len(dict_stdev_below_zero)//10]:
            print('volatile_stock : ', volatile_stock)
            res.remove(volatile_stock[0])
        return res

    def calculate_log_return(self, df, column='Close') -> np.ndarray:
        return np.log(df[column]/(df[column].shift(1)))
    
    def calculate_return(self, df, column='Close') -> pd.DataFrame:
        return df[column].pct_change()
    
    def calculate_covariance_matrix(self, df) -> pd.DataFrame:
        return df.cov()

    def calculate_portfolio_return(self, mean_returns, weights) -> np.ndarray:
        return np.matmul(mean_returns, weights.T)

    def calculate_portfolio_risk(self, covar_returns, weights) -> np.ndarray:
        return np.matmul(np.matmul(weights, covar_returns), weights.T)

    def maximize_returns(self, mean_returns, portfolio_size) -> optimize.OptimizeResult:
        c = (np.multiply(-1, mean_returns))
        A = np.ones([portfolio_size, 1]).T
        b = [1]
        res = optimize.linprog(c, A_ub = A, b_ub = b, bounds=(0,1), method='highs')
        
        return res
    
    def minimize_risk(self, covar_returns, portfolio_size) -> optimize.OptimizeResult:
        def f(x: np.ndarray, covar_returns: np.ndarray) -> float:
            return float(x @ covar_returns @ x.T)
        
        def constraint_eq(x) -> float:
            return float(np.sum(x) - 1.0)
        
        xinit:np.ndarray = np.repeat(0.1, portfolio_size)
        cons: Any = [{'type':'eq', 'fun': constraint_eq}]
        lb = 0
        ub = 1
        bnds = tuple((lb, ub) for _ in range(portfolio_size))
     
        opt = optimize.minimize(f, x0 = xinit, args = (covar_returns,),\
                                bounds = bnds, constraints = cons, tol = 10**-3)
        
        return opt
    

    def minimize_risk_constr(self, mean_returns, covar_returns, portfolio_size, R) -> optimize.OptimizeResult:
        def f(x: np.ndarray, covar_returns: np.ndarray) -> float:
            return float(x @ covar_returns @ x.T)
        
        def constraint_eq(x) -> float:
            return float(np.sum(x) - 1.0)
        
        def constraint_ineq(x, mean_returns, R) -> float:
            a_ineq = np.array(mean_returns)
            b_ineq = R
            ineq_constraint_val = float(np.matmul(a_ineq, x.T)) - b_ineq
            return ineq_constraint_val
        
        xinit: np.ndarray = np.repeat(0.1, portfolio_size)
        cons: Any = ({'type':'eq', 'fun':constraint_eq},
            {'type':'ineq', 'fun':constraint_ineq, 'args':(mean_returns, R)})
        lb = 0
        ub = 1
        bnds = tuple((lb, ub) for _ in range(portfolio_size))
        
        opt = optimize.minimize(f, args=(covar_returns,), method='trust-constr',\
                                x0 = xinit, bounds = bnds, constraints = cons, tol=10**-3)
        
        return opt
    
    def retrieve_stocks_weights(self, weights, stock_lists, yearly_return_lists, yearly_cov_matrix) -> pd.DataFrame:
        risk_point = []
        return_point = []
        stock_weights = []
        for i, weight in enumerate(weights):
            return_point.append(round(self.calculate_portfolio_return(yearly_return_lists, weight)*100,2))
            risk_point.append(round(self.calculate_portfolio_risk(yearly_cov_matrix, weight)*100,2))
            stock_weight = {}
            for j in range(len(weights[i])):
                stock_weight[stock_lists[j]] = float(round(weights[i][j]*100,2))
            stock_weight = sorted(stock_weight.items(), key=lambda x: x[1], reverse=True)
            stock_weights.append(stock_weight)
        df_portman = pd.DataFrame({
            'return_point': return_point,
            'risk_point': risk_point,
            'stock_weights': stock_weights,
        })
        return df_portman

    def calculate_sharpe_ratio(self, df_portman, min_return) -> pd.DataFrame:
        df_portman['sharpe_ratio'] = (df_portman['return_point'] - min_return) / df_portman['risk_point']
        return df_portman

    def calc_weights_pypfopt_max_weight(self, df_pypfopt, max_weight=0.2) -> np.ndarray:
        if 'Date' in df_pypfopt.columns:
            df_pypfopt.set_index('Date', inplace=True)
        mu = expected_returns.mean_historical_return(df_pypfopt)
        S = risk_models.sample_cov(df_pypfopt)

        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda x : x <= max_weight) # for now only 1 constraint
        # ef.min_volatility()
        weights = ef.max_sharpe(risk_free_rate=0.05)
        ef.portfolio_performance(risk_free_rate=0.05, verbose=True)

        return weights