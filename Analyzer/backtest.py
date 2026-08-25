import backtrader as bt
from datetime import datetime
import pandas as pd
import numpy as np
from functools import reduce
from datetime import date
from dateutil.relativedelta import relativedelta

class CombinedYahooFinanceData(bt.feeds.GenericCSVData):
    lines = ('fear_greed_index',)
    params = (
        ('dtformat', '%Y-%m-%d'),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('fear_greed_index', 6),
        ('openinterest', -1),
    )

class FearGreed(bt.Strategy):
    def __init__(self):
        self.fear_greed_index = self.datas[0].fear_greed_index
        print('init fear_greed')
        print(self.fear_greed_index)
        
    def next(self):
        if not self.position:
            if self.fear_greed_index <= 34:
                self.buy()
        elif self.fear_greed_index >= 60 :
            self.close()

class ManualBacktest():
    def __init__(self):
        self.df_port = None

    def make_portfolio(self, allocation):
        dfs = []
        for k, v in allocation.items():
            df = pd.read_csv('../csv_data/stocks/'+k+'.csv')
            df.rename(columns={'Close': k}, inplace=True)
            dfs.append(df[['Date', k]])


        # Define how to merge two dataframes
        def merge_dfs(left, right, key='Date'):
            return pd.merge(left, right, on=key, how='inner') # or 'inner', 'left', 'right'

        # Apply the merge function cumulatively to the list of dataframes
        df_port = reduce(merge_dfs, dfs)

        start_price = {k: float(df_port[k][0]) for k in allocation}

        for k in allocation:
            column_name = k+'_money' 
            df_port[column_name] = allocation[k] * df_port[k] / start_price[k]


        columns_to_sum = [k+'_money' for k in allocation]
        print('columns_to_sum : ')
        print(columns_to_sum)
        df_port['Total_Money'] = df_port[columns_to_sum].sum(axis=1)
        df_port['pct_daily_return'] = df_port['Total_Money'].pct_change()*100
        df_port['Date'] = pd.to_datetime(df_port['Date'].str[0:10])

        self.df_port = df_port
        return df_port
    
    def calculate_return_1n(self, allocation, df):
        weight = 1/len(allocation)
        n = len(df)
        start_total_money = sum(df[asset][0] * weight for asset in allocation)
        end_total_money = sum(df[asset][n-1] * weight for asset in allocation)
        return ((end_total_money/start_total_money) - 1) * 100

    def calculate_return(self):
        if self.df_port:
            return ((self.df_port['Total_Money'][len(self.df_port)-1] / self.df_port['Total_Money'][0]) - 1) * 100

        return None

    def calculate_stdev(self):
        if self.df_port:
            return self.df_port['Total_Money']
        return None

    def calculate_sharpe_ratio(self, min_return):
        if self.df_port:
            ret = self.calculate_return()
            stdev = self.calculate_stdev() 
            return (ret - min_return) / stdev
        return None

    def nearest_date(self, dates, target_date):
        return min(dates,key=lambda x: abs(x-target_date))

    def delta_date(self, current_date, years = 0, months = 0, days = 0, fmt='%Y-%m-%d'):
        if isinstance(current_date, str):
            datetime_object = datetime.strptime(current_date, fmt)
        else:
            datetime_object = current_date
        return datetime_object + relativedelta(years = years, months = months, days = days)
    
    def calculate_return_df(self, df, start_date, end_date, date_column='Date', return_column='Total_Money'):
        df[date_column] = df[date_column].astype(str)
        df[date_column] = df[date_column].str[0:10]
        df_port = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)].reset_index(drop=True)
        self.df_port_return = df_port
        n = len(df_port)
        print('Total Money end : ', df_port[return_column][n-1])
        print('Total Money start : ', df_port[return_column][0])
        return (df_port[return_column][n-1] - df_port[return_column][0]) / df_port[return_column][0] * 100
    
    def calculate_std_df(self, df, start_date, end_date, date_column='Date', std_column='pct_daily_return'):
        df[date_column] = df[date_column].astype(str)
        df[date_column] = df[date_column].str[0:10]
        df_port = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)].reset_index(drop=True)
        self.df_port_std = df_port
        n = len(df_port)
        return np.std(df_port[std_column]) * np.sqrt(n)
        
    def calculate_return_benchmark(self, start_date, end_date, benchmark='SPY', date_column='Date'):
        df = pd.read_csv('../csv_data/stocks/'+benchmark+'.csv')
        return self.calculate_return_df(df, start_date, end_date, date_column, 'Close')

    def calculate_std_benchmark(self, start_date, end_date, benchmark='SPY', date_column='Date', return_column='Close'):
        df = pd.read_csv('../csv_data/stocks/'+benchmark+'.csv')
        df_port = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)].reset_index(drop=True)
    
        df_port['pct_daily_return'] = df[return_column].pct_change()*100
        n = len(df_port)
        return np.std(df_port['pct_daily_return']) * np.sqrt(n)




    def make_portfolio_rebalance(self, start_date, end_date, allocation, date_column='Date'):
        dfs = []

        for k, v in allocation.items():
            df = pd.read_csv('../csv_data/stocks/'+k+'.csv')
            df.rename(columns={'Close': k}, inplace=True)
            dfs.append(df[[date_column, k]])


        # Define how to merge two dataframes
        def merge_dfs(left, right, key='Date'):
            return pd.merge(left, right, on=key, how='inner') # or 'inner', 'left', 'right'

        # Apply the merge function cumulatively to the list of dataframes
        df_port = reduce(merge_dfs, dfs)
        # print('len(df_port) (before trim) : ', len(df_port))
        # print('end_date make_portfolio_rebalance : ', end_date)
        df_port[date_column] = df_port[date_column].astype(str)
        df_port[date_column] = df_port[date_column].str[0:10]
        df_port = df_port[(df_port[date_column] >= start_date) & (df_port[date_column] <= end_date)].reset_index(drop=True)
        # print('len(df_port) : ', len(df_port))
        start_price = {k: float(df_port[k][0]) for k in allocation}

        for k in allocation:
            column_name = k+'_money' 
            df_port[column_name] = allocation[k] * df_port[k] / start_price[k]
        
        columns_to_sum = [k+'_money' for k in allocation]
        # print('columns_to_sum : ')
        # print(columns_to_sum)
        df_port['Total_Money'] = df_port[columns_to_sum].sum(axis=1)
        df_port['pct_daily_return'] = df_port['Total_Money'].pct_change()*100
        df_port[date_column] = pd.to_datetime(df_port[date_column].str[0:10])

        high_water_mark = []
        drawdown = []

        max_money = 0
        for money in df_port['Total_Money']:
            if money > max_money:
                max_money = money
            high_water_mark.append(max_money)

            drawdown.append((money - max_money) / max_money * 100)
        
        df_port['High_Water_Mark'] = high_water_mark
        df_port['Drawdown'] = drawdown


        self.df_port = df_port
        self.high_water_mark = high_water_mark[-1]
        self.drawdown = drawdown[-1]
        return df_port

    def calculate_underperformed_stocks(self, allocation, df, column='_money'):
        return_per_asset = {}
        n = len(df)
        for k in allocation:
            asset_delta_column = df[k+column][n-1] - df[k+column][0]
            return_per_asset[k] = float(asset_delta_column)
        return dict(sorted(return_per_asset.items(), key=lambda item: item[1]))

data = CombinedYahooFinanceData(dataname='TSLA_fear_greed.csv',
                                 fromdate=datetime(2025, 1, 6),
                                 todate=datetime(2025, 12, 6),
                                 reverse=False,
                                 header = True)