from abc import ABC, abstractmethod
import pandas as pd
import time
import yfinance as yf

class DataSource(ABC):
    @abstractmethod
    def get_stock_data(self, symbol, period):
        pass

class YahooFinanceSource(DataSource):
    def __init__(self):
        self.DataSource = DataSource

    def get_stock_data(self, symbol, period="5y"):
        try:
            ticker = yf.Ticker(symbol)
            print('ticker : ', ticker)
            data = ticker.history(period = period)

            data['Symbol'] = symbol

            return data
        except Exception as e:
            print('error in YahooFinance \n')
            return pd.DataFrame()

class StockDataCollector:
    def __init__(self, data_source = YahooFinanceSource()):
        self.data_source = data_source
        self.stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD',
            'DIS', 'BABA', 'V', 'JPM', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD',
            'BAC', 'ADBE', 'CRM', 'PYPL', 'INTC', 'VZ', 'KO', 'PEP', 'T',
            'CSCO', 'XOM', 'ABT', 'TMO', 'ACN', 'TXN', 'QCOM', 'COST', 'SBUX',
            'MDT', 'LOW', 'HON', 'UPS', 'IBM', 'GS', 'CAT', 'MMM', 'SLB', 'QQQ',
            'LLY', 'JPM', 'ASML', 'HD', 'MU', 'CVX', 'NVS', 'PM', 'NVO', 'SHEL',
            'KLAC', 'TXN', 'PFE' ,'SPOT', 'BX', 'HOOD', 'GM', 'SE', 'TLT', 'SMH'
        ]
        self.stock_data = {}
    
    def collect_stock_data(self, symbols=None, period='6y', delay=1.0):
        if symbols is None:
            symbols = self.stocks
        for i, symbol in enumerate(symbols):
            try:
                print(' i and symbol : ', i, symbol)
                stock_data = self.data_source.get_stock_data(symbol)
                if not stock_data.empty:
                    stock_data.to_csv('csv_data/stocks/'+symbol+'.csv')
                    self.stock_data[symbol] = stock_data
                
                if delay > 0 and i < len(symbols) - 1:
                    time.sleep(delay)
            except Exception as e:
                print('error on collect_stock_data\n', e)
                continue
sc = StockDataCollector()
sc.collect_stock_data()