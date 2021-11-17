# import tensorflow as tf
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import datetime
from finta import TA
from pandas_datareader import data as pdr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# List of symbols for technical indicators
INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

def get_stock_data(symbol,num_days=10000,interval='1d'):

    start = (datetime.date.today() - datetime.timedelta(num_days) )
    end = datetime.datetime.today()

    data = yf.download(symbol, start=start, end=end, interval=interval)
    data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
    return data

def get_indicator_data(data):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    """

    for indicator in INDICATORS:
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features
    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    del (data['Adj Close'])
    
    return data

def get_x_y(data):
    y = np.ceil((data['open']-data['close']).clip(lower=0, upper=1)).to_numpy()
    X = data.copy()
    del (X['open'])
    del (X['close'])
    X = X.to_numpy()
    return X,y

data = get_stock_data('TSLA')
data = get_indicator_data(data)
data = data.iloc[16:]
data = data.dropna()
test_data = data.iloc[-250:]
data = data.iloc[:-251]
X_train, y_train = get_x_y(data)
X_test, y_test = get_x_y(test_data)
today_data = data.iloc[-2:]
del (today_data['open'])
del (today_data['close'])

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
# print(rf.predict(today_data.to_numpy()))
print(rf.score(X_test,y_test))
y_pred = rf.predict(X_test)
print(f1_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

k=5
kfold = KFold(k)
acc_score = []
model = RandomForestClassifier()
X=X_train
y=y_train 

for train_index , test_index in kfold.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))