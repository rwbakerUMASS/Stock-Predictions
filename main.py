from numpy.core.numeric import inf
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import datetime
from finta import TA
from pandas_datareader import data as pdr
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

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

    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    del (data['high'])
    del (data['low'])
    del (data['volume'])
    del (data['Adj Close'])
    
    return data

def get_x_y(data):
    y = (data['open']-data['close']).to_numpy()
    X = data.copy()
    del (X['open'])
    del (X['close'])
    X = X.to_numpy()
    return X,y

def ensemble(X_train, y_train, X_test):
    rf = RandomForestRegressor()
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    return y_pred

def tune_model(param_grid,train,valid):
    train_X, train_y = train
    valid_X, valid_y = valid
    best = dict(score=inf)
    param_grid = ParameterGrid(param_grid)
    for params in param_grid:
        print('Params: ',params)
        model = Lasso(**params)
        model.fit(train_X,train_y)
        print('Model fit')
        y_pred = model.predict(valid_X)
        print('Predictions Made')
        score = mean_squared_error(y_pred, valid_y,squared=False)
        print('Score: ',score,'\n')
        if score < best['score']:
            best['score'] = score
            best['params'] = params
    return best

@ignore_warnings(category=ConvergenceWarning)
def tune_ada(param_grid,train,valid):
    train_X, train_y = train
    valid_X, valid_y = valid
    best = dict(score=inf)
    param_grid = ParameterGrid(param_grid)
    for params in param_grid:
        print('Params: ',params)
        model = AdaBoostRegressor(**params)
        model.fit(train_X,train_y)
        print('Model fit')
        y_pred = model.predict(valid_X)
        print('Predictions Made')
        score = mean_squared_error(y_pred, valid_y,squared=False)
        print('Score: ',score,'\n')
        if score < best['score']:
            best['score'] = score
            best['params'] = params
    return best

def tune_rf(param_grid,train,valid):
    train_X, train_y = train
    valid_X, valid_y = valid
    best = dict(score=inf)
    param_grid = ParameterGrid(param_grid)
    for params in param_grid:
        print('Params: ',params)
        model = RandomForestRegressor(**params)
        model.fit(train_X,train_y)
        print('Model fit')
        y_pred = model.predict(valid_X)
        print('Predictions Made')
        score = mean_squared_error(y_pred, valid_y, squared=False)
        print('Score: ',score,'\n')
        if score < best['score']:
            best['score'] = score
            best['params'] = params
    return best

def k_fold(model,k,X,y):
    kfold = KFold(k)
    scores = []
    iter = 1
    for train_index , test_index in kfold.split(X):
        X_train , X_test = X[train_index,:],X[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        model.fit(X_train,y_train)
        # print(f'ITER: {iter} | Model Fit')
        pred_values = model.predict(X_test)
        # print(f'ITER: {iter} | Predictions Made')
        score = mean_squared_error(pred_values, y_test,squared=False)
        # print(f'ITER: {iter} | RMSE: {score}\n{"-"*20}')
        scores.append(score)
        iter += 1
    return sum(scores)/k
def main():
    data = get_stock_data('SPY')
    data = get_indicator_data(data)
    data = data.iloc[16:]
    data = data.dropna()
    test_data = data.iloc[-2000:]
    train_data = data.iloc[:-2001]
    X_train, y_train = get_x_y(train_data)
    X_test, y_test = get_x_y(test_data)
    X,y = get_x_y(data)
    today_data = data.iloc[-1:]
    del (today_data['open'])
    del (today_data['close'])

    best_lasso={'score': 1.9106815845016039, 'params': {'alpha': 0.25, 'max_iter': 5000}}
    lasso_param_grid = dict(alpha=[0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4],max_iter=[5000,10000])

    best_ada = {'score': 1.583492062243895, 'params': {'base_estimator': DecisionTreeRegressor(), 'learning_rate': 0.5, 'loss': 'linear', 'n_estimators': 50}}
    models = [Lasso(**(best_lasso['params'])),DecisionTreeRegressor()]
    ada_param_grid = dict(base_estimator=models,n_estimators=[10,25,50],learning_rate=[0.5,1,1.5,2],loss=['linear','square'])

    best_rf = {'score': 1.5823973188679223, 'params': {'max_depth': None, 'n_estimators': 200}}
    rf_param_grid = dict(n_estimators=[25,50,100,200], max_depth=[None,3,5,7,9])

    best_lasso = tune_model(lasso_param_grid,(X_train,y_train),(X_test,y_test))
    best_ada = tune_ada(ada_param_grid,(X_train,y_train),(X_test,y_test))
    best_rf = tune_rf(rf_param_grid,(X_train,y_train),(X_test,y_test))
    print(f'Best Lasso: {best_lasso}')
    print(f'Best Ada: {best_ada}')
    print(f'Best RF: {best_rf}')
    lasso_model = Lasso(**(best_lasso['params']))
    ada_model = AdaBoostRegressor(**(best_ada['params']))
    rf_model = RandomForestRegressor(**(best_rf['params']))
    print(f'Lasso KFold: {k_fold(lasso_model,25, X, y)}')  #1.140228025186827
    print(f'Ada KFold: {k_fold(ada_model, 25, X, y)}')     #0.911521338476026
    print(f'RF KFold: {k_fold(rf_model,25, X, y)}')        #0.899540035805616

if __name__ == '__name__':
    main()