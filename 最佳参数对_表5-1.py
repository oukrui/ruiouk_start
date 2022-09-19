import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd

def svm_timeseries_prediction(all_data, c_parameter, gamma_paramenter):
    X_data = all_data
    #Y_data = single1
    print(len(X_data))
    # 整个数据的长度
    long = len(X_data)
    # 取前多少个X_data预测下一个数据
    X_long = 7
    svr_rbf = SVR(kernel='rbf', C=c_parameter, gamma=gamma_paramenter)
    X = []
    Y = []
    for k in range(len(X_data) - X_long - 1):
        t = k + X_long
        X.append(X_data[k:t])
        Y.append(X_data[t + 1])
    y_rbf = svr_rbf.fit(X, Y).predict(X)
    mse = mean_squared_error(Y, y_rbf)
    print(mse)
    return  mse


if __name__ == '__main__':
    keys = pd.read_excel('mf201701_1231_1_new(周).xlsx', sheet_name=None).keys()
    for key in keys:
        one_key = key
        data = pd.read_excel('mf201701_1231_1_new(周).xlsx', sheet_name=one_key)
        J = data['四川.棉丰四川.棉丰/220kV.资棉一线264有功值']
        best_c = None
        best_g = None
        max = 10000
        for c in range(10, 100, 1):
            for g in range(1, 11, 1):
                mse = svm_timeseries_prediction(J, c, g)
                print('c:', c, 'g:', g, 'mse:', mse)
                if mse < max:
                    best_c = c
                    best_g = g
                    max = mse

        print('best_c:', best_c)
        print('best_g:', best_g)
        print('best mse:',max)
        break

