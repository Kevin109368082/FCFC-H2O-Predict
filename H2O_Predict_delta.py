# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:21:07 2021

@author: Kevin
"""

from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
# import keras
from tensorflow.keras.models import load_model
# import tensorflow as tf 
import time
import sys


a = []
b = []
c = np.array(500)


def get_xy_test(filename_x, filename_y):
    try:
        y = pd.read_csv(filename_y, index_col="Timestamp")
        x = pd.read_csv(filename_x, index_col="Timestamp")
    except:
        print(filename_y)
        print(filename_x)
        print("file not found")
        sys.exit()

    cols = ['PTA1-DCS-TIC_60102', 'PTA1-DCS-FT_602A03', 'PTA1-DCS-CI_60801']

    x = x[cols]
    data_count_x = x.shape[0]
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            if x.iloc[i, j] == 'No Data' or x.iloc[i, j] == 'Scan Off':
                print(i, '. ', x.iloc[i, j], ' -> ', x.iloc[i - 1, j])
                x.iloc[i, j] = x.iloc[i - 1, j]
    x = x.dropna()            
    x = x.astype(float)
    print("x",x)
    data_x = []
    TS = []
    for i in range(0, int(data_count_x / 60)):
        mean = x.iloc[i * 60:(i + 1) * 60, :].mean()
        TS.append(x.iloc[(i + 1) * 60 - 1].name)
        data_x.append(mean)
    data_x = np.array(data_x)
    Y_time = y.iloc[-1].name
    X_time = x.iloc[-1].name
    print("Y_time:",type(Y_time),Y_time)
    print("X_time:",type(X_time),X_time)
    
    y = pd.to_numeric(y['CTA-1-V-205-H2O'])
    y = np.array(y)
    x_inter = list(range(y.shape[0]))  # x值筆數
    y_inter = list(y.astype(float))  # y值原始值

    cs = CubicSpline(x_inter, y_inter)
    xs = np.arange(0, y.shape[0] - 0.999, 0.25)  # (資料開始，資料結束，每多少插一個值)
    Diff = int(X_time[-5:-3]) - int(Y_time[-5:-3])
    if Diff < 0:
        Diff += 24

    data_y = []
    for i in xs:
        data_y.append(cs(i))
    data_y = np.array(data_y)

    for i in range(0, (3 - Diff)):
        data_y = np.insert(data_y, 0, 0)
    for i in range(0, Diff):
        data_y = np.append(data_y, 0)

    test_data = []
    test_data.append(data_x)
    test_data.append(data_y)
    temp =  time.mktime(time.strptime(TS[-1],"%Y/%m/%d %H:%M"))
    temp += 4*60*60 
    return test_data, Diff, time.strftime("%Y/%m/%d %H:%M",time.localtime(temp))


def main(input_filename_X, input_filename_Y):
    # input_filename_X = "DATA_20210201_20210325_test.csv"
    # input_filename_Y = "Y_20210201_20210325_test.csv"
    # model = keras.models.load_model('./model/haha.h5')
    # model = load_model('./model/delta.h5')
    
    result, step, TS = get_xy_test(input_filename_X, input_filename_Y)

    
    mean = [10.41372523, 127.73592378, 20.26400958, 20.75449152]
    scale = [0.56865834, 0.89375679, 2.87747812, 4.60617893]
    x_array = result[0]
    y_array = result[1]
    y_array = np.reshape(y_array, (16, 1))

    if step != 0:
        for _ in range(step):
            x_array_del = np.delete(x_array, -1, axis=0)
            y_array = np.delete(y_array, -1, axis=0)

        Y_predicted, first_predict_x = first_predict(mean, scale, x_array_del, y_array, model)
        y_spline = [y_array[-9], y_array[-5], y_array[-1], Y_predicted[0]]
        index = list(range(0, 4))  # x值筆數
        cs = CubicSpline(index, y_spline)
        xs = np.arange(0, 3.25, 0.25)  # (資料開始，資料結束，每多少插一個值)
        y_splined = cs(xs)
        for _ in range(3 - step):
            y_splined = np.delete(y_splined, -1, axis=0)

        test_x = x_array[x_array.shape[0] - 4::, :]
        test_y = y_splined[y_splined.shape[0] - 4::, :]
        concatenate_array = np.concatenate((test_y, test_x), axis=1)
        concatenate_array = (concatenate_array - mean) / scale
        second_predict_x = np.reshape(concatenate_array, (1, 4, 4))
        second_y_predicted = model.predict(second_predict_x)
        second_y_predicted = second_y_predicted * 0.21015904404610278 + 0.0006757440830529237
        print('Y_predicted:{} delta:{}'.format(test_y[-1],second_y_predicted))
        return second_y_predicted + test_y[-1], TS

    else:
        # 取後4合併
        Y_predicted, first_predict_x = first_predict(mean, scale, x_array, y_array, model)
        print(first_predict_x)
        print(first_predict_x.shape)
        first_predict_x = np.reshape(first_predict_x,(4,4))
        #把4小時時戳、還有4*4的X抓出來
        print(TS)
        a.append(TS)
        b.append(first_predict_x)
        print(len(a))
        print(a)
        return Y_predicted, TS


def first_predict(mean, scale, x_array, y_array, model):
    test_x = x_array[x_array.shape[0] - 4::, :]
    test_y = y_array[y_array.shape[0] - 4::, :]
    concatenate_array = np.concatenate((test_y, test_x), axis=1)
    concatenate_array = (concatenate_array - mean) / scale
    first_predict_x = np.reshape(concatenate_array, (1, 4, 4))
    Y_predicted = model.predict(first_predict_x)
    Y_predicted = Y_predicted * 0.21015904404610278 + 0.0006757440830529237
    
    return Y_predicted + test_y[-1], first_predict_x


flag = True
folder = './separator_csv2-20210417T064245Z-001/separator_csv2/'
# folder = './20210206_20210324/separator_csv/'
while flag:
    try:
        print("0:manual, 1:auto, 2:exit")
        mode_select = int(input("Please key-in 0, 1 or 2: "))
        # mode_select = 0
        if mode_select == 3:
            sys.exit()
        flag = (mode_select != 0) and (mode_select != 1) and (mode_select != 2)
    except Exception as e:
        pass
    else:
        # input_filename_X = input("Input filename for X : ")
        # input_filename_Y = input("Input filename for Y : ")
        # input_filename_X = './20210416_NTUT_MODEL/2021'+'0113'+'_'+'1700'+'x.csv'
        # input_filename_Y = './20210416_NTUT_MODEL/2021'+'0113'+'_'+'1700'+'y.csv'
        input_filename_X = 'x_1.csv'
        input_filename_Y = 'y_1.csv'
        
if mode_select == 0:
    result, TS = main(input_filename_X, input_filename_Y)
    print(result[0][0])
    
    with open("./result/result.csv", "a") as f:
        f.write("{},{}\n".format(TS, result[0][0]))
if mode_select == 1:
    while True:
        try:
            result, TS = main(input_filename_X, input_filename_Y)
            print(result[0][0])
            with open("./result/result.csv", "a") as f:
                f.write("{},{}\n".format(TS, result[0][0]))
            print("press ctrl + c exit program.")
            time.sleep(3600)
        except KeyboardInterrupt:
            print("")
            print("Exit.")
            sys.exit()
if mode_select == 2:
    num = 0
    model = load_model('./model/delta3.h5')
    while True:
        try:
            num += 1
            input_filename_X = folder +'x_{}.csv'.format(num)
            print(input_filename_X)
            input_filename_Y = folder +'y_{}.csv'.format(num)
            result, TS = main(input_filename_X, input_filename_Y)
            print(result[0][0])
            with open("./result/test.csv", "a") as f:
                f.write("{},{}\n".format(TS, result[0][0]))
            print("press ctrl + c exit program.")
        except KeyboardInterrupt:
            print("")
            print("Exit.")
            sys.exit()
            
