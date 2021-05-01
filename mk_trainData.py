import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import sys
print(sys.version)

def get_xy_test(filename_x, filename_y):
    # filename_x = "X.csv"
    # filename_y = "Y.csv"
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
    
    y_ans =  np.array(cs(xs))
    y_train = np.zeros((4,12908))
    train_data = np.zeros((12901,4,4))
    
    for i in range(0,len(y)-3):
        y_temp = y[i:i+4]
        x_inter = list(range(y_temp.shape[0]))  # x值筆數
        y_inter = list(y_temp.astype(float))  # y值原始值
        cs = CubicSpline(x_inter, y_inter)
        xs = np.arange(0, y_temp.shape[0] - 0.999, 0.25)
        y_train_buf = np.array(cs(xs))
        # for j in range(0,4):
        y_train[:,i*4] = y_train_buf[-7:-3]
        y_train[:,i*4+1] = y_train_buf[-6:-2]
        y_train[:,i*4+2] = y_train_buf[-5:-1]
        y_train[:,i*4+3] = y_train_buf[-4::]
    train_ans = np.zeros((12901))
    
    for i in range(0,y_train.shape[1]-7):
        train_data[i,:,0] = y_train[:,i]
        train_data[i,:,1::] = data_x[i+9:i+13,:]
        train_ans[i] = y_ans[i+13]-y_ans[i+9]
    mean = [10.41372523, 127.73592378, 20.26400958, 20.75449152]
    scale = [0.56865834, 0.89375679, 2.87747812, 4.60617893]
    for i in range(0,train_data.shape[0]):
        train_data[i,:,:] = (train_data[i,:,:] - mean) / scale
    return train_data, train_ans


TD , TA = get_xy_test("X.csv","Y.csv")
print(TD)
print(TA)

X_train = TD[0:int(TD.shape[0]*0.7)]
Y_train = TA[0:int(TD.shape[0]*0.7)]

X_valid = TD[int(TD.shape[0]*0.7):int(TD.shape[0]*0.85)]
Y_valid = TA[int(TA.shape[0]*0.7):int(TA.shape[0]*0.85)]

X_test = TD[int(TD.shape[0]*0.85):int(TD.shape[0])]
Y_test = TA[int(TA.shape[0]*0.85):int(TA.shape[0])]

np.save('X_train',X_train)
np.save('Y_train',Y_train)
np.save('X_valid',X_valid)
np.save('Y_valid',Y_valid)
np.save('X_test',X_test)
np.save('Y_test',Y_test)