import numpy as np
import pandas as pd
import time
import sys


def seperate_data(filename_x,filename_y):
    try:
        y = pd.read_csv(filename_y)#, index_col="Timestamp"
        x = pd.read_csv(filename_x)#, index_col="Timestamp")
    except:
        print("file not found")
        sys.exit()
    cols_x = np.array(x.columns[:])

    print(cols_x)
    print(cols_x.shape)
    print(x.shape)
    start_time = "2021/2/1 9:00"
    end_time = '2021/3/25 00:00'
    start_time_ts =  time.mktime(time.strptime(start_time,"%Y/%m/%d %H:%M"))
    end_time_ts = start_time_ts + 16*60*60

    # y_ts = np.array(y.iloc[:].index)
    y_ts = np.array(y)
    x_ts = np.array(x)
    # print(x_ts[:,0])
    text = ['01', '05', '09', '13' , '17' , '21']
    count = 0

    for i in range(0,y_ts.shape[0]):
        if y_ts[i,0][-5:-3] not in text:
            print(y_ts[i,0])
            count += 1
            continue
        y_ts[i-count,0] = time.mktime(time.strptime(y_ts[i,0],"%Y/%m/%d %H:%M"))
        y_ts[i-count,1] = y_ts[i,1]
    for i in range(0,x_ts.shape[0]):
        x_ts[i,0] = time.mktime(time.strptime(x_ts[i,0],"%Y/%m/%d %H:%M"))  
    # print(y_ts.shape)
    y_ts = np.delete(y_ts,np.s_[-1*count::], axis = 0)

    num = 0 
    while(end_time_ts <= time.mktime(time.strptime(end_time,"%Y/%m/%d %H:%M"))):

        aaa = np.where((y_ts[:,0] > start_time_ts) & (y_ts[:,0] <= end_time_ts))
        bbb = np.where((x_ts[:,0] > start_time_ts) & (x_ts[:,0] <= end_time_ts))
        out_y = y_ts[aaa[0],:]
        out_x = x_ts[bbb[0],:]
        # print(out.shape)
        for i in range(0,out_y.shape[0]):
            out_y[i,0] = time.strftime("%Y/%m/%d %H:%M",time.localtime(out_y[i,0]))
        for i in range(0,out_x.shape[0]):
            out_x[i,0] = time.strftime("%Y/%m/%d %H:%M",time.localtime(out_x[i,0]))
            # print(time.strftime("%Y/%m/%d %H:%M",time.localtime(i)))
        
        if (out_y.shape[0] < 4) | (out_x.shape[0] < 960) :
            start_time_ts += 3600    
            end_time_ts = start_time_ts + 16*60*60  
            # print(out)          
            continue
        elif out_y.shape[0] > 4:
            print('error',out_y)
        else:
            num += 1
            print(out_y)
            df_y = pd.DataFrame(out_y, columns = ['Timestamp','CTA-1-V-205-H2O'])
            df_x = pd.DataFrame(out_x, columns = cols_x)
            # print(out_y)
            # print(out_x)
            file_name_xx = "./separator_csv/x_{}".format(num)
            file_name_yy = "./separator_csv/y_{}".format(num)
            # np.savetxt(file_name_xx, out_x , delimiter=",")
            # np.savetxt(file_name_yy, out_y , delimiter=",")
            df_y.to_csv(file_name_yy)
            df_x.to_csv(file_name_xx)
            
        start_time_ts += 3600    
        end_time_ts = start_time_ts + 16*60*60
        
seperate_data('DATA_20210201_20210325.csv','Y_20210201_20210325.csv')
