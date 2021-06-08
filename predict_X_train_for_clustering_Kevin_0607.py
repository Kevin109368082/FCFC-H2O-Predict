import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy import stats
import joblib
# ==============
# X_train = np.load('X_total_3f.npy')
# Y_predict = np.load('Y_predicted.npy')
# Y_real = np.load('Y_total_3f.npy')
# ==============
X_train = np.load('./Data_clustering/X_clustering.npy')
Y_predict = np.load('./Data_clustering/Y_predict_clustering.npy')
Y_real = np.load('./Data_clustering/Y_real_clusting.npy')

unusal_label = [312,348,391,1782,2119]
X_train = np.delete(X_train,unusal_label,axis=0)
Y_predict = np.delete(Y_predict,unusal_label,axis=0)
Y_real = np.delete(Y_real,unusal_label,axis=0)

Y_predict_1dim = np.reshape(Y_predict,(Y_predict.shape[0],))
print('MSE')
print(np.mean(np.square(Y_predict_1dim - Y_real)))
Y_real_2dim = np.reshape(Y_real,(Y_real.shape[0],1))
y_error = Y_real_2dim - Y_predict

# ==============
print(X_train.shape)
# input()
CENTER = 50
X_train_2dim = np.reshape(X_train, (X_train.shape[0], 16))
X_train_2dim = np.concatenate((X_train_2dim,y_error),axis=1) #第16行是預測誤差
MAX_VAR = 10
DIST_VAR_TH = 0.5
ERR_VAR_TH = 0.9
N_INIT = 50  #用不同的初始化質心執行演算法的次數。這裡和KMeans類意義稍有不同，KMeans類裡的n_init是用同樣的訓練集資料來跑不同的初始化質心從而執行演算法。而MiniBatchKMeans類的n_init則是每次用不一樣的取樣資料集來跑不同的初始化質心執行演算法。
GAIN = 2

DOES = True
print(CENTER)
kmeans = MiniBatchKMeans(n_clusters=CENTER, n_init=N_INIT, random_state=0, verbose=0, tol=1e-6).fit(X_train_2dim[:,0:16])
recluster_data = X_train_2dim
all_save_cluster_center = np.zeros((1, 16))


# def draw_(km, train_data):
#     distance_var_list = []
#     class_num_list = []
#     cluster_dict = dict()
#     for i in range(TEST_CENTER):
#         cluster_dict[str(i)] = train_data[km.labels_ == i]
#         class_num = train_data[km.labels_ == i].shape[0]
#         diff = km.cluster_centers_[i] - cluster_dict[str(i)][:,0:16]
#         dist = np.sqrt(np.sum(np.square(diff), axis=1))
#         avg_dist = np.average(dist)
#         dist_var = np.square(np.std(dist))
#         class_num_list.append(class_num)
#         distance_var_list.append(dist_var)
#         print('center:{} avg_dist:{} dist_var:{} num:{}'.format(i, avg_dist, dist_var, class_num))
#     index = np.arange(TEST_CENTER)

#     c = list(zip(class_num_list, distance_var_list))
#     c.sort(reverse=False)  # 降序
#     class_num_list[:], distance_var_list[:] = zip(*c)

#     plt.figure()
#     plt.xlabel('center ID')
#     plt.ylabel('Var')
#     plt.title('Test center : {}'.format(TEST_CENTER))
#     plt.plot(index, distance_var_list)
#     plt.plot(index, [x / 10 for x in class_num_list])
#     # plt.ylim(0, 5)
#     # plt.plot(index, class_num_list)
#     plt.show()
    
def draw_(km, train_data):
    distance_var_list = []
    distance_avg_list = []
    cluster_dict = dict()
    class_num_list = [] #各群成員數量
    error_var_list = []
    error_mean_list = []
    #分群 紀錄各群成員數量、距離變異數等等
    for i in range(TEST_CENTER):
        cluster_dict[str(i)] = train_data[km.labels_ == i]
        class_num = train_data[km.labels_ == i].shape[0]
        diff = km.cluster_centers_[i] - cluster_dict[str(i)][:,0:16]
        dist = np.sqrt(np.sum(np.square(diff), axis=1))
        avg_dist = np.average(dist)
        dist_var = np.square(np.std(dist))
        distance_var_list.append(dist_var)
        class_num_list.append(class_num)
        error_var = np.square(np.std(cluster_dict[str(i)][:,16]))    #各群誤差變異數
        error_var_list.append(error_var)
        distance_avg_list.append(avg_dist)
        error_mean = np.mean(cluster_dict[str(i)][:,16])
        error_mean_list.append(error_mean)
        print('center:{} avg_dist:{} dist_var:{} num:{}'.format(i, avg_dist, dist_var, class_num))
        print('error_var:{}'.format(error_var))
    index = np.arange(TEST_CENTER)
    
    c = list(zip(class_num_list, distance_var_list, distance_avg_list, error_var_list))
    c.sort(reverse=False)  # 降序
    class_num_list[:], distance_var_list[:], distance_avg_list[:], error_var_list[:]= zip(*c)
    
    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (after): {} (distance)'.format(TEST_CENTER))
    plt.plot(index, distance_var_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['distance var','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()
    
    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (after): {} (distance)'.format(TEST_CENTER))
    plt.plot(index, distance_avg_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['distance mean','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()
    
    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (after): {} (error)'.format(TEST_CENTER))
    plt.plot(index, error_var_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['error var','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()

    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (after): {} (error)'.format(TEST_CENTER))
    plt.plot(index, error_mean_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['error mean','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()

while True:
    distance_var_list = []
    distance_avg_list = []
    cluster_dict = dict()
    class_num_list = [] #各群成員數量
    error_var_list = []
    error_mean_list = []
    #分群 紀錄各群成員數量、距離變異數等等
    for i in range(CENTER):
        cluster_dict[str(i)] = recluster_data[kmeans.labels_ == i]
        class_num = recluster_data[kmeans.labels_ == i].shape[0]
        diff = kmeans.cluster_centers_[i] - cluster_dict[str(i)][:,0:16]
        dist = np.sqrt(np.sum(np.square(diff), axis=1))
        avg_dist = np.average(dist)
        dist_var = np.square(np.std(dist))
        distance_var_list.append(dist_var)
        class_num_list.append(class_num)
        error_var = np.square(np.std(cluster_dict[str(i)][:,16]))    #各群誤差變異數
        error_var_list.append(error_var)
        distance_avg_list.append(avg_dist)
        error_mean = np.mean(cluster_dict[str(i)][:,16])
        error_mean_list.append(error_mean)
        
        print('center:{} avg_dist:{} dist_var:{} num:{}'.format(i, avg_dist, dist_var, class_num))
        print('error_var:{}'.format(error_var))
    index = np.arange(CENTER)
    
    c = list(zip(class_num_list, distance_var_list, distance_avg_list, error_var_list, error_mean_list))
    c.sort(reverse=False)  # 降序
    class_num_list[:], distance_var_list[:], distance_avg_list[:], error_var_list[:], error_mean_list[:]= zip(*c)
    
    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (initial): {} (distance)'.format(CENTER))
    plt.plot(index, distance_var_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['distance var','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()
    
    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (initial): {} (distance)'.format(CENTER))
    plt.plot(index, distance_avg_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['distance mean','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()
    
    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (initial): {} (error)'.format(CENTER))
    plt.plot(index, error_var_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['error var','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()
    
    plt.figure()
    plt.xlabel('center ID')
    plt.title('center (initial): {} (error)'.format(CENTER))
    plt.plot(index, error_mean_list)
    plt.plot(index, [x / 10 for x in class_num_list])  #print各群內數量除以10
    plt.legend(['error mean','numbers'],loc = 'upper right')
    # plt.axis([0,len(class_num_list),0,5])
    plt.show()
    


    # 需要重新分類的群心
    all_cluster_center = kmeans.cluster_centers_
    recluster_label = []
    save_cluster_label = []
    #以距離變異數做條件，紀錄需要被重新分類的群心，同時記錄要被保留的群心
    # for i in range(0, len(distance_var_list)):
    #     if (distance_var_list[i] > DIST_VAR_TH):
    #         recluster_label.append(i)
    #     else:
    #         save_cluster_label.append(i)
    #以誤差變異數做條件，紀錄需要被重新分類的群心，同時記錄要被保留的群心
    for i in range(0, len(error_var_list)):
        if (error_var_list[i] > ERR_VAR_TH):
            recluster_label.append(i)
        else:
            save_cluster_label.append(i)

    recluster_center = all_cluster_center[recluster_label]
    save_cluster_center = all_cluster_center[save_cluster_label]
    save_cluster_center_array = np.array(save_cluster_center)

    print('all_save_cluster_center', all_save_cluster_center.shape)
    print('save_cluster_center_array', save_cluster_center_array.shape)
    #(1,16)全0的陣列+要存的群心
    all_save_cluster_center = np.concatenate((all_save_cluster_center, save_cluster_center_array), axis=0)

    # print(save_cluster_center.shape)
    # print(len(recluster_center))
    #如果沒有要re分群的就break
    if len(recluster_center) < 1:
        break
    recluster_data = np.zeros((1, 17))
    for i in range(0, len(recluster_label)):
        recluster_data = np.concatenate((recluster_data, cluster_dict[str(recluster_label[i])]), axis=0)
    recluster_data = np.delete(recluster_data, 0, axis=0) #需要重分的data
    print('recluster {} type:{}'.format(recluster_data.shape, type(recluster_data)))
    CENTER = GAIN * len(recluster_center)  # 刪掉的要乘幾倍

    kmeans = MiniBatchKMeans(n_clusters=CENTER, n_init=N_INIT, random_state=0, verbose=1, tol=1e-6).fit(
        recluster_data[:,0:16])

    print(recluster_data.shape)
    if DOES:
        all_save_cluster_center = np.delete(all_save_cluster_center, 0, axis=0)
        DOES = False
    TEST_CENTER = len(all_save_cluster_center)
    test_kmeans = MiniBatchKMeans(n_clusters=TEST_CENTER, n_init=N_INIT, random_state=0, verbose=0,
                                  tol=1e-6).partial_fit(all_save_cluster_center)
    test_kmeans.fit(X_train_2dim[:,0:16])

    draw_(test_kmeans, X_train_2dim)
print('---------------------Here-------------------')
TEST_CENTER = len(all_save_cluster_center)
test_kmeans = MiniBatchKMeans(n_clusters=TEST_CENTER, n_init=N_INIT, random_state=0, verbose=0, tol=1e-6).partial_fit(
    all_save_cluster_center)
test_kmeans.fit(X_train_2dim[:,0:16])
# test_kmeans.fit_predict(X_train_2dim)

draw_(test_kmeans, X_train_2dim)
joblib.dump(test_kmeans, './KmeansModel/Kmeans_model0607')
#各群平均預測誤差
y_error =  Y_real - Y_predict_1dim
labels = test_kmeans.labels_
error_dict = dict()
for i in range(TEST_CENTER):
        error_dict[str(i)] = np.mean(y_error[labels == i])
#模型可以存下來嗎?        
data = list(error_dict.items())
error_arr = np.array(data)
np.save('./KmeansModel/error_arr0607.npy', error_arr)
#實際運用時要先預測是哪一類        
test_data = np.ones((1,16),float)
test_data_label = test_kmeans.predict(test_data)
predict_error = error_dict[str(test_data_label[0])]
