import numpy as np 
import tensorflow as tf 

def find(x):
    mean = []
    std = []
    for i in range(x.shape[-1]):
        mean.append(x[:, :, i].mean())
        std.append(x[:, :, i].std())
    mean = np.array(mean)
    std = np.array(std)
    return mean, std

def n(data, factor):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :] -= factor[0]
            data[i, j, :] /= factor[1]
    return data

def Model(x_shape, y_shape):
    x = tf.keras.Input(x_shape)
    
    lx = tf.keras.layers.Dense(8, activation="sigmoid")(x[:, :, 0:3])
    lx = tf.keras.layers.LSTM(4, return_sequences=True)(lx)
    lx = tf.keras.layers.LSTM(3, return_sequences=True)(lx)
    lx = tf.keras.layers.LSTM(2, return_sequences=True)(lx)

    ly = tf.keras.layers.Dense(8, activation="sigmoid")(x[:, :, 3][:, :, np.newaxis])
    ly = tf.keras.layers.LSTM(4, return_sequences=True)(ly)
    ly = tf.keras.layers.LSTM(3, return_sequences=True)(ly)
    ly = tf.keras.layers.LSTM(2, return_sequences=True)(ly)

    l = tf.keras.layers.Concatenate(axis=-1)([lx, ly])
    l = tf.keras.layers.Flatten()(l)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.Dense(16, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(0.1))(l)

    # l = tf.keras.layers.LSTM(8, return_sequences=True)(x)
    # l = tf.keras.layers.LSTM(16)(l)
    # l = tf.keras.layers.Dense(16, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(0.1))(l)

    o = tf.keras.layers.Dense(y_shape, activation="linear")(l)

    return tf.keras.Model(inputs=x, outputs=o)

def scheduler(epoch, lr):
    if(epoch%20 == 0):
        return 0.001
    else:
        return lr * (0.7)

def train(model, x, y, valid_data=None):
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=100, restore_best_weights=True))
    # callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", verbose=1, factor=0.8, patience=3))
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1))
    model.summary()
    op = tf.keras.optimizers.Nadam()
    model.compile(optimizer=op, loss="mse")
    history = model.fit(x, y, batch_size=8, epochs=10000, validation_data=valid_data, callbacks=callbacks, verbose=2)
    return history

if(__name__ == "__main__"):
    x_train = np.load("data_spline_train/X_train.npy")
    y_train = np.load("data_spline_train/Y_train.npy")
    y_train_org = np.load("data_spline_train/Y_train.npy")
    
    mean_y = y_train.mean()
    std_y = y_train.std()
    print(mean_y, std_y)
    # input()
    y_train = (y_train-mean_y)/std_y
    # print(y_train.mean(), y_train.std())

    x_v = np.load("data_spline_train/X_valid.npy")
    y_v = np.load("data_spline_train/Y_valid.npy")
    y_v = (y_v-mean_y)/std_y
    # print(y_v.mean(), y_v.std())

    x_t = np.load("data_spline_train/X_test.npy")
    y_t = np.load("data_spline_train/Y_test.npy")

    # model = Model(x_train.shape[1::], 1)
    # # model.summary()
    # # input()
    # history = train(model, x_train, y_train, (x_v, y_v))
    # model.save("./model/delta0424_valloss3_mse.h5")
    # y_result = model.predict(x_t, batch_size=32)
    # y_result = y_result*std_y+mean_y
    # np.savetxt("y_delta.csv", y_result, delimiter=",")
    # np.savetxt("y_test.csv", y_t, delimiter=",")
    # print(np.abs(y_t - y_result).mean())

