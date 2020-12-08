
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import datetime



def build_model():
    """
    Build NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    input = tf.keras.layers.Input(shape=(8,))

    '''x = tf.keras.layers.Dense(256, activation='relu')(input)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)'''

    # best sf is no dropout, 128
    x = tf.keras.layers.Dense(256, activation='relu')(input)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    output = tf.keras.layers.Dense(2, activation='relu')(x)

    #x = tf.keras.layers.Dense(32, activation='relu')(x)

    #output = tf.keras.layers.Dense(2)(x)
    #output = tf.keras.layers.Softmax()(x)
    #output = tf.keras.layers.Lambda(lambda x: (x*9)+9)(x)

    model = tf.keras.models.Model(inputs=input, outputs=output, name="football_player")
    return model

def normalize_y(y):
    y_norm = (y-9)/9
    return y_norm
#
lr = 1e-4
epochs = 150
batch_size = 128
logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))

# Load expert data
df = pd.read_csv('./data/plays_defense_expert1.csv')

pre_X = df.drop(['action'],axis=1).values

df.action = df.action.astype(int)

pre_y = df['action'].values



y = []
X = []

for i, e in enumerate(pre_y):
    if e == 2:
        y.append([0, 0])
        X.append(pre_X[i])
    elif e == 3:
        y.append([0, 1])
        X.append(pre_X[i])
    elif e == 4:
        y.append([0, 2])
        X.append(pre_X[i])
    elif e == 1:
        y.append([1, 0])
        X.append(pre_X[i])      
    elif e == 16:
        y.append([1, 1])
        X.append(pre_X[i])
    elif e == 5:
        y.append([1, 2])
        X.append(pre_X[i])
    elif e == 8:
        y.append([2, 0])
        X.append(pre_X[i])
    elif e == 7:
        y.append([2, 1])
        X.append(pre_X[i])
    elif e == 6:
        y.append([2, 2])
        X.append(pre_X[i])

X = np.array(X)
y = np.array(y)

######
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.02,random_state=42)

#y_train = normalize_y(y_train)
#y_test = normalize_y(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(46,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(19)
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(46,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(6)
    # og 19
])

model = build_model()

model.compile(optimizer='adam',
		loss=['mse'],
        metrics=['mae'])

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

# save checkpoint callback
checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir, 'weights.h5'),
																monitor='mae',
																verbose=0,
																save_best_only=True,
																save_weights_only=False,
																mode='auto',
																save_freq=1)

# do training for the specified number of epochs and with the given batch size
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
		validation_data=(X_test, y_test),
		callbacks=[tbCallBack, checkpointCallBack])  # add this extra parameter to the fit function

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

'''probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

arr = []
arr.append(X_test[0])
arr = np.asarray(arr)
print(arr)
predictions = probability_model.predict(arr)
print(np.argmax(predictions[0]))'''

arr = []
arr.append(X_test[0])
arr = np.asarray(arr)
predictions = model.predict(arr)

print(y_test[0])
print(predictions)

ydir = int(round(predictions[0][0]))
xdir = int(round(predictions[0][1]))

print(ydir)
print(xdir)

model.save(filepath="./agent/saved_models/defensive-movement.h5")


