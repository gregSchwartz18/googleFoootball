
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import datetime


def build_model(num_inputs):
    """
    Build NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    input = tf.keras.layers.Input(shape=(46,))

    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform')(input)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform')(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform')(x)

    output = tf.keras.layers.Dense(19)(x)
    #output = tf.keras.layers.Softmax()(x)
    #output = tf.keras.layers.Lambda(lambda x: (x*9)+9)(x)

    model = tf.keras.models.Model(inputs=input, outputs=output, name="football_player")
    return model

def normalize_y(y):
    y_norm = (y-9)/9
    return y_norm
#
lr = 1e-4
epochs = 100
batch_size = 128
logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))

# Load expert data
df = pd.read_csv('plays_head_dist.csv')
X = df.drop(['action'],axis=1).values
df.action = df.action.astype(int)
y = df['action'].values


######
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.02,random_state=42)

#y_train = normalize_y(y_train)
#y_test = normalize_y(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(46,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(19)
])

model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

# save checkpoint callback
checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir, 'weights.h5'),
																monitor='accuracy',
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

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

arr = []
arr.append(X_test[0])
arr = np.asarray(arr)
print(arr)
predictions = probability_model.predict(arr)
print(np.argmax(predictions[0]))

model.save(filepath="latest_model.h5")


