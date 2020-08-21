# import libraries
import os
import time
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from scipy import stats

from keras.utils import normalize
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding, add
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import tensorflow.keras
import keras


home='C:/Users/einel/Documents/projects/EEG-Eye-State-data-set'
root_logdir = os.path.join(os.curdir, "my_logs")
print(root_logdir) # The "." represents your current directory


def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id), run_id


# Load in the data frame
data_frame=home+'/data/EEG Eye State.arff'
raw_data = loadarff(data_frame)
df_data = pd.DataFrame(raw_data[0])


# Preprocess Data
# split into data and targets
data_targets=df_data.iloc[:,-1]
data=df_data.iloc[:,:-1]

# data normalization with sklearn z-score
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# convert targets into 1 or 0
data_targets=np_utils.to_categorical(data_targets, num_classes=2)

# split the data
xTrain, xTest, yTrain, yTest = train_test_split(data, data_targets, test_size = 0.2)
# print(xTrain, yTrain)


# prepare save files
run_logdir, run_id = get_run_logdir(root_logdir)
# print(run_logdir)
tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(run_logdir)

# create  sequential model
'''model = Sequential()
model.add(Dense(256, input_dim=14))
model.add(Dense(256, activation='relu'))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(2, activation='sigmoid'))
print(model.summary())'''


# create functional model
inputs=keras.Input(shape=(14,))
x=Dense(200, activation='relu')(inputs)
x=Dense(200, activation='relu')(x)
block_1_output = Dense(100, activation='sigmoid')(x)

x=Dense(100, Activation='relu')(block_1_output)
x=Dense(100, activation='relu')(x)
x=Dense(200, activation='relu')(x)

block_2_output = add([x, block_1_output])

outputs = keras.layers.Dense(2)(block_2_output)


model=keras.Model(inputs=inputs, outputs=outputs, name="eeg_model")

keras.utils.plot_model(model, "./data/eeg_model.png", show_shapes=True)


# Try RNN network- needs work
'''model = Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(Dense(100, input_dim=14))

# Add a LSTM layer with 128 internal units.
model.add(LSTM(128))

# Add a Dense layer with 10 units.
model.add(Dense(10))

model.add(Dense(2))'''


model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=xTrain, y=yTrain,epochs=100,batch_size=32, validation_data=(xTest,yTest))

score, acc = model.evaluate(x=xTest, y=yTest)
# print(model.predict(xTest))

print('Accuracy: ',100*(acc))


# back to home directory
# print(home)
# save the model
new_dir=home+'/data/'+str(100*acc)+'_'+run_id+'.h5'
model.save(new_dir)
