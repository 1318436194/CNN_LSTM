# tensorflow_version 2.0
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras.callbacks import CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
engine = create_engine('sqlite:///PI_database.db', echo=True)
parameters = 3 #Use the temperature and humidity collected in the last three moments to predict the value of the next moment
epochs = 500
sql = "SELECT * FROM data ORDER BY timestamp DESC"
df = pd.read_sql(sql=sql, con=engine)
print(df)


def load_data(data, n_prev = 3): 
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev])
        docY.append(data[i+n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

# 60% is Train dataset, 20% is Test dataset
def train_test_split(df, n_prev = 3):
    SAMPLES = len(df)
    TRAIN_SPLIT =  int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)
    x_values,y_values = load_data(df, n_prev)
    x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])
    return (x_train, x_test, x_validate), (y_train, y_test, y_validate)
# Take the temperature and humidity values of the last 400 acquisitions to train the model
(x_train, x_test, x_val), (y_train, y_test, y_val) = train_test_split(df["temp"][0:400].apply(pd.to_numeric), n_prev = parameters)
print(len(x_train))
print(len(y_test))
print(x_train.shape)
print(y_test.shape)
# ANN Network, 32 64 two hidden layers 
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(parameters,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss="mse", metrics=["mae"])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, verbose=2, batch_size=32, callbacks=[
    CSVLogger('Temptrain.log'),
    ModelCheckpoint('Tempmodel.h5', monitor='val_mae', verbose=2, save_best_only=True)
])
# Predict
predicted = model.predict(x_test)
dataf = pd.DataFrame(predicted)
dataf.columns = ["predict"]
dataf["actual"] = y_test
dataf.plot(figsize=(15, 5))

(x_train, x_test, x_val), (y_train, y_test, y_val) = train_test_split(df["humi"][0:400].apply(pd.to_numeric), n_prev = parameters)
print(len(x_train))
print(len(y_test))
print(x_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(parameters,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, verbose=2, batch_size=32, callbacks=[
    CSVLogger('Humitrain.log'),
    ModelCheckpoint('Humimodel.h5', monitor='val_mae', verbose=2, save_best_only=True)
])

predicted = model.predict(x_test)
dataf = pd.DataFrame(predicted)
dataf.columns = ["predict"]
dataf["actual"] = y_test
dataf.plot(figsize=(15, 5))
plt.show()
