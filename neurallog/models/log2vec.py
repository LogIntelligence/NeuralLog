from tensorflow.keras import layers
from tensorflow import keras

def deeplog_model(h=10, no_events=500, dropout=0.1):
    inputs = layers.Input(shape=(h, 300))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(no_events, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model