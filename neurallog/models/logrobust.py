from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input, Lambda, TimeDistributed, Flatten, Activation, \
    RepeatVector, Permute, multiply
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def logrobust_model(max_len=75, num_hidden=128):
    _input = Input(shape=(max_len, 300))
    x = Bidirectional(LSTM(num_hidden, return_sequences=True, dropout=0.5),
                      merge_mode='concat')(_input)
    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(num_hidden * 2)(attention)
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = multiply([x, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_hidden * 2,))(sent_representation)

    pred = Dense(1, activation='sigmoid')(sent_representation)

    model = Model(inputs=_input, outputs=pred)

    return model
