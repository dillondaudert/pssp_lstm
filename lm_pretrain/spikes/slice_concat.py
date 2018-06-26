# slice and concat inputs / outputs in Keras
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Lambda, RNN, SimpleRNNCell, Bidirectional, Masking


x = Input(shape=(None, 4))
mask_x = Masking(mask_value=0.)(x)
# reverse for backwards (wtf)
x_rev = Lambda(lambda l: tf.reverse_sequence(l, tf.constant((5,), dtype=tf.int32), seq_axis=0))(x)
x_rev_mask = Masking(mask_value=0.)(x_rev)

cell = SimpleRNNCell(units=2,
                    activation="linear",
                    kernel_initializer=tf.keras.initializers.constant(10.),
                    recurrent_initializer='ones')
r_fw = RNN(cell, return_sequences=True)(mask_x)
r_bw = RNN(cell, go_backwards=True, return_sequences=True)(mask_x)
y = concatenate([r_fw, r_bw], axis=-1)
r_rev_bw = RNN(cell, return_sequences=True)(x_rev_mask)
#y_bidir = concatenate([r_fw, r_rev_bw], axis=-1)
model = keras.Model(inputs=[x], outputs=[y, r_rev_bw])
model.compile(loss="mse", optimizer="rmsprop")

model.summary()

a = np.ones(shape=(1, 5, 4))
a[0,1,:] *= 2.
a[0,2,:] *= 3.
a[0,3,:] *= 4.
a[0,4,:] *= 5.
a = np.concatenate([a, np.zeros(shape=(1, 2, 4))], axis=1)

x_in = a
print(x_in)
print(model.predict(x_in))
