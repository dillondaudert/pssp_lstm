# slice and concat inputs / outputs in Keras
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Lambda, RNN, SimpleRNNCell, Bidirectional, Masking


x = Input(shape=(None, 4), batch_size=1)
s_len = Input(shape=(1,), dtype="int32", batch_size=1)
mask_x = Masking(mask_value=0.)(x)

cell = SimpleRNNCell(units=1,
                    activation="linear",
                    kernel_initializer=tf.keras.initializers.constant(1.),
                    recurrent_initializer='zeros')
#fw_out = RNN(cell, return_sequences=True)(mask_x)
#bw_out = RNN(cell, return_sequences=True, go_backwards=True)(mask_x)
fw_out, bw_out = Bidirectional(RNN(cell, return_sequences=True), merge_mode=None)(mask_x)

# lambda to remove first 2 tokens of sequences
lam_cut = Lambda(lambda l: l[:, 2:, :])

# lambda to perform reverse_sequence
lam_rev = Lambda(lambda l, s_len: tf.reverse_sequence(l, tf.reshape(s_len, shape=(-1,)), seq_axis=1))
lam_rev.arguments = {"s_len": s_len}

# reverse tokens (not padding) of fw pass
rev_fw = lam_rev(fw_out)
# cut fw and bw
cut_rev_fw = lam_cut(rev_fw)
cut_bw_out = lam_cut(bw_out)
# we removed 2 from fw, so reduce seq_len
lam_rev.arguments = {"s_len": s_len - tf.constant(2, dtype=tf.int32)}
# reverse tokens of fw back to original
cut_fw_out = lam_rev(cut_rev_fw)


y = concatenate([cut_fw_out, cut_bw_out], axis=-1)
model = keras.Model(inputs=[x, s_len], outputs=[y, fw_out, rev_fw, cut_rev_fw])
model.compile(loss="mse", optimizer="rmsprop")

model.summary()

a = np.ones(shape=(1, 5, 4))
a[0,1,:] *= 2.
a[0,2,:] *= 3.
a[0,3,:] *= 4.
a[0,4,:] *= 5.
a = np.concatenate([np.ones(shape=(1, 1, 4))*-1, a, np.ones(shape=(1, 1, 4))*-2], axis=1)
a = np.concatenate([a, np.zeros(shape=(1, 2, 4))], axis=1)
s_len = np.array([7,], dtype=int)

x_in = a
print(x_in)
print(model.predict([x_in, s_len]))
