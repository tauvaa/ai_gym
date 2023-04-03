import numpy as np
import tensorflow as tf

keras = tf.keras


def get_reg_values(beta):
    beta = np.array(beta)
    X = np.random.rand(10000, len(beta))
    Y = np.matmul(X, beta)
    return X, Y


X, Y = get_reg_values([4, 3, 5])

model = keras.Sequential([keras.layers.Dense(1)])
model.compile(
    loss=keras.losses.MeanSquaredError(), optimizer="adam"
)
Y = Y.astype(np.float32)
print(Y)
model.fit(X, Y, epochs=100)
model.summary()
print(model.get_weights())
