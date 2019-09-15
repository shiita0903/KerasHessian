import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Dense


def build_model() -> Model:
    model = Sequential([
        Dense(2, use_bias=False, activation="relu", name="Dense1", input_shape=(2,)),
        Dense(2, use_bias=False, activation="softmax", name="Dense2")
    ])
    model.summary()
    model.set_weights([
        np.array([[1, 2], [3, 4]]),  # Dense1の重み
        np.array([[6, 5], [7, 8]]),  # Dense2の重み
    ])
    return model


def main():
    inputs = np.array([[1, 2]])
    labels = np.array([[0, 1]])
    model = build_model()

    y_true = K.placeholder((None, 2,))  # one-hotなラベルを入れるPlaceholder
    loss = K.categorical_crossentropy(y_true, model.output)
    gradient = K.gradients(loss, model.get_layer("Dense1").kernel)[0]

    # KerasのバックエンドのAPIでヘシアンを求める関数はないので、TensorFlowの関数を利用する
    hessian = tf.hessians(loss, model.get_layer("Dense1").kernel)[0]
    s = hessian.shape
    hessian = K.reshape(hessian, [s[0] * s[1], s[2] * s[3]])  # 4次元テンソルを2次元に整形

    with K.get_session():
        print("===== gradient =====")
        print(gradient.eval({model.input: inputs, y_true: labels}))
        print("===== hessian =====")
        print(hessian.eval({model.input: inputs, y_true: labels}))


if __name__ == "__main__":
    main()
