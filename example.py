import os
import numpy as np
import tensorflow
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (Add, Dense, Flatten, Conv2D, Concatenate,
                                    Input, BatchNormalization, Activation
                                    )

def residual_branch(x, number_of_filters):
    """
      Residual branch with
    """
    # Create skip connection
    x_skip = x
    # 
    x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1,1), padding="same")(x_skip)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(number_of_filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    # Add the skip connection to the regular mapping
    x = Add()([x, x_skip])

    # Nonlinearly activate the result
    x = Activation("relu")(x)

    # Return the result
    return x

def my_model(input_shape = (41, 37, 3), number_of_filters_1 = 3, number_of_filters_2 = 3, nbs_output=1, learning_rate=0.001):

    # Define the input
    X = Input(input_shape, name='input')
    # ResNet branch 1
    X_1 = residual_branch(X, number_of_filters_1) # Aquí iría ir el encoder
    # ResNet branch 2
    X_2 = residual_branch(X, number_of_filters_2) # Aquí iría la red 1D-CNN
    # Concatenation layer
    X_3 = Concatenate(name='concat_meta')([X_1, X_2]) # Aquí se concatenan las salidas
    # Dense layers
    X_3 = Dense(100, activation='relu', name='dense_1')(X_3)
    X_out = Dense(nbs_output, activation='linear', name='dense_2')(X_3)

    model = Model(inputs = X, outputs = X_out, name='my_model')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=['mae'])
    return model

model = my_model()
model.summary()