import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Reshape, Conv1D, MaxPooling1D, LeakyReLU, Flatten, Dense, Dropout, Input, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

num_classes = 3

# Definimos una red CNN simple
def cnn_simple(input_layer):
    x = tf.keras.layers.Reshape((window_size, 1))(input_layer)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, activation='linear', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)
    return x

def classifier(enco):
    x = enco
    x = Dense(50, activation='linear')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return x


# Definir la LSTM
def lstm_function(input_layer):
    # Reshape para que sea compatible con LSTM
    lstm_input = Reshape((window_size, 1))(input_layer)

    # Capa LSTM con 150 unidades
    lstm_output = LSTM(100, return_sequences=False)(lstm_input)

    return lstm_output

def fc(enco):

    x = enco

    x = Flatten()(x)

    return x

encode = tf.keras.models.Model(inputs=autoencoder.input,
                               outputs=autoencoder.get_layer('bottleneck').output)

# Pasar la misma entrada por la red CNN simple
cnn_output = cnn_simple(entrada)

lstm_output = lstm_function(entrada)


# Concatenar las salidas del encoder y de la CNN
concatenated_output = Concatenate()([fc(encode.output), cnn_output, lstm_output])


# Crear el modelo completo
full_model = Model(inputs=entrada, outputs=classifier(concatenated_output))
encode = tf.keras.models.Model(entrada, encode.output)

# Freeze the layers in the encoder model
for layer in encode.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

full_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

X_train_encoded = full_model.predict(X_train_noisy)
X_val_encoded = full_model.predict(X_val_noisy)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce la tasa de aprendizaje a la mitad
    patience=5,  # Espera 5 épocas antes de reducir la tasa de aprendizaje
    min_lr=1e-6  # No reduce la tasa de aprendizaje por debajo de este valor
)


# Add Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # The performance metric to monitor
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

batch_size = 64  # O prueba con 16, 128, etc.

# Train the classifier with early stopping
classifier_history = full_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=batch_size,
    #callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict,
    verbose=1
)

# Evaluate the model
loss, accuracy = full_model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
