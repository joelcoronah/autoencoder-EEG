from re import X
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
    
    # Primera capa convolucional con BatchNormalization
    x = tf.keras.layers.Conv1D(16, kernel_size=3, activation='linear', padding='same', kernel_regularizer=l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.05)(x) 

    ## Segunda capa convolucional
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.ActivityRegularization(l1=0.004)(x)

    x = tf.keras.layers.Dropout(0.05)(x)
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
    lstm_input = Reshape((window_size, 1))(input_layer)
    
    # Capa LSTM con regularización L2
    lstm_output = LSTM(100, return_sequences=False)(lstm_input)
    lstm_output = BatchNormalization()(lstm_output)  # Añadir BatchNormalization
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

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

full_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

X_train_encoded = full_model.predict(X_train_noisy)
X_val_encoded = full_model.predict(X_val_noisy)

# Reducir la tasa de aprendizaje en caso de estancamiento
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Early Stopping para evitar sobreentrenamiento
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Entrenar el modelo
classifier_history = full_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    callbacks=[early_stopping, reduce_lr],  # Activar ReduceLROnPlateau y EarlyStopping
    class_weight=class_weights_dict,
    verbose=1
)

# Evaluar el modelo
loss, accuracy = full_model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

Class  Precision  Sensitivity  Specificity  F1 Score
0      NORMAL   0.915584     0.915584     0.942668  0.933002
1  INTERICTAL   0.927731     0.927731     0.952009  0.920767
2       ICTAL   0.982699     0.982699     0.995823  0.959459