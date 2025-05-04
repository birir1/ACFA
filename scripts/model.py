from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Reshape

def create_model(input_shape=(224, 224, 1), lstm_units=64):
    input_layer = Input(shape=input_shape)
    
    # Convolutional Layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    feature_dim = x.shape[-1]
    reshaped = Reshape((1, feature_dim))(x)

    lstm_output = LSTM(lstm_units)(reshaped)
    z = Dense(128, activation='relu')(lstm_output)
    z = Dropout(0.5)(z)
    output = Dense(3, activation='softmax')(z)  # 3 classes

    model = Model(inputs=input_layer, outputs=output)
    return model
