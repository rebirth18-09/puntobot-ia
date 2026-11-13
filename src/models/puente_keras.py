"""Práctica de Keras con cuatro intenciones; no reemplaza el modelo elegido."""
import keras

def crear_modelo_puente(n_caracteristicas, config, learning_rate=None):
    entradas = keras.Input(shape=(n_caracteristicas,))
    x = entradas
    for neuronas in config["puente_keras"]["neuronas_ocultas"]:
        x = keras.layers.Dense(neuronas, activation="relu")(x)
    salidas = keras.layers.Dense(len(config["clases"]), activation="softmax")(x)
    modelo = keras.Model(entradas, salidas, name="puntobot_puente_keras")
    tasa = config["entrenamiento"]["learning_rate"] if learning_rate is None else learning_rate
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=tasa),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo
