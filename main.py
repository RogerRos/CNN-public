from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# arquitectura de la red neuronal convulcional per distingir entre gossos y gats.

def IA_create():
    model = Sequential()
    
    model.add(Conv2D(64,(5,5), input_shape=(64,64,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(4,4)))

    model.add(Conv2D(128,(4,4), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Conv2D(256,(2,2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation="relu"))

    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

IA = IA_create()

IA.summary()

# entrenament de la red neuronal convulcional per distingir entre cotxes y motos.

d_entrenament = "C:/Users/roger/OneDrive/Documents/TDR/Red neuronal convulcional/dades/entrenament"
d_test = "C:/Users/roger/OneDrive/Documents/TDR/Red neuronal convulcional/dades/test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    d_entrenament,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    d_test,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

IA.fit(train_generator, epochs=30, validation_data=test_generator)

IA.save("C:/Users/roger/OneDrive/Documents/TDR/Red neuronal convulcional/model_IA")
