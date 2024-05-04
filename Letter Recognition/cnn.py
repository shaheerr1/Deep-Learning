import numpy as np
import pandas as pd
import seaborn as sn
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


test_data = np.load("C:/numpy/test_data.npy")
test_labels = np.load("C:/numpy/test_labels.npy")
train_data = np.load("C:/numpy/train_data.npy")
train_labels = np.load("C:/numpy/train_labels.npy")


model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dense(36, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
# model.summary()


best_loss_checkpoint = ModelCheckpoint(
    filepath="C:/best_loss_checkpoint/best_loss_model.h5",
    monitor="loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min"
)

best_val_loss_checkpoint = ModelCheckpoint(
    filepath="C:/best_val_loss_checkpoint/best_val_loss_model.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min"
)


# history = model.fit(
#     train_data,
#     train_labels,
#     validation_data=(test_data, test_labels),
#     epochs=10,
#     batch_size=200,
#     callbacks=[best_loss_checkpoint, best_val_loss_checkpoint]
# )


# plt.plot(history.history["acc"], 'b', label="acc")
# plt.plot(history.history["val_acc"], 'r', label="val_acc")
# plt.xlabel("epoch")
# plt.ylabel("frequency")
# plt.legend()
# plt.show()


# model.load_weights("C:/best_val_loss_checkpoint/best_val_loss_model.h5")
# loss, acc = model.evaluate(test_data, test_labels)
# print(loss, acc)


# predictions = model.predict(test_data)
