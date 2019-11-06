import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

projectPath = "/content/drive/My Drive/INF573-Project"

train = pd.read_csv(f"{projectPath}/sign-language-mnist/sign_mnist_train.csv")
test = pd.read_csv(f"{projectPath}/sign-language-mnist/sign_mnist_test.csv")


labels = train['label'].values
from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

train.drop('label', axis = 1, inplace = True)
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

batch_size = 128
classes = 24
epochs = 50
x_train = x_train / 255
x_test = x_test / 255

x_train.shape
x_test.shape

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train.shape



model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(classes, activation = 'softmax'))



model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


test_image= cv2.imread(f"{projectPath}/a2.jpg", cv2.IMREAD_GRAYSCALE)
cv2_imshow(test_image)



width = 28
height = 28
dim = (width, height)
resized = cv2.resize(test_image, dim, interpolation = cv2.INTER_AREA)
cv2_imshow(resized)


test_images = np.array([resized.flatten()])
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


y_pred = model.predict(test_images)


