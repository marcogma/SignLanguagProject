import cv2 
import numpy as np
#from google.colab.patches import cv2_imshow
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import glob
from ASLModel import ASLModel
import random 
from utils import rotate_images, save_images
import argparse

app = argparse.ArgumentParser()
app.add_argument("-p", "--path",
	help="Path to the project directory", 
        default = "/content/drive/My Drive/INF573-Project")
app.add_argument("-d", "--datapath",
	help="Path to the data directory", 
        default = "/content/drive/My Drive/INF573-Project/Data")
app.add_argument("-m", "--modelpath",
	help="Path to the saved model h5 file", 
        default = "")
args = vars(app.parse_args())

projectPath = args['path']
datapath = args['datapath']
modelpath = args['modelpath']

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

if __name__ == '__main__':
   
   if modelpath == '':
      X = np.array([cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")] + [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")]+
                [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")] + [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")] + 
                [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")] + [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")]+
                [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")] + [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")]+
                [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")] + [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")])
      X_test = np.array([cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")] + [cv2.imread(img) for img in glob.glob(f"{datapath}/D/*.jpg")]) 
      Y = np.array([letters[random.randint(0, 23)] for i in range(X.shape[0])])
      Y_test = np.array([letters[random.randint(0, 23)] for i in range(X_test.shape[0])])
      model = ASLModel(X, Y,X_test,Y_test, labels=letters)
      model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
      model.add(MaxPooling2D(pool_size = (2, 2)))

      model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
      model.add(MaxPooling2D(pool_size = (2, 2)))

      model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
      model.add(MaxPooling2D(pool_size = (2, 2)))

      model.add(Flatten())
      model.add(Dense(128, activation = 'relu'))
      model.add(Dropout(0.20))
      model.add(Dense(24, activation = 'softmax'))

      model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
      history = model.fit()   
      model.fit()
   else:
      model.loadModelFromPath(modelpath)
      print(model.translate(cv2.imread('Data/l.jpg')))


