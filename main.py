import cv2 
import numpy as np
#from google.colab.patches import cv2_imshow
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import glob
from ASLModel import ASLModel
from SelectRoi import staticROI
import random 
from utils import rotate_images, save_images
import argparse
from tqdm import tqdm

app = argparse.ArgumentParser()
app.add_argument("-p", "--path",
	help="Path to the project directory", 
        default = "./")
app.add_argument("-d", "--datapath",
	help="Path to the data directory", 
        default = "")
app.add_argument("-m", "--modelpath",
	help="Path to the saved model h5 file", 
        default = "./savedmodel/SignLanguageModel.h5")
args = vars(app.parse_args())

projectPath = args['path']
datapath = args['datapath']
modelpath = args['modelpath']

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
traindatas = [datapath+'train/train_set1/',datapath+'train/train_set2/',datapath+'train/train_set3/']
testdatas = datapath+'test/'
if __name__ == '__main__':
   
   if datapath != '':
      images=[]
      labels=[]
      images_test=[]
      labels_test=[]
      for d in traindatas:
          print('Loading datas in folder: ', d)
          for i in tqdm(range(len(letters))):
              l = letters[i]
              tmp = [f for f in glob.glob(d+l+"/*.jpg")] + [f for f in glob.glob(d+l+"/*.jpeg")] + [f for f in glob.glob(d+l+"/*.png")]
              images  += tmp
              labels += [l for el in tmp]

      for i in tqdm(range(len(letters))):
          l = letters[i]
          tmp = [f for f in glob.glob(testdatas+l+"/*.jpg")] + [f for f in glob.glob(testdatas+l+"/*.jpeg")] + [f for f in glob.glob(testdatas+l+"/*.png")]
          images_test  += tmp
          labels_test += [l for el in tmp]
          images  += tmp
          labels += [l for el in tmp]
    
      
      model = ASLModel(images, labels,images_test,labels_test, labels=letters)
      model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation = 'relu', input_shape=(ASLModel.dimimages[0], ASLModel.dimimages[1] ,1) ))
      model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
      model.add(MaxPooling2D(pool_size = (2, 2)))
      model.add(Dropout(0.2))

      model.add(Conv2D(64, kernel_size = (3, 3),padding='same', activation = 'relu'))
      model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
      model.add(MaxPooling2D(pool_size = (2, 2)))
      model.add(Dropout(0.2))

      model.add(Conv2D(256, kernel_size = (3, 3), padding='same', activation = 'relu'))
      model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
      model.add(MaxPooling2D(pool_size = (2, 2)))
      model.add(Dropout(0.2))

      model.add(Flatten())
      model.add(Dense(1024, activation = 'relu'))
      model.add(Dropout(0.5))
      model.add(Dense(len(letters), activation = 'softmax'))

      model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
      history = model.fit()
      static_ROI = staticROI(model)
   if modelpath != '':
      model = ASLModel(labels=letters)
      model.loadModelFromPath(modelpath)
      static_ROI = staticROI(model)

      
