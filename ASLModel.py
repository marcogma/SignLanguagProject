from keras import Sequential
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

class ASLModel(Sequential):
      dimimages = (28,28)
      # Receives a data set and the labels
      def __init__(self, X = None, Y = None, X_test = None, Y_test = None, labels=[]):
          super(ASLModel, self).__init__()
          self.X = X
          self.Y = Y
          self.X_test = X_test
          self.Y_test = Y_test
          self.labels = labels
          self.stat = dict()
          self.preprocessed = False
          self.modello = None

      def fit(self, epochs = 10,batch_size = 25, savingpath = ''):
          print('\n---------------> START FITTING DATA <---------------')
          if self.preprocessed == False:
             self.X = ASLModel.preprocessX(self.X)
             self.X_test = ASLModel.preprocessX(self.X_test)
             self.Y, self.stat = ASLModel.preprocessY(self.Y, self.labels)
             self.Y_test, tmp = ASLModel.preprocessY(self.Y_test, self.labels)
             self.preprocessed = True
          self.trainStats()
          super().fit(self.X, self.Y, validation_data = (self.X_test, self.Y_test), epochs=epochs, batch_size=batch_size)
          print('\n---------------> END FITTING DATA <---------------')
          if savingpath != '':
             super().save(savingpath)
             print('\nModel saved in ', savingpath)

      def translate(self, input_img):
          img = np.zeros([1,input_img.shape[0],input_img.shape[1],input_img.shape[2]], dtype=input_img.dtype)
          img[0] = input_img
          gray = ASLModel.preprocessX(img)[0]
          gray = gray.reshape(1,gray.shape[0],gray.shape[1],gray.shape[2])
          if self.modello == None:
             y = super().predict(gray)
          else:
             y = self.modello.predict(gray)
          return self.labels[np.argmax(y)]

      def trainStats(self):
          tot = len(self.Y)
          sizes = self.stat.values()
          fig1, ax1 = plt.subplots()
          ax1.pie(sizes, labels=self.labels, autopct='%1.1f%%',shadow=True, startangle=90)
          ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
          plt.show()
          plt.imshow(self.X[0].reshape(self.X[0].shape[0], self.X[0].shape[1]))
          plt.show()
          return ''

      def getDataset(self):
          return self.X, self.Y

      def loadModelFromPath(self, path = ''):     
          self.modello = load_model(path)

      @staticmethod
      def preprocessX(X):
          print('\n|           PREPROCESSING DATA            |')
          X_processed = ASLModel.toGray(X)
          X_processed = ASLModel.resize2D(X_processed)
          print('\n|           END PREPROCESSING DATA            |')
          return X_processed

      @staticmethod
      def preprocessY(Y, labels):
          print('\n|           PREPROCESSING LABELS DATA            |')
          stat = dict()
          link = dict()
          count = 0
          for c in labels:
              stat[c] = 0
              link[c] = count
              count += 1
          Y_int = np.zeros([Y.shape[0],count], dtype=np.int16)
          for i in tqdm(range(Y.shape[0])):
              Y_int[i, link[Y[i]]] = 1
              stat[Y[i]] += 1
              
          print('\n|           END PREPROCESSING LABELS DATA            |')
          return Y_int, stat

      # Set of Images ----> Set of resized images
      @staticmethod
      def resize2D(X):
          print('\nSTART RESIZING 2D DATA')
          resized = np.zeros([X.shape[0],ASLModel.dimimages[0], ASLModel.dimimages[1],1], dtype=X.dtype)
          for i in tqdm(range(0,len(X))):
              rz = cv2.resize(X[i], ASLModel.dimimages, interpolation = cv2.INTER_AREA)
              resized[i] = rz.reshape(ASLModel.dimimages[0], ASLModel.dimimages[1], 1)
          print('\nSTOP RESIZING 2D DATA')
          return resized

      @staticmethod
      def toGray(X):
          print('\nSTART CONVERTION TO GRAY SCALE')
          print(X.shape)
          grays = np.zeros([X.shape[0],X.shape[1], X.shape[2]], dtype=np.uint8)
          for i in tqdm(range(0,len(X))):
              grays[i] = cv2.cvtColor(X[i],cv2.COLOR_BGR2GRAY)
          print('\nSTOP CONVERTION TO GRAY SCALE')
          return grays


