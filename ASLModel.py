from keras import Sequential
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

class ASLModel(Sequential):
      dimimages = (64,64)
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

      def fit(self, epochs = 2,batch_size = 32, savingpath = ''):
          print('\n---------------> START FITTING DATA <---------------')
          if self.preprocessed == False:
             self.X = ASLModel.preprocessX(self.X)
             self.X_test = ASLModel.preprocessX(self.X_test)
             self.Y, self.stat = ASLModel.preprocessY(self.Y, self.labels)
             self.Y_test, tmp = ASLModel.preprocessY(self.Y_test, self.labels)
             self.preprocessed = True
             self.trainStats()
          self.X = self.X.reshape(self.X.shape[0],self.X.shape[1], self.X.shape[2], 1)
          self.X_test = self.X_test.reshape(self.X_test.shape[0],self.X_test.shape[1], self.X_test.shape[2], 1)
          history = super().fit(self.X, self.Y, validation_data = (self.X_test, self.Y_test), 
                      epochs=epochs, batch_size=batch_size)
          print('\n---------------> END FITTING DATA <---------------')
          if savingpath != '':
             super().save(savingpath)
             print('\nModel saved in ', savingpath)
          y_pred = super().predict(self.X_test)
          print('\n', classification_report(np.where(self.Y_test > 0)[1],  np.argmax(y_pred, axis=1), target_names=self.labels))
          plt.figure(figsize=(8,8))
          cnf_matrix = confusion_matrix(np.where(self.Y_test > 0)[1], np.argmax(y_pred, axis=1))
          plt.imshow(cnf_matrix, interpolation='nearest')
          plt.colorbar()
          tick_marks = np.arange(len(self.labels))
          _ = plt.xticks(tick_marks, self.labels, rotation=90)
          _ = plt.yticks(tick_marks, self.labels)
          return history
      def translate(self, input_img):
          gray = ASLModel.preprocessX(input_img,False)[0]
          gray = gray.reshape(1,gray.shape[0],gray.shape[1],1)
          if self.modello == None:
             y = super().predict(gray)
          else:
             y = self.modello.predict(gray)
          ind1 = np.argmax(y)
          prob1 = y[0,np.argmax(y)]
          y[0,ind1] = -1
          ind2 = np.argmax(y)
          prob2 = y[0,np.argmax(y)]
          y[0,ind2] = -1
          ind3 = np.argmax(y)
          prob3 = y[0,np.argmax(y)]
          y[0,ind3] = -1
          return self.labels[ind1], prob1,self.labels[ind2], prob2,self.labels[ind3], prob3

      def trainStats(self):
          tot = len(self.Y)
          sizes = self.stat.values()
          fig1, ax1 = plt.subplots()
          ax1.pie(sizes, labels=self.labels, autopct='%1.1f%%',shadow=True, startangle=90)
          ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
          plt.show()
          plt.imshow(self.X[0].reshape(self.X[0].shape[0], self.X[0].shape[1]),cmap = 'gray')
          plt.show()
          return ''

      def getDataset(self):
          return self.X, self.Y

      def loadModelFromPath(self, path = ''):     
          self.modello = load_model(path)

      @staticmethod
      def preprocessX(images, path=True):
          print('\n|           PREPROCESSING DATA            |')
          if path == False and len(images.shape) == 3:
             images = [images]
          X_processed = []
          if path == True:
             for i in tqdm(range(len(images))):
                 f = images[i]
                 X_processed.append(cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), ASLModel.dimimages, interpolation = cv2.INTER_AREA))
          else:
             for i in tqdm(range(len(images))):
                 X_processed.append(cv2.resize(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY), ASLModel.dimimages, interpolation = cv2.INTER_AREA))
          X_processed = np.array(X_processed)
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
          Y_int = np.zeros([len(Y),count], dtype=np.int16)
          for i in tqdm(range(len(Y))):
              Y_int[i, link[Y[i]]] = 1
              stat[Y[i]] += 1
              
          print('\n|           END PREPROCESSING LABELS DATA            |')
          return Y_int, stat


