import cv2
from keras.models import load_model
import numpy as np

alfabet = ['a','b','c','d', 'e', 'f','g','h','i','k', 'l', 'm','n','o','p','q','r', 's', 't','u','v','w','x','y']
def translate(img, model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (28, 28)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    gray = np.array([resized.flatten()])
    gray = gray.reshape(gray.shape[0], 28, 28, 1)
    y = model.predict(gray)
    cols = y.shape[1]
    print(cols, len(alfabet))
    for i in range(cols):
        if y[0,i] > 0.98:
           return alfabet[i]
    return ""

def show_webcam(model,mirror = False):
    cam = cv2.VideoCapture(0)
    
    width = cv2.VideoCapture.get(cam,cv2.CAP_PROP_FRAME_WIDTH)
    height = cv2.VideoCapture.get(cam,cv2.CAP_PROP_FRAME_HEIGHT) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (255,255,255)
    lineType = 2
    bottomLeftCornerOfText = (0,int(height*0.95))
    while True:
          ret_val, img = cam.read()
          if mirror: 
             img = cv2.flip(img, 1)
          cv2.putText(img,translate(img,model), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
          cv2.imshow('ASL Letters', img)
          if cv2.waitKey(1) == 27:
             break
    cv2.destroyAllWindows()


def main():
    model = load_model('./savedmodel/SignLanguageModel.h5')
    show_webcam(model,mirror=True)
if __name__ == '__main__':
    	main()

