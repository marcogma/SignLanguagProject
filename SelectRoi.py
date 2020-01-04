# ASL Sign Language Prediction
# Marco Garcia
#Christian Bile

import cv2;
import numpy as np;
import random as rng

from ASLModel import ASLModel


bg=None

class staticROI(object):
    def __init__(self, translator=None):
        self.capture = cv2.VideoCapture(0)

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False

        self.roi_hist=None
        self.croppedIm=[]
        self.translator = translator #The classifier
        self.update()

    def update(self):
        flag=False
        track_window = (250,90,400,125)
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        cascade = cv2.CascadeClassifier('rpalm.xml')


        #setting of letters display
    
        width = cv2.VideoCapture.get(self.capture,cv2.CAP_PROP_FRAME_WIDTH)
        height = cv2.VideoCapture.get(self.capture,cv2.CAP_PROP_FRAME_HEIGHT) 

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.9
        fontColor = (255,255,255)
        lineType = 2
        bottomLeftCornerOfText = (0,int(height*0.95))

        while True:
            if self.capture.isOpened():
                # Read frame
                (self.status, self.frame) = self.capture.read()
                self.frame = cv2.flip(self.frame, 1)
                key = cv2.waitKey(2)

                # Close program with keyboard 'q'
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit()


                # Crop and display cropped image
                if flag:
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
                    edges = cv2.Canny(dst,0,100)

                    #cv2.imshow('canny edges',edges)
                    rect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect_kernel)
                    #cv2.imshow('img after closing the edges',edges)

                    cv2.imshow('img dst',dst)
                    ret, thresh = cv2.threshold(dst, 100, 255, 0)
                    contours,hierarchy=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

                    

                    contours_poly = [None]*len(contours)
                    boundRect = [None]*len(contours)
                    centers = [None]*len(contours)
                    radius = [None]*len(contours)
                    for i, c in enumerate(contours):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        boundRect[i] = cv2.boundingRect(contours_poly[i])
                    drawing = np.zeros(thresh.shape, np.uint8)
    

    
                    for i in range(len(contours)):
                        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                        cv2.drawContours(drawing, contours_poly, i, color)
                    
                    #cv2.imshow('Contours', drawing)
                    



                    if len(contour_sizes)>0:
                        mask = np.zeros(thresh.shape, np.uint8)
                        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
                        bRect= cv2.boundingRect(cv2.approxPolyDP(biggest_contour, 3, True))
                        cv2.drawContours(drawing, contours_poly, i, color)
                        cv2.rectangle(self.frame, (int(bRect[0]), int(bRect[1])),(int(bRect[0]+bRect[2]), int(bRect[1]+bRect[3])), color, 2)

                        cropp_img = self.frame[int(bRect[1]):int(bRect[1]+bRect[3]), int(bRect[0]):int(bRect[0]+bRect[2])]
                        cv2.imshow("cropped", cropp_img)


                        cv2.drawContours(mask, contours, -1, 255, -1)
                        #cv2.imshow('mask', mask)

                        ret, track_window = cv2.CamShift(mask, track_window, term_crit)
                        pts = cv2.boxPoints(ret)
                        pts = np.int0(pts)

                        if self.translator != None:
                # TRANSLATION OF CROPED IMAGE
                            l1,p1,l2,p2,l3,p3 = self.translator.translate(cropp_img)
                            s = str(l1)+': '+str(round(p1,3))+str('    ')+str(l2)+': '+str(round(p2,3))+str('    ')+str(l3)+': '+str(round(p3,3))
                            cv2.putText(self.frame,s, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)




                else:
                    while flag==False:
                        (self.status, self.frame) = self.capture.read()
                        key = cv2.waitKey(2)
                        clone = self.frame.copy()
                        flag=self.crop_ROI(cascade)
                        if flag:
                            self.setup_ROI_tracking()
                        cv2.imshow('image', self.frame)



                cv2.imshow('image', self.frame)

            else:
                pass


    def crop_ROI(self, cascade):
        flag=False
        self.cropped_image = self.frame.copy()
        gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
        rectangles = cascade.detectMultiScale(gray)
        
        for (x,y,w,h) in rectangles:
            self.image_coordinates = [(x+w//4,y+h//2)]
            self.image_coordinates.append((x+3*w//4,y+3*h//4))
            flag=True

        return flag


    def setup_ROI_tracking(self):
        roi = self.frame[self.image_coordinates[0][1]:self.image_coordinates[1][1], self.image_coordinates[0][0]:self.image_coordinates[1][0]]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
        #cv2.imshow('cropped image', roi)


if __name__ == '__main__':
    static_ROI = staticROI()
