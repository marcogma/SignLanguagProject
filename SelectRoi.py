import cv2;
import numpy as np;


bg=None

class HandDetector():
    """Implement a HandDetector."""

    def __init__(self):
        """Initialize."""
        self.visited =[]



def show_webcam(mirror = False):
    cam = cv2.VideoCapture(0)
    
    width = cv2.VideoCapture.get(cam,cv2.CAP_PROP_FRAME_WIDTH)
    height = cv2.VideoCapture.get(cam,cv2.CAP_PROP_FRAME_HEIGHT) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (255,255,255)
    lineType = 2
    bottomLeftCornerOfText = (0,int(height*0.95))

    cascade = cv2.CascadeClassifier('right.xml')

    num_frames = 0
    aWeight = 0.5
    top, right, bottom, left = 10, 350, 225, 590

    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)


        # clone the frame
        clone = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rectangles = cascade.detectMultiScale(gray)


        for (x,y,w,h) in rectangles:
            clone = cv2.rectangle(clone,(x,y),(x+w,y+h),(255,0,0),2)
            # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)


        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()



class staticROI(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False

        self.roi_hist=None
        self.update()

    def update(self):
        flag=False
        track_window = (250,90,400,125)
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        cascade = cv2.CascadeClassifier('rpalm.xml')

        while True:
            if self.capture.isOpened():
                # Read frame
                (self.status, self.frame) = self.capture.read()
                self.frame = cv2.flip(self.frame, 1)
                key = cv2.waitKey(2)

                # Close program with keyboard 'q'
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(1)


                # Crop and display cropped image
                if flag:
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)

                    # apply meanshift to get the new location
                    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                    # Draw it on image
                    pts = cv2.boxPoints(ret)
                    pts = np.int0(pts)
                    img2 = cv2.polylines(self.frame,[pts],True, 255,2)
                    #cv2.imshow('img2',img2)
                else:
                    while flag==False:
                        (self.status, self.frame) = self.capture.read()
                        key = cv2.waitKey(2)
                        clone = self.frame.copy()
                        #key = cv2.waitKey(2)
                        flag=self.crop_ROI(cascade)
                        if flag:
                            self.setup_ROI_tracking()
                        cv2.imshow('image', self.frame)



                cv2.imshow('image', self.frame)

            else:
                pass

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False

            self.selected_ROI = True

            # Draw rectangle around ROI
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.frame.copy()
            self.selected_ROI = False

    def crop_ROI(self, cascade):
        flag=False
        #print("not workinggngngngngngngngngngngngngn")
        self.cropped_image = self.frame.copy()
        gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
        rectangles = cascade.detectMultiScale(gray)
        
        for (x,y,w,h) in rectangles:
            print("found a manooooo")
            #clone = cv2.rectangle(self.cropped_image,(x,y),(x+w,y+h),(255,0,0),2)
            self.image_coordinates = [(x,y)]
            self.image_coordinates.append((x+w,y+h))
            #self.image_coordinates[0][0] = x
            #self.image_coordinates[0][1] = y
            #self.image_coordinates[1][0] = x+w
            #self.image_coordinates[1][1] = y+h
            self.cropped_image = self.cropped_image[y:y+h, x:x+w]
            flag=True

            # display the frame with segmented hand
        #cv2.imshow("Video Feed", self.cropped_image)
        return flag





    def show_cropped_ROI(self):
        cv2.imshow('cropped image', self.cropped_image)

    def setup_ROI_tracking(self):
        roi = self.frame[self.image_coordinates[0][1]:self.image_coordinates[1][1], self.image_coordinates[0][0]:self.image_coordinates[1][0]]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
        cv2.imshow('cropped image', roi)





    def run_avg(image, aWeight):
        global bg
        # initialize the background
        if bg is None:
            bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, bg, aWeight)

    def segment(image, threshold=25):
        global bg
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)




    def get_contour(image):
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
#term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

if __name__ == '__main__':
    static_ROI = staticROI()