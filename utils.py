import cv2
import numpy as np
import imutils
import random
import math

def save_images(images, folder='', prefix='image_'):
    if folder != '' :
       count = 0
       for i in images:
           cv2.imwrite(folder+prefix+str(count)+'.jpg', i)
           count += 1

def random_images(images, size=0.3):
    maxit = math.ceil(images.shape[0] * size)
    results = []
    while maxit > 0 :
       index = random.randint(0, images.shape[0]-1)
       results.append(images[index])
       maxit -= 1

    return np.array(results)

def rotate_images(images, angle=90., size=0.3):
    results = random_images(images, size)
    for i in range(results.shape[0]):
        results[i] = imutils.rotate(results[i], angle)

    return results

  


