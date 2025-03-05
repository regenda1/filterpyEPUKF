from PIL import Image
from PIL import ImageFile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal as ss

class ExistingImage:

    def filteredCrossImage(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open('C:\\Users\\24000079\\Documents\\Kika\\Obrazky\\filtering\\crossImageCauchyVar10Gamma1dot0.tif')
        return np.array(img)
class Tire:
    def __init__(self):
        img = Image.open('C:\\Users\\24000079\Documents\Kika\Obrazky\\filtering\\image.tif')
        self.tireImage = np.array(img)
        self.startH = 0
        self.startW = 0

    def tireShape(self):
        return self.tireImage.shape
    def setBlockSize(self, w, h): #h 2500 w 512 ide to sem naopak
        self.width = w
        self.height = h

    def blockSize(self):
        return self.width, self.height

    def nextTireBlock(self):
        block = self.tireImage[self.startH : self.startH+self.height, self.startW : self.startW+self.width]
        #self.startH += self.height
        self.startW += self.width
        return block

class SimpleImage:

    def simpleImage(self):
        img = np.full((250, 100), 2.0, dtype=np.float32)

        for r in range(img.shape[0]):
            img[r] += r

        img[80:83, 5:200] = 0
        img[120:123, 20:80] = 0
        img[50:153, 50:53] = 0

        noise = np.random.normal(1, 0.1, img.shape)
        noisedImg = img * noise

        noisedImg /= np.max(noisedImg)+1.0
        noisedImg += 0.001

        return noisedImg

    def twoColorSimpleImage(self):
        img = np.full((250, 100), 0.7, dtype=np.float32)
        img[:, 0:50] = 0.2

        noise = np.random.normal(1, 0.1, img.shape)
        noisedImg = img * noise

        return noisedImg

    def makeImageStatistics(self):

        images = []
        images.append(self.simpleImage())
        images.append(self.twoColorSimpleImage())

        imageNames = ["simpleImage", "twoColorImage"]

        for i, n in zip(images, imageNames):
            min = np.min(i)
            max = np.max(i)
            print("min and max value: ", min, " ", max)

            normImg = cv2.normalize(i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow(n, normImg)
            cv2.waitKey(0)

#so = SimpleImage()
#so.makeImageStatistics()
#img = so.simpleImage()
#cv2.imwrite(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\noisedCrossImage.tif", img)