
from PrepareDataForFiltering import ExistingImage
from PrepareDataForFiltering import SimpleImage
from PrepareDataForFiltering import Tire

from PIL import Image
from filterpy import kalman
from filterpy.common import pretty_str
from scipy.stats import cauchy
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import cv2
import numpy as np

#objects for creating images
si = SimpleImage()
tireObj = Tire()
ei = ExistingImage()

#create existing image from path
#image = ei.filteredCrossImage()

#create simple images
#image = si.twoColorSimpleImage()
#image = si.simpleImage()

#create tire image

#set width and height of image block
imgH = 2500
imgW = 2500
tireObj.setBlockSize(imgW,imgH)
image = tireObj.nextTireBlock()

image = image[920:1400, 840:1280] #inside part of letter E
#image = image[920:1020, 840:940] #piece of letter E

#save original image
cv2.imwrite(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\NewLetterEOriginal.tif", image)

#print some info about image
print("image shape: ", image.shape)
print("max: ", np.max(image))
print("min: ", np.min(image))

#set image to positive values
imageMin = np.abs(np.min(image))
image += imageMin

print("after adding min. Max: ", np.max(image))
print("after adding min. Min: ", np.min(image))


#show image
# normImg = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imshow("noisedImage", normImg)
# cv2.waitKey(0)

#set algorithm constants with values from article
alpha = 1.0
beta = 0.0
kappa = 0.0
L = 100
noiseMean = 1.0

#set algorithm constants with values from experiments
noiseVar = 0.02


# This is class for creating sigma points based on article "Edge-Preserving Unscented Kalman Filter for Speckle Reduction".
# This class is based on Julier sigma points class from filterpyLibrary.
class EdgePreservingSigmaPoints(object):

    def __init__(self, n, kappa=0., sqrt_method=None, subtract=None):

        self.n = n
        self.kappa = kappa
        self.lambdaConst = np.pow(alpha,2)*(n+self.kappa)-n
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()
        self.filteredNeigbours = np.zeros(4)

    #set neighboring pixels of processed pixel. They are set from filtered image from NHSP neighborhood.
    def setNeigbours(self, neigbours):
        self.filteredNeigbours = neigbours

    #this pdf is from article. It should help to preserve edges.
    def edgePreservingPDF(self, sample):
        sum = np.sum(np.power(sample-self.filteredNeigbours,2))

        zeta = 0.01 #const with value from article
        neiMean = np.mean(self.filteredNeigbours)
        sumCoef = 1.0/(2.0*zeta*np.power(neiMean, 1.5))

        ni = sumCoef*sum

        gamma = 20.9#const from article with value set by experiments.
        xForLog = 1.0+(ni/gamma)
        xForExp = -gamma*np.log(xForLog)
        qx = np.exp(xForExp)
        return qx

    #create mean and variance used in sigma points
    def edgePreservingDistributionMoments(self):
        #get mean and std of filtered neigbors
        filteredSamplesMean = np.mean(self.filteredNeigbours)
        filteredSamplesStd = np.std(self.filteredNeigbours)

        #generate samples from Cauchy distribution
        #cauchySamples = cauchy.rvs(loc=filteredSamplesMean, scale=np.abs(filteredSamplesStd), size=L)
        start = filteredSamplesMean-90*filteredSamplesStd
        end = filteredSamplesMean+90*filteredSamplesStd
        cauchySamples = np.linspace(start,end,100)

        #evaluate samples by Cauchy pdf
        q = cauchy.pdf(cauchySamples, loc=filteredSamplesMean, scale=np.abs(filteredSamplesStd))

        #evaluate samples by "article" pdf
        p = []
        for s in cauchySamples:
            p.append(self.edgePreservingPDF(s))

        #compute weights used in mean and variance calculation
        weights = p/q

        #mean and variance calculation
        meanEp = np.sum(weights*cauchySamples)/np.sum(weights)
        varianceEp = np.sum(weights*np.pow(cauchySamples-meanEp,2))/np.sum(weights)

        return meanEp, varianceEp


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self):
        #get statistics for generating sigma points
        epMean, epVar = self.edgePreservingDistributionMoments()

        #augment predicted statistics with noise statistics
        x = np.array([epMean, noiseMean]) #set state
        P = np.zeros((2,2)) #set covariance matrix
        P[0,0] = epVar
        P[1,1] = noiseVar

        # set length of state
        n = np.size(x)

        # set sigma points using formulas from article
        sigmas = np.empty((2*n+1, n))
        U = np.empty(P.shape)
        try:
            U = self.sqrt((n + self.lambdaConst) * P)
        except:
            print("P v catch: ", P)
            print("SP: neigbours: ", self.filteredNeigbours)

        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[:,k])
            sigmas[n+k+1] = self.subtract(x, U[:,k])

        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the unscented Kalman filter. In this
        formulation the weights for the mean and covariance are the same.
        """

        n = self.n
        l = self.lambdaConst

        self.Wm = np.full(2*n+1, .5 / (n + l))
        self.Wm[0] = l / (n+l)
        self.Wc = self.Wm
        self.Wc[0] = self.Wm[0]+(1-np.pow(alpha,2)+beta)


    def __repr__(self):

        return '\n'.join([
            'JulierSigmaPoints object',
            pretty_str('n', self.n),
            pretty_str('kappa', self.kappa),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('subtract', self.subtract),
            pretty_str('sqrt', self.sqrt)
            ])

n = 2 #size of state x
m = 1 #size of measurement

#nonlineare function f(x) which predicts new state from x (for images is used to return just x. I see it in more places)
def f(x, dt):
     return x

#nonlineare function h(x) which predicts measurement from state (I hope so).  (for images is used to return just x. I see it in more places)
def h(x):
    return x

#create sigma points and modified Unscented Kalman filter
points = EdgePreservingSigmaPoints(n, kappa=kappa, sqrt_method=None, subtract=None)
ukf = kalman.UnscentedKalmanFilter(n, m, 1, h, f, points)

#set filtered image to image
filtered_image = image
counter = 0 #couter of predicted rows from image (I used it as some kind of progress bar)

#loops for predict more image blocks than one
# for e in range(1, image.shape[0]-100, 100):
#     for f in range(1, image.shape[1]-100, 100):
#         print("blockCoors...........................................: ", e , " ", f)
#         block = filtered_image[e:e+100, f:f+100]

for i in range(1, image.shape[0]-1):
    print(counter, ". line")
    counter += 1
    for j in range(1, image.shape[1]-1):

        #get measuremnet
        z = image[i, j]

        #get pixel's neighbors from filtered image
        neighsbours = np.array([filtered_image[i-1,j-1], filtered_image[i-1,j], filtered_image[i-1,j+1], filtered_image[i,j-1]])
        points.setNeigbours(neighsbours)

        #predict and update step
        ukf.predict()
        ukf.update(z)

        #set filtered pixel to image
        filtered_image[i, j] = ukf.x[0]


#return image to original values
filtered_image -= imageMin

#print some statistics about filtered image
print("filtered_image max: ", np.max(filtered_image))
print("filtered_image min: ", np.min(filtered_image))

#save filtered image
cv2.imwrite(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\NewLetterEFilteredCauchy90Gamma50.tif", filtered_image)


