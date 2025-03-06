from PrepareDataForFiltering import Tire

import cv2
import numpy as np
from scipy.stats import cauchy
from scipy.linalg import cholesky

tireObj = Tire()

#set width and height of image block
imgH = 2500
imgW = 2500
tireObj.setBlockSize(imgW,imgH)
image = tireObj.nextTireBlock()

#image = image[920:1400, 840:1280] #inside part of letter E
image = image[1050:1300, 150:410] #stripes
#image = image[920:1020, 840:940] #piece of letter E

#save original image
cv2.imwrite(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\StripesOriginal.tif", image)

#print some info about image
print("image shape: ", image.shape)
print("max: ", np.max(image))
print("min: ", np.min(image))

#set image to positive values
imageMin = np.abs(np.min(image))
image += imageMin

print("after adding min. Max: ", np.max(image))
print("after adding min. Min: ", np.min(image))

class EdgePreservingUnscentedKalmanFilter:

    def __init__(self):

        self.filteredNeighbors = np.empty(4) #takes NHSP neigbors

        #init const
        self.stdSize = 90 #value from experiments

        #init consts from article
        self.n = 2

        self.zeta = 0.01 #value from article
        self.gamma = 2.9 #value from experiments
        self.L = 100#number of samples for pdfs. Value from article

        alpha = 1.0
        kappa = 0.0
        self.lambdaConst = np.pow(alpha,2)*(self.n+kappa)-self.n

        beta = 0.0
        self.Wm = self.Wc = np.full(2*self.n+1, 1.0/(2.0*(self.n+self.lambdaConst)))
        self.Wm[0] = self.lambdaConst/(self.n+self.lambdaConst)
        self.Wc[0] = self.Wm[0] + (1-np.pow(alpha,2)+beta)

        self.noiseMean = 1.0 #value from article
        self.noiseVar = 0.02 #value from experiments

    def setNeigbours(self, neigbors):
        self.filteredNeigbors = neigbors
    def edgePreservingPDF(self, sample):
        sum = np.sum(np.power(sample-self.filteredNeigbors,2))

        mi = np.mean(self.filteredNeigbors)
        sumCoef = 1.0/(2.0*self.zeta*np.power(mi, 1.5))

        niSquared = sumCoef*sum

        xForLog = 1.0+(niSquared/self.gamma)
        xForExp = -self.gamma*np.log(xForLog)
        qx = np.exp(xForExp)
        return qx

    def edgePreservingDistributionMoments(self):
        #get mean and std of filtered neigbors
        filteredNeigborsMean = np.mean(self.filteredNeigbors)
        filteredNeigborsStd = np.std(self.filteredNeigbors)

        #generate samples from Cauchy distribution
        cauchySamples = cauchy.rvs(loc=filteredNeigborsMean, scale=filteredNeigborsStd, size=self.L)
        # start = filteredNeigborsMean-self.stdSize*filteredNeigborsStd
        # end = filteredNeigborsMean+self.stdSize*filteredNeigborsStd
        # cauchySamples = np.linspace(start,end,self.L)

        #evaluate samples by Cauchy pdf
        q = cauchy.pdf(cauchySamples, loc=filteredNeigborsMean, scale=filteredNeigborsStd)

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

    def sigma_points(self):
        #get statistics for generating sigma points
        epMean, epVar = self.edgePreservingDistributionMoments()

        #augment predicted statistics with noise statistics
        x = np.array([epMean, self.noiseMean]) #set state
        P = np.zeros((2,2)) #set covariance matrix
        P[0,0] = epVar
        P[1,1] = self.noiseVar

        # set sigma points using formulas from article
        U = np.empty(P.shape)
        try:
            U = cholesky((self.n + self.lambdaConst) * P)
        except:
            print("P v catch: ", P)
            print("SP: neigbours: ", self.filteredNeigbors)

        sigmas = np.empty((self.n, 2 * self.n + 1))
        sigmas[:, 0] = x
        for k in range(self.n):
            # pylint: disable=bad-whitespace
            sigmas[:, k+1]   = x+U[:,k]
            sigmas[:, self.n+k+1] = x-U[:,k]

        return sigmas

    def predictAndUpdate(self, z):
        sigmas = self.sigma_points()
        Xx = sigmas[0]
        Xv = sigmas[1]
        Y = np.multiply(Xx,Xv)

        meanY = np.sum(self.Wm * Y)
        meanX = np.sum(self.Wm * Xx)

        Py = np.sum(np.multiply(Y - meanY, Y - meanY) * self.Wc)
        Pxy = np.sum(np.multiply(Xx - meanX, Y - meanY) * self.Wc)

        K = Pxy * (1.0 / Py)
        # self.y = self.residual_z(z, zp)   # residual

        # update state estimate (x, P)
        prevX = sigmas[:, 0]
        x = prevX + K * (z - meanY)
        return x


epukf = EdgePreservingUnscentedKalmanFilter()

#set filtered image to image
filtered_image = image
counter = 0 #couter of predicted rows from image (I used it as some kind of progress bar)

for i in range(1, image.shape[0]-1):
    print(counter, ". line")
    counter += 1
    for j in range(1, image.shape[1]-1):

        #get measuremnet
        z = image[i, j]

        #get pixel's neighbors from filtered image
        neighsbours = np.array([filtered_image[i-1,j-1], filtered_image[i-1,j], filtered_image[i-1,j+1], filtered_image[i,j-1]])
        epukf.setNeigbours(neighsbours)

        #set filtered pixel to image
        filtered_image[i, j] = epukf.predictAndUpdate(z)[0]


#return image to original values
filtered_image -= imageMin

#print some statistics about filtered image
print("filtered_image max: ", np.max(filtered_image))
print("filtered_image min: ", np.min(filtered_image))

#save filtered image
cv2.imwrite(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\StripesFitlered.tif", filtered_image)
