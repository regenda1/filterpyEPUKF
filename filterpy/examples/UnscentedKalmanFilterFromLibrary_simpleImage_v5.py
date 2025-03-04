# prechadza obrazok pixel po pixeli
#sem idem skusit spravit implementaciu sigma bodov
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

si = SimpleImage()
tireObj = Tire()
#ei = ExistingImage()

#image = ei.filteredCrossImage()

#image = si.twoColorSimpleImage()
#image = image[0:100, :]
#image = si.simpleImage()


imgH = 2500
imgW = 2500
tireObj.setBlockSize(imgW,imgH)
image = tireObj.nextTireBlock()
#image = image[1100:1200, 770:870]
image = image[920:1400, 840:1280]
#image = image[920:1020, 840:940]

#image = np.array(Image.open(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\Lena.png").convert('L')).astype(np.float64)
#image = image[50:350, 50:350]


# normImg = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imshow("noisedImage", normImg)
# cv2.waitKey(0)

#image *= np.random.normal(1.0, 0.05, image.shape)

# normImg = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imshow("noisedImage", normImg)
# cv2.waitKey(0)
cv2.imwrite(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\LetterEOriginal.tif", image)

print("image shape: ", image.shape)
print("max: ", np.max(image))
print("min: ", np.min(image))

# zeroShape = image[image == 0.0].shape
# image[image == 0.0] = image[image == 0.0]-np.abs(np.random.normal(0,0.5,zeroShape))

normConst = np.abs(np.min(image))
image += normConst
# image += 0.01

print("+min max: ", np.max(image))
print("+min min: ", np.min(image))

# imageMax = np.max(image)+2
# image /= imageMax

print("norm max: ", np.max(image))
print("norm min: ", np.min(image))



#image = image[0:50, 25:75]

# normImg = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imshow("noisedImage", normImg)
# cv2.waitKey(0)

alpha = 1.0
beta = 0.0
kappa = 0.0
L = 100
noiseMean = 1.0
noiseVar = 0.02


print("image shape: ", image.shape)

class EdgePreservingSigmaPoints(object):

    def __init__(self, n, kappa=0., sqrt_method=None, subtract=None):

        self.n = n
        self.kappa = kappa
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

    def setNeigbours(self, neigbours):
        self.filteredNeigbours = neigbours
    def edgePreservingPDF(self, sample):
        #print("................................edgePreservingPDF..............................")
        sum = np.sum(np.power(sample-self.filteredNeigbours,2))
        #print("sum: ", sum)
        zeta = 0.01
        # print("filteredNeighrs: ", self.filteredNeigbours)
        #print("filteredNeighbours mean: ", np.mean(self.filteredNeigbours))
        # print("np.power(np.mean(self.filteredNeigbours), 1.5): ", np.power(np.mean(self.filteredNeigbours), 1.5))
        neiMean = np.mean(self.filteredNeigbours)
        # if neiMean < 0.0:
        #     neiMean = 0.001
        sumCoef = 1.0/(2.0*zeta*np.power(neiMean, 1.5))
        #print("sumCoef: ", sumCoef)
        ni = sumCoef*sum

        gamma = 20.9#kika tusim, ze cim je toto mensie, tym je obrazok tmavsi. Ked je 1, tak je to biele
        xForLog = 1.0+(ni/gamma)
        #print("xForLog: ", xForLog)
        # kika v clanku je log. Po spravnosti by log mal oznacovat logaritmus pri zaklade 10. Ale ked ho dam pri zaklade 10, tak nejde. Mozno maju na mysli logaritmus naturalis.
        xForExp = -gamma*np.log(xForLog)
        #print("xForExp: ", xForExp)
        qx = np.exp(xForExp)
        #print("................................edgePreservingPDF..............................")
        return qx

    def edgePreservingDistributionMoments(self):
        # print(".....................edgePreservingDistributionMoments zaciatok..........................")
        # print("filteredNeigbours: ", self.filteredNeigbours)
        filteredSamplesMean = np.mean(self.filteredNeigbours)
        filteredSamplesStd = np.std(self.filteredNeigbours)

        # print("edgePreservingDistributionMoments: filteredSamplesMean",filteredSamplesMean)
        # print("edgePreservingDistributionMoments: filteredSamplesStd", filteredSamplesStd)
        # cim je scale vacsi tym je obrazok tmavsi. Cim je mensi, tym je obrazok blesdi
        # kika tuto doriesit, ze aka ma byt velka ta scale, ked ma mat nejaky support p in q
        #print("filteredNeighbours: ", self.filteredNeigbours)
        #cauchySamples = cauchy.rvs(loc=filteredSamplesMean, scale=np.abs(filteredSamplesStd), size=L)
        start = filteredSamplesMean-90*filteredSamplesStd
        end = filteredSamplesMean+90*filteredSamplesStd
        cauchySamples = np.linspace(start,end,100)
        #od mean + 2xodchylka
        q = cauchy.pdf(cauchySamples, loc=filteredSamplesMean, scale=np.abs(filteredSamplesStd))





        p = []
        for s in cauchySamples:
            p.append(self.edgePreservingPDF(s))

        # print("p: ", p)
        # print("q: ", q)
        weights = p/q
        #print("weights: ", weights)

        #cauchySamples = cauchySamples*weights #kika tento nedokonaly unbias to zasumel este viac, ale mozno len nerozumiem vzorcom

        meanEp = np.sum(weights*cauchySamples)/np.sum(weights)
        # print("weights: ", weights)
        # print("cauchySamples: ", cauchySamples)
        #print("edgePreservingDistributionMoments: meanEp: ", meanEp)
        # print("sumWeights: ", np.sum(weights))
        varianceEp = np.sum(weights*np.pow(cauchySamples-meanEp,2))/np.sum(weights)
        #print("edgePreservingDistributionMoments: varianceEp: ", varianceEp)
        #print(".....................edgePreservingDistributionMoments koniec..........................")
        return meanEp, varianceEp


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self):
        #print("..............sigma points zaciatok..........................")
        epMean, epVar = self.edgePreservingDistributionMoments()  # kika este tieto statistiky obohatit o statistiky sumu
        # print("epMean: ", epMean)
        # print("epVar: ", epVar)
        x = np.array([epMean, noiseMean])

        n = np.size(x)  # dimension of problem

        P = np.zeros((2,2))
        #print("epVar: ", epVar)
        P[0,0] = epVar
        P[1,1] = noiseVar

        sigmas = np.zeros((2*n+1, n))
        #print("sigmas shape: ", sigmas.shape)


        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = np.full(P.shape, 0.001)
        try:
            U = self.sqrt((n + self.kappa) * P)
            # print("SP: P: ", P)
            # print("SP: U: ", U)
        except:
            print("P v catch: ", P)
            print("SP: neigbours: ", self.filteredNeigbours)

        sigmas[0] = x
        #print("SP: x: ", x)
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        #print("SP: sigmas: ", sigmas)
        #print("..............sigma points koniec..........................")
        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the unscented Kalman filter. In this
        formulation the weights for the mean and covariance are the same.
        """

        n = self.n
        k = self.kappa

        self.Wm = np.full(2*n+1, .5 / (n + k))
        self.Wm[0] = k / (n+k)
        self.Wc = self.Wm


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

n = 2 #state size #49*2 = 98
m = 1 #sample size #7*7 = 49
def f(x, dt):
     return x

def h(x):
    return x

points = EdgePreservingSigmaPoints(n, kappa=kappa, sqrt_method=None, subtract=None)
ukf = kalman.UnscentedKalmanFilter(n, m, 1, h, f, points)

filtered_image = image
# filtered_image[filtered_image <= 0.0] = 0.97
# filtered_image[filtered_image >= 1.0] = 0.97
counter = 0
colCounter = 0
# for e in range(1, image.shape[0]-100, 100):
#     for f in range(1, image.shape[1]-100, 100):
#         print("blockCoors...........................................: ", e , " ", f)
#         block = filtered_image[e:e+100, f:f+100]

for i in range(1, image.shape[0]-1):
    print(counter, ". line")
    for j in range(1, image.shape[1]-1):

        z = image[i, j]

        neighsbours = np.array([filtered_image[i-1,j-1], filtered_image[i-1,j], filtered_image[i-1,j+1], filtered_image[i,j-1]])
        points.setNeigbours(neighsbours)
        ukf.predict()
        ukf.update(z)

        #predicted_block = ukf.x.reshape(7,7)
        #print("filteredImage row: ", filtered_image[1, 0:5])
        filtered_image[i, j] = ukf.x[0]
        # if ukf.x[0] < 0.0:
        #     filtered_image[i, j] = 0.0
        # elif ukf.x[0] > 113.0:
        #     filtered_image[i, j] = 112.0

        #print("filteredImage pix: ", filtered_image[i,j])
        # print("i,j: ", i, " ", j)
        # print("************************koniec for*********************")
        # print("************************koniec for*********************")
        # print("************************koniec for*********************")
    counter += 1
    colCounter = 0


print("filtered_image max: ", np.max(filtered_image))
print("filtered_image min: ", np.min(filtered_image))

cv2.imwrite(r"C:\Users\24000079\Documents\Kika\Obrazky\filtering\edgePresUkf\LetterEFilteredCauchy90Gamma50.tif", filtered_image-normConst)


