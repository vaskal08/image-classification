#digit images are 28x28 pixels
import operator
import math
import imageload

class NaiveBayes(object):
    def __init__(self, numberOfPixels, numberOfLabels, numberOfFeatureValue):
        self.numberOfLabels = numberOfLabels
        self.numberOfPixels = numberOfPixels
        self.numberOfFeatureValue = numberOfFeatureValue
        self.conditionalProb = [[[0 for k in xrange(numberOfFeatureValue)] for j in xrange(numberOfLabels)] for i in xrange(numberOfPixels)]

    def setPrior(self, labels):
        priorTmp = [0] * self.numberOfLabels
        total = 0
        for label in labels:
            priorTmp[label] = priorTmp[label] + 1
            total = total + 1
        self.prior = []
        for value in priorTmp:
            prob = value * 1.0 / total
            self.prior.append(prob)
        return

    def train(self, imageWidth, imageHeight, imagesPath, labelsPath, percentage=1.0):
        #load images
        imagesAndLabels = imageload.loadImages(imageWidth, imageHeight, imagesPath, labelsPath, True)
        images = imagesAndLabels[0]
        labels = imagesAndLabels[1]
        numberOfImages = int(percentage*len(images))
        self.estimateParameters(images, labels, numberOfImages)
        print "trained {} of data".format(percentage)

    def estimateParameters(self, images, labels, numberOfImages):
        self.setPrior(labels)
        
        total = [[0 for j in xrange(self.numberOfLabels)] for i in xrange(self.numberOfPixels)]
        
        for i in range(0, numberOfImages): #TODO
            image = images[i]
            curLabel = labels[i]
            for j in range(0, len(image)):
                featureVal = image[j]
                self.conditionalProb[j][curLabel][featureVal] = self.conditionalProb[j][curLabel][featureVal] + 1
                total[j][curLabel] = total[j][curLabel] + 1
                
        for i in range(0, self.numberOfPixels):
            for j in range(0, self.numberOfLabels):
                for k in range(0, self.numberOfFeatureValue):
                    self.conditionalProb[i][j][k] = self.conditionalProb[i][j][k] * 1.0 / total[i][j]
                
    def calculateLogJointProbabilities(self, image):
        res = [0.0] * self.numberOfLabels
        
        for i in range(0, self.numberOfLabels):
            res[i] = math.log(self.prior[i])
            for j in range(0, self.numberOfPixels):
                featureVal = image[j]
                if self.conditionalProb[j][i][featureVal] == 0 or res[i] == float('-inf'):
                    res[i] = float('-inf')
                else:
                    res[i] = res[i] + math.log(self.conditionalProb[j][i][featureVal])
        return res
    def test_one(self, imageWidth, imageHeight, image):
        imageValues = imageload.imageToValues(image, imageWidth, imageHeight)
        posterior = self.calculateLogJointProbabilities(imageValues)
        guess = posterior.index(max(posterior))
        
        return guess
    def test(self, imageWidth, imageHeight, imagesPath, labelsPath):
        #load images
        imagesAndLabels = imageload.loadImages(imageWidth, imageHeight, imagesPath, labelsPath, False)
        images = imagesAndLabels[0]
        labels = imagesAndLabels[1]

        successes = 0
        tests = len(images)
        
        r = list(range(0, len(images)))
        
        for i in r:
            image = images[i]
            posterior = self.calculateLogJointProbabilities(image)
            guess = posterior.index(max(posterior))
            y = labels[i]
            if guess == y:
                successes = successes + 1

        percentageCorrect = ((successes*1.0)/(tests*1.0))*100
        #print "{} successes".format(successes)
        #print "{} tests".format(tests)
        #print "correct {} percent of the time".format(percentageCorrect)
        return percentageCorrect
