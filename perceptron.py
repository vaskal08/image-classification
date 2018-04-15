#digit images are 28x28 pixels
from random import randint
class PerceptronNetwork(object):
    def __init__(self, f, y):
        self.features = [0.0] * f
        self.outputs = [0.0] * y
        self.weights = [[randint(0, 9)]*f]*y

        self.images = []
        self.labels = []

        #for i in range(0, len(self.weights)):
        #    print "features of output {}".format(i)
        #    print self.weights[i]
    
    def loadTrainingData(self, imageWidth, imageHeight, imagesPath, labelsPath):
        trainingImages = open(imagesPath, "r")

        count = 0
        currImage = []
        for line in trainingImages:
            for c in line:
                if c == '+':
                    currImage.append(0.5)
                elif c == '#':
                    currImage.append(1.0)
                else:
                    currImage.append(0)
            count = count + 1
            if count == 28:
                self.images.append(currImage)
                currImage = []
                count = 0

        trainingImages.close()

        traininglabels = open(labelsPath, "r")

        for line in traininglabels:
            label = int(line)
            self.labels.append(label)

        traininglabels.close()

    def score(self, input, output):
        w = self.weights[output]
        y = 0.0
        for i in range(0, len(w)):
            y = y + (w[i]*input[i])
        return y
    def train(self, percentage=1.0):
        print "training {} of data".format(percentage)

# ----- DIGITS ----- #

width = 28
height = 28
y = list(range(0, 10))

# paths
trainingImagesPath = "data/digitdata/trainingimages"
trainingLabelsPath = "data/digitdata/traininglabels"

testImagesPath = "data/digitdata/testimages"
testLabelsPath = "data/digitdata/testlabels"

validationImagesPath = "data/digitdata/validationimages"
validationLabelsPath = "data/digitdata/validationlabels"

# perceptron classification
percep = PerceptronNetwork(width*height, len(y))
percep.loadTrainingData(width, height, trainingImagesPath, trainingLabelsPath)
percep.train(0.1)
#print percep.score([0.5, 1.0], 2)