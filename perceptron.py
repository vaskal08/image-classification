#digit images are 28x28 pixels
import random
import operator
class PerceptronNetwork(object):
    def __init__(self, f, y):
        self.features = [0.0] * f
        self.outputs = y
        self.weights = [[0]*f]*len(y)
    
    def loadImages(self, imageWidth, imageHeight, imagesPath, labelsPath):
        imageFile = open(imagesPath, "r")

        count = 0
        currImage = []
        images = []
        labels = []

        factor = 2.0
        plus = 0.5
        pound = plus*factor

        for line in imageFile:
            for i in range(0, imageWidth):
                c = line[i]
                if c == '+':
                    currImage.append(plus)
                elif c == '#':
                    currImage.append(pound)
                elif c == ' ':
                    currImage.append(0)
            count = count + 1
            if count == imageHeight:
                images.append(currImage)
                currImage = []
                count = 0

        imageFile.close()

        labelsFile = open(labelsPath, "r")

        for line in labelsFile:
            label = int(line)
            labels.append(label)

        labelsFile.close()

        return (images, labels)

    def score(self, input, output):
        w = self.weights[output]
        y = 0.0
        for i in range(0, len(w)):
            y = y + (w[i]*input[i])
        return y

    def train(self, imageWidth, imageHeight, imagesPath, labelsPath, percentage=1.0):
        #load images
        imagesAndLabels = self.loadImages(imageWidth, imageHeight, imagesPath, labelsPath)
        images = imagesAndLabels[0]
        labels = imagesAndLabels[1]

        r = list(range(0, int(percentage*len(images))))
        #random.shuffle(r)

        for i in r:
            image = images[i]
            # function to map image to features
            scores = []
            for y in self.outputs:
                score = self.score(image, y)
                scores.append(score)
            yPrime = scores.index(max(scores))
            y = labels[i]
            if yPrime != y:
                self.weights[y] = map(operator.add, self.weights[y], image)
                self.weights[yPrime] = map(operator.sub, self.weights[yPrime], image)

        print "trained {} of data".format(percentage)

    def test(self, imageWidth, imageHeight, imagesPath, labelsPath):
        #load images
        imagesAndLabels = self.loadImages(imageWidth, imageHeight, imagesPath, labelsPath)
        images = imagesAndLabels[0]
        labels = imagesAndLabels[1]

        successes = 0
        tests = len(images)
        
        r = list(range(0, len(images)))
        #random.shuffle(r)

        for i in r:
            image = images[i]
            
            scores = []
            for y in self.outputs:
                score = self.score(image, y)
                scores.append(score)
            guess = scores.index(max(scores))
            y = labels[i]
            if guess == y:
                successes = successes + 1

        percentageCorrect = ((successes*1.0)/(tests*1.0))*100
        print "{} successes".format(successes)
        print "{} tests".format(tests)
        print "correct {} percent of the time".format(percentageCorrect)

# ----- DIGITS ----- #
print "########## DIGITS ##########"
digitWidth = 28
digitHeight = 28
digitY = list(range(0, 10))

# paths
digitTrainingImagesPath = "data/digitdata/trainingimages"
digitTrainingLabelsPath = "data/digitdata/traininglabels"

digitTestImagesPath = "data/digitdata/testimages"
digitTestLabelsPath = "data/digitdata/testlabels"

digitValidationImagesPath = "data/digitdata/validationimages"
digitValidationLabelsPath = "data/digitdata/validationlabels"

# perceptron classification
digitPercep = PerceptronNetwork(digitWidth*digitHeight, digitY)
digitPercep.train(digitWidth, digitHeight, digitTrainingImagesPath, digitTrainingLabelsPath)
print "---------- test ----------"
#test images
digitPercep.test(digitWidth, digitHeight, digitTestImagesPath, digitTestLabelsPath)
#validation images
print "---------- validation ----------"
digitPercep.test(digitWidth, digitHeight, digitValidationImagesPath, digitValidationLabelsPath)


# ----- FACES ----- #
print "########## FACES ##########"
faceWidth = 60
faceHeight = 70
faceY = [0, 1]

# paths
faceTrainingImagesPath = "data/facedata/facedatatrain"
faceTrainingLabelsPath = "data/facedata/facedatatrainlabels"

faceTestImagesPath = "data/facedata/facedatatest"
faceTestLabelsPath = "data/facedata/facedatatestlabels"

faceValidationImagesPath = "data/facedata/facedatavalidation"
faceValidationLabelsPath = "data/facedata/facedatavalidationlabels"

# perceptron classification
facePercep = PerceptronNetwork(faceWidth*faceHeight, faceY)
facePercep.train(faceWidth, faceHeight, faceTrainingImagesPath, faceTrainingLabelsPath)
print "---------- test ----------"
#test images
facePercep.test(faceWidth, faceHeight, faceTestImagesPath, faceTestLabelsPath)
#validation images
print "---------- validation ----------"
facePercep.test(faceWidth, faceHeight, faceValidationImagesPath, faceValidationLabelsPath)
