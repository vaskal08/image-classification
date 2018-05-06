#digit images are 28x28 pixels
import operator
import imageload
class PerceptronNetwork(object):
    def __init__(self, f, y):
        self.features = [0.0] * f
        self.outputs = y
        self.weights = [[0]*f]*len(y)

    def score(self, input, output):
        w = self.weights[output]
        y = 0.0
        for i in range(0, len(w)):
            y = y + (w[i]*input[i])
        return y

    def train(self, imageWidth, imageHeight, imagesPath, labelsPath, percentage=1.0):
        #load images
        imagesAndLabels = imageload.loadImages(imageWidth, imageHeight, imagesPath, labelsPath, True)
        images = imagesAndLabels[0]
        labels = imagesAndLabels[1]

        r = list(range(0, int(percentage*len(images))))

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
        imagesAndLabels = imageload.loadImages(imageWidth, imageHeight, imagesPath, labelsPath, False)
        images = imagesAndLabels[0]
        labels = imagesAndLabels[1]

        successes = 0
        tests = len(images)
        
        r = list(range(0, len(images)))

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
        #print "{} successes".format(successes)
        #print "{} tests".format(tests)
        #print "correct {} percent of the time".format(percentageCorrect)
        return percentageCorrect
