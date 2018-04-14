#digit images are 28x28 pixels
from random import randint
class PerceptronNetwork(object):
    def __init__(self, f, y):
        self.features = [0.0] * f
        self.outputs = [0.0] * y
        self.weights = [[randint(0, 9)]*f]*y

        print self.features
        print self.outputs
        print self.weights

        #for i in range(0, len(self.weights)):
        #    print "features of output {}".format(i)
        #    print self.weights[i]
    
    def loadTrainingData(self):
        print "loading training data"

    def score(self, input, output):
        w = self.weights[output]
        y = 0.0
        for i in range(0, len(w)):
            y = y + (w[i]*input[i])
        return y
    def train(self, percentage=1.0):
        print "training {} of data".format(percentage)

percep = PerceptronNetwork(2, 4)
percep.loadTrainingData()
percep.train(0.1)
print percep.score([0.5, 1.0], 2)