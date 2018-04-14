from enum import Enum
class ClassificationAlgorithm(Enum):
    PERCEPTRON = 1
    NAIVE_BAYES = 2
    CUSTOM = 3

class ImageClassifier(object):
    def __init__(self, algorithm):
        self.dataPath = "data"
        self.algorithm = algorithm
        print self.algorithm

    def load(self, type):
        print "loading data :{}/{}data".format(self.dataPath, type)
        file = open("{}/{}data/traininglabels".format(self.dataPath, type), "r")

    def train(self, percentage):
        print "training {}% of data".format(percentage)

imageClassifier = ImageClassifier(ClassificationAlgorithm.PERCEPTRON)
imageClassifier.load("digit")
imageClassifier.train(20)