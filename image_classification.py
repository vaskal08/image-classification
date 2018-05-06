from perceptron import PerceptronNetwork
from bayes import NaiveBayes
import time

# statistics code from https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list
def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5

percents = list(range(1, 11))

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

print "---------- TEST ----------"

# perceptron classification
print "---------- Perceptron ----------"
digitPercepAvgs = []
digitPercepStds = []
digitPercepTimes = []
for percent in percents:
    p = percent/10.0
    x = 5
    res = []
    times = []
    for i in range(0, 5):
        digitPercep = PerceptronNetwork(digitWidth*digitHeight, digitY)
        t1 = time.time()
        digitPercep.train(digitWidth, digitHeight, digitTrainingImagesPath, digitTrainingLabelsPath, p)
        dt = time.time() - t1
        percentageCorrect = digitPercep.test(digitWidth, digitHeight, digitTestImagesPath, digitTestLabelsPath)
        res.append(percentageCorrect)
        times.append(dt)
    avgTime = mean(times)
    avgAcc = mean(res)
    stdAcc = stddev(res)
    digitPercepTimes.append(avgTime)
    digitPercepStds.append(stdAcc)
    digitPercepAvgs.append(avgAcc)
print "times: {}".format(digitPercepTimes)
print "means: {}".format(digitPercepAvgs)
print "stds: {}".format(digitPercepStds)

# naive bayes classification
#print "---------- Naive Bayes ----------"
digitBayesAvgs = []
digitBayesStds = []
digitBayesTimes = []
for percent in percents:
    p = percent/10.0
    x = 5
    res = []
    times = []
    for i in range(0, 5):
        digitBayes = NaiveBayes(digitWidth*digitHeight, 10, 2)
        t1 = time.time()
        digitBayes.train(digitWidth, digitHeight, digitTrainingImagesPath, digitTrainingLabelsPath, p)
        dt = time.time() - t1
        percentageCorrect = digitBayes.test(digitWidth, digitHeight, digitTestImagesPath, digitTestLabelsPath)
        res.append(percentageCorrect)
        times.append(dt)
    avgTime = mean(times)
    avgAcc = mean(res)
    stdAcc = stddev(res)
    digitBayesAvgs.append(avgAcc)
    digitBayesStds.append(stdAcc)
    digitBayesTimes.append(avgTime)
print "times: {}".format(digitBayesTimes)
print "means: {}".format(digitBayesAvgs)
print "stds: {}".format(digitBayesStds)

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

print "---------- TEST ----------"

# perceptron classification
print "---------- Perceptron ----------"
facePercepAvgs = []
facePercepStds = []
facePercepTimes = []
for percent in percents:
    p = percent/10.0
    x = 5
    res = []
    times = []
    for i in range(0, 5):
        facePercep = PerceptronNetwork(faceWidth*faceHeight, faceY)
        t1 = time.time()
        facePercep.train(faceWidth, faceHeight, faceTrainingImagesPath, faceTrainingLabelsPath, p)
        dt = time.time() - t1
        percentageCorrect = facePercep.test(faceWidth, faceHeight, faceTestImagesPath, faceTestLabelsPath)
        res.append(percentageCorrect)
        times.append(dt)
    avgTime = mean(times)
    avgAcc = mean(res)
    stdAcc = stddev(res)
    facePercepTimes.append(avgTime)
    facePercepStds.append(stdAcc)
    facePercepAvgs.append(avgAcc)
print "times: {}".format(facePercepTimes)
print "means: {}".format(facePercepAvgs)
print "stds: {}".format(facePercepStds)


# naive bayes classification
print "---------- Naive Bayes ----------"
faceBayesAvgs = []
faceBayesStds = []
faceBayesTimes = []
for percent in percents:
    p = percent/10.0
    x = 5
    res = []
    times = []
    for i in range(0, 5):
        faceBayes = NaiveBayes(faceWidth*faceHeight, 2, 2)
        t1 = time.time()
        faceBayes.train(faceWidth, faceHeight, faceTrainingImagesPath, faceTrainingLabelsPath, p)
        dt = time.time() - t1
        percentageCorrect = faceBayes.test(faceWidth, faceHeight, faceTestImagesPath, faceTestLabelsPath)
        res.append(percentageCorrect)
        times.append(dt)
    avgTime = mean(times)
    avgAcc = mean(res)
    stdAcc = stddev(res)
    faceBayesAvgs.append(avgAcc)
    faceBayesStds.append(stdAcc)
    faceBayesTimes.append(avgTime)
print "times: {}".format(faceBayesTimes)
print "means: {}".format(faceBayesAvgs)
print "stds: {}".format(faceBayesStds)