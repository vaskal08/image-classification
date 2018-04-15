from perceptron import PerceptronNetwork

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