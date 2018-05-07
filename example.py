from perceptron import PerceptronNetwork
from bayes import NaiveBayes

# The purpose of this class is to show an example of training the perceptron and naive bayes classifiers with 100% of the digit training data and testing one digit. For use during demos.

# paths
digitTrainingImagesPath = "data/digitdata/trainingimages"
digitTrainingLabelsPath = "data/digitdata/traininglabels"

faceTrainingImagesPath = "data/facedata/facedatatrain"
faceTrainingLabelsPath = "data/facedata/facedatatrainlabels"

digitWidth = 28
digitHeight = 28
digitY = list(range(0, 10))

digit = """                            
                            
                            
                            
                            
              +#++          
             +####+         
           ++######         
           +###+++#         
          +###+  +#+        
          +###   +##        
         +###+    ##+       
         +##+     +#+       
         ###      +#+       
        +##+      +##+      
        +##       +##+      
        +#+       +##+      
        +#+       +##       
        +#+       +#+       
        +##+     +##+       
         ###+    +##+       
         +####++###++       
         +#########         
          ++#######         
            +###+++         
                            
                            
                            
                            """

digitPercep = PerceptronNetwork(digitWidth*digitHeight, digitY)
digitPercep.train(digitWidth, digitHeight, digitTrainingImagesPath, digitTrainingLabelsPath)

print "Perceptron guess:"
print digitPercep.test_one(digitWidth, digitHeight, digit)

digitBayes = NaiveBayes(digitWidth*digitHeight, 10, 2)
digitBayes.train(digitWidth, digitHeight, digitTrainingImagesPath, digitTrainingLabelsPath)

print "Naive Bayes guess:"
print digitBayes.test_one(digitWidth, digitHeight, digit)