import random

def imageToValues(image, imageWidth, imageHeight):
    currImage = []
    count = 0
    for line in image.splitlines():
        for i in range(0, imageWidth):
            c = line[i]
            if c == '+':
                currImage.append(1)
            elif c == '#':
                currImage.append(1)
            elif c == ' ':
                currImage.append(0)
        count = count + 1
        if count == imageHeight:
            break
    return currImage

def loadImages(imageWidth, imageHeight, imagesPath, labelsPath, rand):
    imageFile = open(imagesPath, "r")

    count = 0
    currImage = []
    images = []
    labels = []

    for line in imageFile:
        for i in range(0, imageWidth):
            c = line[i]
            if c == '+':
                currImage.append(1)
            elif c == '#':
                currImage.append(1)
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

    if rand == True:
        c = list(zip(images, labels))
        random.shuffle(c)
        images, labels = zip(*c)

    return (images, labels)