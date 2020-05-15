import numpy as np
import cv2

def makeContour(im, imgName): # creates the traces for the contours
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    exContours, exHierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    exHierarchy = exHierarchy[0]
    deletedIndices = []
    print(np.shape(contours))
    print(np.shape(exContours))
    print(contours)
    print(np.shape(hierarchy))
    print(np.shape(exHierarchy))
    for each in range(len(hierarchy)):
        for external in exHierarchy:
            allSame = True
            for value in range(len(hierarchy[each])):
                if hierarchy[each][value] != external[value]:
                    allSame = False
            if allSame:
                deletedIndices.append(each)
    importantContours = np.delete(contours, deletedIndices)
    importantHierarchies = np.delete(hierarchy, deletedIndices)
    img = cv2.drawContours(im, contours, -1, (255, 255, 255), 3)
    cv2.imshow(imgName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.drawContours(im, importantContours, -1, (255, 255, 255), 3)
    cv2.imshow(imgName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def subtractBackground(img):
    x, y, z = np.shape(img)
    for l in range(x):
        for w in range(y):
            if sum(img[l][w])/3 >= 250:
                for color in range(z):
                    img[l][w][color] = 0
    return img

def subtractBody(img):
    x, y, z = np.shape(img)
    for l in range(x):
        for w in range(y):
            avg = sum(img[l][w])/3
            if img[l][w][2] < avg - 3:
                for color in range(z):
                    img[l][w][color] = 0
    return img
testImage = "newtest.jpg"
testImage2 = "newtest2.jpg"
image = cv2.imread(testImage)
image2 = cv2.imread(testImage2)
image = subtractBackground(image)
image2 = subtractBackground(image2)
image = subtractBody(image)
image2 = subtractBody(image2)
makeContour(image, testImage)
makeContour(image2, testImage2)
