import numpy as np
import cv2
import imutils

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

def matchOpenEye(img, imgName):
    pass

def matchClosedEye(img, imgName):
    pass
    template = cv2.imread('closed_eye.jpg',0)

def findPupil(img, ex, ey, ew, eh, imgName):
    cropped_img = img[ex: ex + ew, ey: ey + eh]
    imgray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # Apply Hough transform on the blurred image.
    circles = cv2.HoughCircles(imgray,
                                        cv2.HOUGH_GRADIENT, 1, 25, param1=50,
                                        param2=20, minRadius=eh//10, maxRadius=40)
    if circles is not None:
        # Convert the circle parameters a, b and r to integers.
        circles = np.round(circles[0, :]).astype("int")
        for values in circles:
            a, b, r = values[0], values[1], values[2]
            cv2.circle(img, (ex + a, ey + b), r, (0, 255, 0), 2)
    showImage(img, imgName)

def findFrontFace(img, imgName):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = front_face_cascade.detectMultiScale(imgray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = img[y:y + h, x:x + w]
        roi_gray = imgray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            #matchOpenEye(img[ex:ex + ew, ey:ey + eh], imgName)
            findPupil(img, x + ex, y + ey, ew, eh, imgName)
    showImage(img, imgName)
    return len(face) > 0

def findSideFace(img, imgName):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = side_face_cascade.detectMultiScale(imgray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    showImage(img, imgName)
    return len(face) > 0

def showImage(img, imgName):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

front_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
side_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

testImages = ["tian_side_eye.dng", "tian_closed_eye.dng", "newtest.jpg", "side_tian.jpg", "bry.jpg", "newtest2.jpg"]

for name in testImages:
    image = cv2.imread(name)
    x, y, z = np.shape(image)
    if x > 2000:
        image = imutils.resize(image, width=1800)
    if y > 1000:
        image = imutils.resize(image, height=1000)
    image = subtractBody(image)
    print("Head pose is facing to the side: " + str(findSideFace(image, name)))
    print("Head pose is facing to the front: " + str(findFrontFace(image, name)))


"""
image = subtractBody(image)
image2 = subtractBody(image2)
makeContour(image, testImage)
makeContour(image2, testImage2)
"""