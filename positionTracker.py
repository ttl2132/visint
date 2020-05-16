import numpy as np
import cv2
import imutils
from imutils import face_utils
import dlib
import math

"""
The xml files for the Haar cascade classifiers are sourced from the OpenCV library at:
https://github.com/opencv/opencv/tree/master/data/haarcascades
"""


def subtract_background(img):
    x, y, z = np.shape(img)
    for l in range(x):
        for w in range(y):
            if sum(img[l][w]) / 3 >= 250:
                for color in range(z):
                    img[l][w][color] = 0
    return img


def subtract_body(img):
    x, y, z = np.shape(img)
    for l in range(x):
        for w in range(y):
            avg = sum(img[l][w]) / 3
            if img[l][w][2] < avg - 3:
                for color in range(z):
                    img[l][w][color] = 0
    return img


def find_pupil(img, ex, ey, ew, eh, imgName):
    cropped_img = img[ex: ex + ew, ey: ey + eh]
    imgray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # Apply Hough transform on the blurred image.
    circles = cv2.HoughCircles(imgray,
                               cv2.HOUGH_GRADIENT, 1, 25, param1=50,
                               param2=20, minRadius=eh // 10, maxRadius=40)
    if circles is not None:
        # Convert the circle parameters a, b and r to integers.
        circles = np.round(circles[0, :]).astype("int")
        for values in circles:
            a, b, r = values[0], values[1], values[2]
            cv2.circle(img, (ex + a, ey + b), r, (0, 255, 0), 2)
    show_image(img, imgName)


def is_front_facing(img, imgName, findEyes=False):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = front_face_cascade.detectMultiScale(imgray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        eyes = eye_cascade.detectMultiScale(imgray[y:y + h, x:x + w])
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            if findEyes:
                find_pupil(img, x + ex, y + ey, ew, eh, imgName)
                # matchOpenEye(img[ex:ex + ew, ey:ey + eh], imgName)
                # matchClosedEye(img[ex:ex + ew, ey:ey + eh], imgName)
    show_image(img, imgName)
    return len(face) > 0


def is_side_facing(img, imgName):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = side_face_cascade.detectMultiScale(imgray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_image(img, imgName)
    return len(face) > 0


def show_image(img, imgName):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_rect_values(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def calculate_distance(pt, pt2):
    x1, y1 = pt[0], pt[1]
    x2, y2 = pt2[0], pt2[1]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def eye_aspect_ratio(eye):
    left_vertical = calculate_distance(eye[1], eye[5])
    right_vertical = calculate_distance(eye[2], eye[4])
    horizontal = calculate_distance(eye[0], eye[3])
    return (left_vertical + right_vertical) / (2.0 * horizontal)

def shape_to_np(shape):
    # initialize the list of (x, y)-coordinates
    converted = np.zeros((68, 2))
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(68):
        converted[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return converted

#The eye parameter is given as a the pupil's bounding corners clockwise starting from the top left.
def pupil_area_percentage(img, imgName, eye):
    x, y, w, h = eye[0][0], eye[0][1], eye[2][0]-eye[0][0], eye[2][0]-eye[0][0]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(img, imgName)
    white_count = 0
    pupil_count = 0
    #This is the parameter for how dark the pixel needs to be for it to count as part of the pupil.
    color_limit = 100
    for row in range(w):
        for col in range(h):
            if imgray[x+row][y+col] > color_limit:
                white_count+=1
            else:
                pupil_count+=1
    return pupil_count / (white_count+ pupil_count)

""" Some code is sourced from the following tutorial:
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
"""
def dlibs_predict(img, imgName):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(imgray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(imgray, rect)
        shape = face_utils.shape_to_np(shape)
        x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        #This parameter is tested.
        closedEyeLimit = 0.15
        lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
        leftPupil = np.array([shape[lStart+1], shape[lStart+2], shape[lStart+4], shape[lStart+5]])
        rightPupil = np.array([shape[rStart + 1], shape[rStart + 2], shape[rStart + 4], shape[rStart + 5]])
        if eye_aspect_ratio(leftEye) < closedEyeLimit and eye_aspect_ratio(rightEye) < closedEyeLimit:
            print("Eyes are closed")
        elif pupil_area_percentage(img, imgName, leftPupil) < 0.1 or pupil_area_percentage(img, imgName, rightPupil) < 0.1:
            print("Eyes are looking sideways")
        else:
            print("Eyes are looking forward")
    show_image(img, imgName)


front_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
side_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

testImages = ["newtest.jpg", "Fed.jpg", "tian_side_eye.dng", "tian_closed_eye.dng", "side_tian.jpg", "bry.jpg"]

for name in testImages:
    image = cv2.imread(name)
    x, y, z = np.shape(image)
    if x > 2000:
        image = imutils.resize(image, width=1800)
    if y > 1000:
        image = imutils.resize(image, height=1000)
    dlibs_predict(image, name)
    print("Head pose is facing to the side: " + str(is_side_facing(image, name)))
"""
image = subtractBody(image)
image2 = subtractBody(image2)
"""
