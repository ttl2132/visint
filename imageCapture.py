import math
import cv2
#import PIL
#from PIL import Image
#import numpy as np
#import random
#from numba import jit
from segmentation import segment
from torchvision import models
#import os
import multiprocessing


frames = []


def videoImageCapture(filepath):
    cap = cv2.VideoCapture(filepath)
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    global samples, frames
    # v = Vibe()
    first = True
    framerate = cap.get(5)
    print("Framerate: ", framerate)
    index = 0
    while cap.isOpened():
        frameID = cap.get(1)
        ret, frame = cap.read()
        #print(frame.shape)
        frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)), interpolation = cv2.INTER_AREA)
        scalePercent = 1
        if frameID % (math.floor(framerate)*3) == 0:
            print("if statement")
            cv2.imshow("frame", frame)
            removeBack(frame)
            #frames.append(index)
            #p1 = multiprocessing.Process(target=removeBack, args=[index])
            #p1.start()

            # cv2.imwrite("frame.jpg", frame)
            # segment(dlab, "frame.jpg")
            # cv2.imshow("segmented", cv2.imread("newframe.jpg"))
            # if cv2.waitKey(70) and 0xff == ord('q'):
            #    break
        else:
            cv2.imshow("frame", frame)
        if cv2.waitKey(70) and 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def removeBack(frame):
    #cv2.imwrite("frame.jpg", frame)
    im = segment(dlab, frame)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow("segmented", im)


def webcamImageCapture():
    cap = cv2.VideoCapture(0)
    first = True
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # This here is an opencv background removal thingy that seems to work better than the vibe thing

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Gets a retainer and frame, frame is essentially the image data
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgmask = fgbg.apply(frame)
        # cnts = detectAndDraw(fgmask.copy())
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        cv2.imshow("mask", res)
        detectAndDraw(res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def detectAndDraw(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2 = cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
    cv2.imshow("im2", im2)
    # cv2.waitKey()
    return im2


dlab = None


def main():
    print("IT has begun")
    global dlab

    dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
    print("Model created")
    # webcamImageCapture()
    videoImageCapture("testvid2.mp4")
    pass


if __name__ == "__main__":
    main()
