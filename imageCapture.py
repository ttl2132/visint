import math
import cv2
# import PIL
# from PIL import Image
# import numpy as np
# import random
# from numba import jit
from segmentation import segment
from torchvision import models
# import os
import multiprocessing

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def videoImageCapture(filepath):
    cap = cv2.VideoCapture(filepath)

    frames = []

    framerate = cap.get(5)
    print("Framerate: ", framerate)
    # Loops over all frames in the video
    while cap.isOpened():
        frameID = cap.get(1)
        ret, frame = cap.read()
        if frame is None:
            print("Finished")
            break
        #frame = cv2.resize(frame, (320,180), interpolation=cv2.INTER_AREA)
        if frameID % (math.floor(framerate) * 3) == 0:
            # Every 3ish seconds, get a frame to remove the background on
            #print("if statement")
        #   cv2.imshow("frame", frame)
            frames.append(segment(dlab, frame))

        # Since there is no longer a display aspect, it simply continues
        #else:
        # Otherwise continue playing the video
        #    cv2.imshow("frame", frame)
        #    if cv2.waitKey(70) and 0xff == ord('q'):
        #        break

    cap.release()
    cv2.destroyAllWindows()
    return frames



# Haven't updated webcamImageCapture with background removal
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

# A method intended to draw contours onto an image. Wasn't used
def detectAndDraw(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2 = cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
    cv2.imshow("im2", im2)
    # cv2.waitKey()
    return im2


def main():
    print("IT has begun")
    # Initialization of semantig segmentation architecture
    print("Model created")
    videoImageCapture("testvid2.mp4")


if __name__ == "__main__":
    main()
