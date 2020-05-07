import cv2
import PIL
from PIL import Image
import numpy as np
import random
from numba import jit


# from c_func import func1, func2
# index_dic = {1:(-1,-1),2:(-1,0),3:(-1,1),4:(0,-1),5:(0,0),6:(0,1),7:(1,-1),8:(1,0),9:(1,1)}


@jit
def func1(frame, N):
    height, width, channel = frame.shape
    print(height)
    print(width)
    num = 0
    samples = np.zeros((height, width, N, channel), np.int32)
    for i in range(height):
        for j in range(width):
            #print(num)
            for k in range(N):
                x_shift = random.randint(-1, 1)
                y_shift = random.randint(-1, 1)
                y = i + y_shift
                x = j + x_shift
                # deal with bound
                if x < 0:
                    x = 0
                if x > width - 1:
                    x = width - 1
                if y < 0:
                    y = 0
                if y > height - 1:
                    y = height - 1
                samples[i][j][k][:] = frame[y, x, :]
            num += 1
    return samples


@jit
def func2(frame, samples, min_count, N, Radius, Phi):
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    num = 0
    for i in range(height):
        for j in range(width):
            # print(num)
            count = 0
            index = 0
            while (count < min_count and index < N):
                disdance = color_space_disdance(frame[i][j], samples[i][j][index])
                if disdance < Radius:
                    count += 1
                index += 1
            if (count >= min_count):
                rand = random.randint(0, Phi - 1)
                if rand == 0:
                    rand_index = random.randint(0, N - 1)
                    samples[i][j][rand_index] = frame[i][j]
                rand = random.randint(0, Phi - 1)
                if rand == 0:
                    rand_index = random.randint(0, N - 1)
                    x_shift = random.randint(-1, 1)
                    y_shift = random.randint(-1, 1)
                    y = i + y_shift
                    x = j + x_shift
                    if x < 0:
                        x = 0
                    if x > width - 1:
                        x = width - 1
                    if y < 0:
                        y = 0
                    if y > height - 1:
                        y = height - 1
                    samples[y][x][rand_index] = frame[i][j]
            else:
                mask[i][j] = 255
            num += 1
    return mask, samples


def color_space_disdance(i, j):
    if len(i) == 3:
        return ((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2 + (i[2] - j[2]) ** 2) ** 0.5
    elif len(i) == 1:
        return abs(i - j)
    else:
        raise ValueError("color space error")


class Vibe:
    def __init__(self, min_count=2, N=20, Radius=20, Phi=16):
        self.min_count = min_count
        self.N = N
        self.Radius = Radius
        self.Phi = Phi

    def init(self, frame):
        self.samples = func1(frame, self.N)

    def test_and_update(self, frame):
        mask, samples = func2(frame, self.samples, self.min_count, self.N, self.Radius, self.Phi)
        return mask


def videoImageCapture(filepath):
    cap = cv2.VideoCapture(filepath)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    global samples
    #v = Vibe()
    first = True
    while cap.isOpened():
        ret, frame = cap.read()
        scalePercent = 1
        width = int(frame.shape[1] * scalePercent)
        height = int(frame.shape[0] * scalePercent)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(frame)

        cv2.imshow("mask", fgmask)
        if cv2.waitKey(70) and 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def webcamImageCapture():
    cap = cv2.VideoCapture(0)
    first = True
    fgbg = cv2.createBackgroundSubtractorMOG2()
    #This here is an opencv background removal thingy that seems to work better than the vibe thing

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Gets a retainer and frame, frame is essentially the image data
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgmask = fgbg.apply(frame)

        cv2.imshow("mask", fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def detectAndDraw(im):
    pass

def main():
    webcamImageCapture()
    #videoImageCapture("tester.mp4")
    pass


if __name__ == "__main__":
    main()
