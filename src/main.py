import cv2 as cv
import numpy as np
import os
import keras
import glob
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

bg = None


def load_model(path):
    model = keras.models.load_model(path)

    print(model.summary())
    return model


def run_average(image, accumulated_weight):
    global bg
    if bg is None:
        bg = image.copy().astype('float')
        return

    cv.accumulateWeighted(image, bg, accumulated_weight)


def segment(image, threshold=25):
    global bg
    diff = cv.absdiff(bg.astype('uint8'), image)

    thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(
        thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv.contourArea)
        return thresholded, segmented


def get_predicted_gesture(model):
    image = cv.imread('temp.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (100, 100))

    image = image.reshape(1, 100, 100, 1)

    prediction = model.predict_on_batch(image)
    gesture = np.argmax(prediction)

    match gesture:
        case 0:
            return 'blank'
        case 1:
            return 'ok'
        case 2:
            return 'thumbs up'
        case 3:
            return 'thumbs down'
        case 4:
            return 'fist'
        case 5:
            return 'five'
        case _:
            return 'blank'


def main():
    acummulated_weight = 0.5
    num_frames = 0
    k = 0
    model = load_model('./hand_recognition_model.keras')

    cap = cv.VideoCapture(0)
    fps = int(cap.get(cv.CAP_PROP_FPS))

    top, right, bottom, left = 10, 350, 225, 590

    while True:
        ret, frame = cap.read()

        frame = cv.resize(frame, (700, 700))
        frame = cv.flip(frame, 1)

        frame_copy = frame.copy()
        h, w = frame.shape[:2]

        region = frame[top:bottom, right:left]
        region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
        region = cv.GaussianBlur(region, (7, 7), 1)

        if num_frames < 30:
            run_average(region, acummulated_weight)
        else:
            hand = segment(region)
            if hand:
                thresholded, segmented = hand
                cv.drawContours(
                    frame_copy, [segmented + (right, top)], -1, (0, 0, 255))

                if k % (fps / 6) == 0:
                    cv.imwrite('temp.jpg', thresholded)
                    gesture = get_predicted_gesture(model)
                    cv.putText(frame_copy, gesture, (70, 45),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv.imshow('thresholded', thresholded)

        cv.rectangle(frame_copy, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.imshow('video feed', frame_copy)

        k += 1
        num_frames += 1

        keypress = cv.waitKey(1) & 0xFF
        if keypress == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
