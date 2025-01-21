import cv2 as cv
import numpy as np
import keras
from keras.models import load_model

background = None
calibration_frames = 30  # Frames to calibrate during recalibration
frame_count = 0


def calibrate_background(region, accumulated_weight):
    global background
    if background is None:
        background = region.copy().astype('float')
        return

    cv.accumulateWeighted(region, background, accumulated_weight)


def segment(image, threshold=25):
    global background
    diff = cv.absdiff(background.astype('uint8'), image)

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
            return 'Blank'
        case 1:
            return 'Ok'
        case 2:
            return 'Thumbs up'
        case 3:
            return 'Thumbs down'
        case 4:
            return 'Fist'
        case 5:
            return 'Five'
        case _:
            return 'Blank'


def main():
    global background, frame_count
    accumulated_weight = 0.1
    recalibrating = True
    model = load_model('./hand_recognition_model.keras')

    cap = cv.VideoCapture(0)
    fps = int(cap.get(cv.CAP_PROP_FPS))

    top, right, bottom, left = 0, 300, 300, 600

    while True:
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame_copy = frame.copy()
        region = cv.cvtColor(frame[top:bottom, right:left], cv.COLOR_BGR2GRAY)
        region = cv.GaussianBlur(region, (7, 7), 1.0)

        if recalibrating:
            if frame_count < calibration_frames:
                calibrate_background(region, accumulated_weight)
                cv.putText(
                    frame_copy,
                    "Recalibrating...",
                    (70, 45),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                frame_count += 1
            else:
                recalibrating = False

        else:
            segmented_region = segment(region)
            if segmented_region:
                thresholded, segmented = segmented_region
                cv.drawContours(
                    frame_copy, [segmented + (right, top)], -1, (0, 0, 255)
                )
                cv.imwrite("temp.jpg", thresholded)
                gesture = get_predicted_gesture(model)
                cv.putText(
                    frame_copy, gesture, (70, 45),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                cv.imshow("thresholded", thresholded)

        cv.rectangle(frame_copy, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.imshow("video feed", frame_copy)

        keypress = cv.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        elif keypress == ord("r"):
            recalibrating = True
            background = None
            frame_count = 0
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
