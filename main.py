import pygame
import cv2
import mediapipe as mp
print(mp.__version__)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

pygame.init()
pygame.mixer.init()

pygame.mixer.music.load("music.mp3")  # Replace with your music file path

is_playing = False


def recognize_gesture(hand_landmarks):

    thumb_tip = hand_landmarks[4]  # Thumb tip
    index_tip = hand_landmarks[8]  # Index finger tip
    middle_tip = hand_landmarks[12]  # Middle finger tip
    wrist = hand_landmarks[0]  # Wrist

    # Thumbs Up Gesture (Thumb tip higher than index tip)
    if thumb_tip.y < index_tip.y and abs(thumb_tip.x - index_tip.x) > 0.1:
        return "thumbs_up"

    # Open Palm Gesture (Fingers extended and aligned)
    if (
        index_tip.y < wrist.y
        and middle_tip.y < wrist.y
        and thumb_tip.x < wrist.x
    ):
        return "palm"

    return "none"


cap = cv2.VideoCapture(0)
print("Show a thumbs up to play, show your palm to pause.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = recognize_gesture(hand_landmarks.landmark)
            if gesture == "thumbs_up" and not is_playing:
                print("Playing music...")
                pygame.mixer.music.play(-1)  # Loop the music
                is_playing = True
            elif gesture == "palm" and is_playing:
                print("Pausing music...")
                pygame.mixer.music.pause()
                is_playing = False

    cv2.imshow("Hand Gesture Music Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
hands.close()
