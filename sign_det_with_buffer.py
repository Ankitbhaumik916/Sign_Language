import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

# Load model
model = load_model('asl_cnn_model.h5')
labels = [chr(i) for i in range(65, 91) if i not in [74, 90]]  # A-Y except J, Z

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

# Buffers
prediction_buffer = deque(maxlen=15)  # Last 15 predictions
word_queue = deque(maxlen=20)         # Final word (max 20 letters)
last_stable = ''
cooldown = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = max(int(min(x_coords) * w) - 20, 0)
            x_max = min(int(max(x_coords) * w) + 20, w)
            y_min = max(int(min(y_coords) * h) - 20, 0)
            y_max = min(int(max(y_coords) * h) + 20, h)

            hand_img = frame[y_min:y_max, x_min:x_max]

            try:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (28, 28))
                hand_img = hand_img / 255.0
                hand_img = hand_img.reshape(1, 28, 28, 1)

                # Predict
                pred = model.predict(hand_img, verbose=0)
                letter = labels[np.argmax(pred)]
                prediction_buffer.append(letter)

                # Stabilize every few frames
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]
                    if most_common != last_stable and cooldown == 0:
                        word_queue.append(most_common)
                        last_stable = most_common
                        cooldown = 10  # Skip next 10 frames
                if cooldown > 0:
                    cooldown -= 1

                # Draw prediction
                cv2.putText(frame, f'Letter: {last_stable}', (x_min, y_min - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

            except Exception as e:
                print(f"Error: {e}")

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display current word
    current_word = ''.join(word_queue)
    cv2.putText(frame, f'Word: {current_word}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('Sign2Word - Ankit Edition 💬', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):  # Clear word
        word_queue.clear()

cap.release()
cv2.destroyAllWindows()
