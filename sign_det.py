import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained CNN model
model = load_model('asl_cnn_model.h5')

# Map index to letters (excluding J and Z)
labels = [chr(i) for i in range(65, 91) if i not in [74, 90]]  # A-Y except J, Z

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box from landmarks
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w) - 20
            x_max = int(max(x_coords) * w) + 20
            y_min = int(min(y_coords) * h) - 20
            y_max = int(max(y_coords) * h) + 20

            # Make sure coords are in frame
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, w)
            y_max = min(y_max, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            try:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (28, 28))
                hand_img = hand_img / 255.0
                hand_img = hand_img.reshape(1, 28, 28, 1)

                # Predict
                pred = model.predict(hand_img, verbose=0)
                predicted_letter = labels[np.argmax(pred)]

                # Draw label
                cv2.putText(frame, f'Prediction: {predicted_letter}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing hand region: {e}")

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Sign Language Detector - Ankit Edition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
