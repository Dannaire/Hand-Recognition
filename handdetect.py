import cv2
import mediapipe as mp

class FingerGesture:
    def __init__(self, name, tip_id, pip_id, label):
        self.name = name       
        self.tip_id = tip_id   
        self.pip_id = pip_id    
        self.label = label      

    
    def is_raised(self, landmark_dict):
        tip_y = landmark_dict[self.tip_id][1]
        pip_y = landmark_dict[self.pip_id][1]
        return tip_y < pip_y  # jika ujung jari lebih atas dari sendi, berarti terangkat
    
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

fingers = [
    FingerGesture("thumb", 4, 2, "k"),
    FingerGesture("index", 8, 6, "o"),
    FingerGesture("middle", 12, 10, "m"),
    FingerGesture("ring", 16, 14, "j"),
    FingerGesture("pinky", 20, 18, "ar")
]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_chars = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            landmark_dict = {id: (int(lm.x * w), int(lm.y * h)) for id, lm in enumerate(hand_landmarks.landmark)}

            for finger in fingers:
                if finger.is_raised(landmark_dict):
                    detected_chars.append(finger.label)
                    x, y = landmark_dict[finger.tip_id]
                    cv2.putText(frame, finger.label, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 255), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if all(label in detected_chars for label in ["k", "o", "m", "j", "ar"]):
        cv2.putText(frame, "KOMJAR VS EVERYBODY", (10, 60), 
            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        
    cv2.imshow("Hand Gesture: k o m j ar", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
