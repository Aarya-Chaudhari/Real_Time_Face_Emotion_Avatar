import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils
LIPS = mp_face_mesh.FACEMESH_LIPS
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Emotion settings
emotions = {
    "happy": {"emoji": "😊", "color": (0, 255, 0)},
    "sad": {"emoji": "😢", "color": (255, 0, 0)},
    "angry": {"emoji": "😠", "color": (0, 0, 255)},
    "surprise": {"emoji": "😲", "color": (0, 255, 255)},
    "neutral": {"emoji": "😐", "color": (255, 255, 255)},
    "fear": {"emoji": "😨", "color": (255, 140, 0)},
    "disgust": {"emoji": "🤢", "color": (138, 43, 226)}
}

def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_emotion(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    iris_left = landmarks[468]
    eye_center = landmarks[33]

    face_width = distance(landmarks[234], landmarks[454])

    # Features
    mouth_open = distance(top_lip, bottom_lip) / face_width
    mouth_stretch = distance(left_mouth, right_mouth) / face_width
    eye_open = (distance(left_eye_top, left_eye_bottom) + distance(right_eye_top, right_eye_bottom)) / (2 * face_width)
    
    # SAD: Is iris significantly lower than middle of eyelid?
    eye_top_avg = (left_eye_top.y + right_eye_top.y) / 2
    eye_bottom_avg = (left_eye_bottom.y + right_eye_bottom.y) / 2
    iris_avg = iris_left.y
    eye_center_y = (eye_top_avg + eye_bottom_avg) / 2
    sad_offset = iris_avg - eye_center_y  # How far iris moved down from center

    # DEBUG (optional)
    print(f"[DEBUG] mouth_open={mouth_open:.3f}, mouth_stretch={mouth_stretch:.3f}, eye_open={eye_open:.3f}, sad_offset={sad_offset:.3f}")

    # Emotion rules (tuned)
    if mouth_stretch > 0.40 and mouth_open < 0.06:
        return "happy"
    elif mouth_open >= 0.12:
        return "surprise"
    elif 0.06 < mouth_open < 0.12:
        return "fear"
    elif sad_offset > 0.01 and eye_open < 0.04:
        return "sad"
    elif mouth_open < 0.03 and eye_open < 0.08 and mouth_stretch < 0.38:
        return "disgust"
    elif eye_open > 0.096 and mouth_open < 0.06:
        return "angry"
    else:
        return "neutral"

# Webcam loop
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    avatar_canvas = np.zeros_like(frame)
    emotion = "neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_list = face_landmarks.landmark
            emotion = get_emotion(landmark_list)
            color = emotions[emotion]["color"]
            emoji = emotions[emotion]["emoji"]

            # Draw avatar
            drawing_utils.draw_landmarks(
                avatar_canvas, face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION, None,
                drawing_utils.DrawingSpec(color=color, thickness=1, circle_radius=1)
            )
            drawing_utils.draw_landmarks(
                avatar_canvas, face_landmarks, LIPS, None,
                drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            for idx in LEFT_IRIS + RIGHT_IRIS:
                pt = landmark_list[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(avatar_canvas, (cx, cy), 2, (0, 255, 255), -1)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    label = f"{emotion.upper()} {emotions[emotion]['emoji']}"
    cv2.putText(avatar_canvas, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, emotions[emotion]["color"], 3)
    cv2.putText(avatar_canvas, f"FPS: {int(fps)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

    # Show windows
    cv2.imshow("Webcam Feed", cv2.resize(frame, (640, 480)))
    cv2.imshow("Avatar Emotion Mesh", cv2.resize(avatar_canvas, (640, 480)))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
