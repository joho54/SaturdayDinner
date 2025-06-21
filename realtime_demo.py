import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# --- 설정값 ---
MAX_SEQ_LENGTH = 100
MODEL_SAVE_PATH = 'lstm_model.keras'

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드
try:
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print(f"✅ 모델 로딩 성공: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    print("먼저 main.py를 실행하여 모델을 학습하고 저장해주세요.")
    exit()

# 실시간 처리에 필요한 함수 (main.py에서 가져옴)
def preprocess_landmarks(landmarks_list):
    processed_frames = []
    for frame in landmarks_list:
        combined = []
        for key in ["pose", "left_hand", "right_hand", "face"]:
            lm = frame[key]
            if lm:
                combined.extend([[l.x, l.y, l.z] for l in lm.landmark])
            else:
                num_points = {"pose": 33, "left_hand": 21, "right_hand": 21, "face": 468}[key]
                combined.extend([[0,0,0]] * num_points)
                
        arr = np.array(combined)
        root = arr[0].copy()
        arr -= root
        
        max_val = np.max(np.abs(arr))
        if max_val > 0:
            arr /= max_val
            
        processed_frames.append(arr.flatten())
        
    return np.array(processed_frames)

# 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

sequence = deque(maxlen=MAX_SEQ_LENGTH)
current_prediction = ""
confidence = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 랜드마크 추출
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    # 랜드마크 그리기
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 랜드마크 데이터 처리 및 시퀀스에 추가
    frame_data = {
        "pose": results.pose_landmarks,
        "left_hand": results.left_hand_landmarks,
        "right_hand": results.right_hand_landmarks,
        "face": results.face_landmarks
    }
    sequence.append(frame_data)

    # 시퀀스가 꽉 찼을 때만 예측
    if len(sequence) == MAX_SEQ_LENGTH:
        # 전처리
        processed_sequence = preprocess_landmarks(list(sequence))
        input_data = np.expand_dims(processed_sequence, axis=0)
        
        # 예측
        pred_prob = model.predict(input_data)[0][0]
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
        
        if pred_prob > 0.5:
            current_prediction = "화재"
        else:
            current_prediction = "화장실"

    # 결과 텍스트 표시
    text = f"Prediction: {current_prediction} ({confidence:.2f})"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 화면에 출력
    cv2.imshow('Real-time Sign Language Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close() 