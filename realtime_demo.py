import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# --- 설정값 ---
MAX_SEQ_LENGTH = 100
MODEL_SAVE_PATH = 'lstm_model.keras'

# --- 한글 폰트 설정 ---
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
try:
    font = ImageFont.truetype(FONT_PATH, 30)
except IOError:
    print(f"❌ 폰트를 찾을 수 없습니다: {FONT_PATH}")
    print("다른 경로의 한글 폰트를 지정해주세요.")
    font = ImageFont.load_default()

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

def draw_korean_text(img, text, pos, font, color=(0, 255, 0)):
    """Pillow를 사용하여 OpenCV 이미지에 한글 텍스트를 그립니다."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 실시간 처리에 필요한 함수 (main.py에서 가져옴)
def preprocess_landmarks(landmarks_list):
    processed_frames = []
    for frame in landmarks_list:
        combined = []
        # 얼굴 랜드마크 제외
        for key in ["pose", "left_hand", "right_hand"]:
            lm = frame[key]
            if lm:
                combined.extend([[l.x, l.y, l.z] for l in lm.landmark])
            else:
                num_points = {"pose": 33, "left_hand": 21, "right_hand": 21}[key]
                combined.extend([[0,0,0]] * num_points)
                
        arr = np.array(combined)
        # 랜드마크가 하나도 없는 경우에 대한 예외 처리
        if arr.shape[0] == 0:
            # 포즈(33) + 왼손(21) + 오른손(21) = 75개의 랜드마크
            return np.zeros((len(landmarks_list), 75 * 3))
            
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
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
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
        pred_prob = model.predict(input_data, verbose=0)[0][0]
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
        
        if pred_prob > 0.5:
            current_prediction = "화재"
        else:
            current_prediction = "화장실"

    # 결과 텍스트 표시
    text = f"예측: {current_prediction} (신뢰도: {confidence:.2f})"
    frame = draw_korean_text(frame, text, (20, 30), font, (0, 255, 0))
    
    # 화면에 출력
    cv2.imshow('실시간 수어 인식', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close() 