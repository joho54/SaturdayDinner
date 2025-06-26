import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# --- 설정값 ---
MAX_SEQ_LENGTH = 30
MODEL_SAVE_PATH = 'fixed_transformer_model.keras'
ACTIONS = ["화재", "화장실", "화요일", "화약", "화상", "None"]

# --- 한글 폰트 설정 ---
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
try:
    font = ImageFont.truetype(FONT_PATH, 30)
except IOError:
    # print(f"❌ 폰트를 찾을 수 없습니다: {FONT_PATH}")
    # print("다른 경로의 한글 폰트를 지정해주세요.")
    font = ImageFont.load_default()

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드
try:
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    # print(f"✅ 수정된 모델 로딩 성공: {MODEL_SAVE_PATH}")
    # print("모델 구조:")
    model.summary()
except Exception as e:
    # print(f"❌ 모델 로딩 실패: {e}")
    # print("먼저 fix_training_data.py를 실행하여 모델을 학습하고 저장해주세요.")
    exit()

def draw_korean_text(img, text, pos, font, color=(0, 255, 0)):
    """Pillow를 사용하여 OpenCV 이미지에 한글 텍스트를 그립니다."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def normalize_sequence_length(sequence, target_length=30):
    """시퀀스 길이를 정규화합니다."""
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    normalized_sequence = []
    for i in range(sequence.shape[1]):
        f = np.interp(x_new, x_old, sequence[:, i])
        normalized_sequence.append(f)
    
    return np.array(normalized_sequence).T

def extract_dynamic_features(sequence):
    """속도와 가속도 특징을 추출합니다."""
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    dynamic_features = np.concatenate([sequence, velocity, acceleration], axis=1)
    return dynamic_features

def convert_to_relative_coordinates(landmarks_list):
    """절대 좌표를 어깨 중심 상대 좌표계로 변환합니다."""
    relative_landmarks = []
    
    for frame in landmarks_list:
        if not frame["pose"]:
            relative_landmarks.append(frame)
            continue
        
        pose_landmarks = frame["pose"].landmark
        
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
        
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        if shoulder_width == 0:
            shoulder_width = 1.0
        
        new_frame = {}
        
        if frame["pose"]:
            relative_pose = []
            for landmark in pose_landmarks:
                rel_x = (landmark.x - shoulder_center_x) / shoulder_width
                rel_y = (landmark.y - shoulder_center_y) / shoulder_width
                rel_z = (landmark.z - shoulder_center_z) / shoulder_width
                relative_pose.append([rel_x, rel_y, rel_z])
            new_frame["pose"] = relative_pose
        
        for hand_key in ["left_hand", "right_hand"]:
            if frame[hand_key]:
                relative_hand = []
                for landmark in frame[hand_key].landmark:
                    rel_x = (landmark.x - shoulder_center_x) / shoulder_width
                    rel_y = (landmark.y - shoulder_center_y) / shoulder_width
                    rel_z = (landmark.z - shoulder_center_z) / shoulder_width
                    relative_hand.append([rel_x, rel_y, rel_z])
                new_frame[hand_key] = relative_hand
            else:
                new_frame[hand_key] = None
        
        relative_landmarks.append(new_frame)
    
    return relative_landmarks

def improved_preprocess_landmarks(landmarks_list):
    """개선된 랜드마크 전처리 함수."""
    if not landmarks_list:
        return np.zeros((MAX_SEQ_LENGTH, 675))
    
    relative_landmarks = convert_to_relative_coordinates(landmarks_list)
    
    processed_frames = []
    for frame in relative_landmarks:
        combined = []
        for key in ["pose", "left_hand", "right_hand"]:
            if frame[key]:
                if isinstance(frame[key], list):
                    combined.extend(frame[key])
                else:
                    combined.extend([[l.x, l.y, l.z] for l in frame[key].landmark])
            else:
                num_points = {"pose": 33, "left_hand": 21, "right_hand": 21}[key]
                combined.extend([[0,0,0]] * num_points)
        
        if combined:
            processed_frames.append(np.array(combined).flatten())
        else:
            processed_frames.append(np.zeros(75 * 3))
    
    if not processed_frames:
        return np.zeros((MAX_SEQ_LENGTH, 675))
    
    sequence = np.array(processed_frames)
    
    if len(sequence) > 0:
        try:
            sequence = normalize_sequence_length(sequence, MAX_SEQ_LENGTH)
            sequence = extract_dynamic_features(sequence)
            
            # 정규화 개선: 더 강한 정규화
            sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)
            
            return sequence
        except Exception as e:
            # print(f"⚠️ 시퀀스 처리 중 오류 발생: {e}")
            return np.zeros((MAX_SEQ_LENGTH, 675))
    
    return np.zeros((MAX_SEQ_LENGTH, 675))

# 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # print("❌ 웹캠을 열 수 없습니다.")
    exit()

sequence = deque(maxlen=MAX_SEQ_LENGTH)
current_prediction = ""
confidence = 0.0
pred_probs = np.zeros(len(ACTIONS))
pred_class_index = -1
prediction_count = 0

# None 클래스명 자동 추출
NONE_CLASS = ACTIONS[-1]

# print("🚀 수정된 실시간 수어 인식 시작!")
# print("📝 사용법: 'q' 키를 눌러 종료")

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
        prediction_count += 1
        
        # 랜드마크 통계
        pose_count = sum(1 for frame in sequence if frame["pose"] is not None)
        left_hand_count = sum(1 for frame in sequence if frame["left_hand"] is not None)
        right_hand_count = sum(1 for frame in sequence if frame["right_hand"] is not None)
        
        processed_sequence = improved_preprocess_landmarks(list(sequence))
        input_data = np.expand_dims(processed_sequence, axis=0)
        
        # 예측 전 상태 출력
        # print(f"\n🔍 예측 시도 #{prediction_count}: 시퀀스 길이 {len(sequence)}")
        # print(f"📊 랜드마크 통계: 포즈={pose_count}, 왼손={left_hand_count}, 오른손={right_hand_count}")
        # print(f"📊 전처리된 시퀀스 형태: {processed_sequence.shape}")
        # print(f"📈 시퀀스 통계: 평균={np.mean(processed_sequence):.6f}, 표준편차={np.std(processed_sequence):.6f}")
        # print(f"🎯 모델 입력 형태: {input_data.shape}")
        # print(f"실제 입력 shape: {input_data.shape}")
        # print(f"processed_sequence[0, :10]: {processed_sequence[0, :10]}")
        # print(f"예측 전 pred_probs: {pred_probs}")
        
        # 예측
        pred_probs = model.predict(input_data, verbose=0)[0]
        pred_class_index = np.argmax(pred_probs)
        
        # print(f"예측 후 pred_probs: {pred_probs}")
        
        current_prediction = ACTIONS[pred_class_index]
        confidence = pred_probs[pred_class_index]

        # print(f"✅ 예측 #{prediction_count}: {current_prediction} (신뢰도: {confidence:.3f})")
        # print(f"📈 확률 분포: {', '.join([f'{ACTIONS[i]}={pred_probs[i]:.3f}' for i in range(len(ACTIONS))])}")

    # --- 결과 시각화 ---
    
    # 1. 예측 결과 텍스트
    display_label = {a: a for a in ACTIONS}
    label_text = display_label.get(current_prediction, "")
    if current_prediction == NONE_CLASS and confidence < 0.8:
        display_text = "..."
    else:
        display_text = f"예측: {label_text} (신뢰도: {confidence:.2f})"
        
    frame = draw_korean_text(frame, display_text, (20, 30), font, (0, 255, 0))

    # 2. 확률 막대그래프
    bar_start_x = frame.shape[1] - 300

    for i, prob in enumerate(pred_probs):
        action_korean = ACTIONS
        y_pos = 50 + i * 40

        # 막대그래프 배경
        cv2.rectangle(frame, (bar_start_x, y_pos), (bar_start_x + 250, y_pos + 30), (200, 200, 200), -1)
        
        # 확률 막대
        bar_width = int(prob * 250)
        bar_color = (100, 100, 100)
        if i == pred_class_index:
            bar_color = (0, 255, 0)

        cv2.rectangle(frame, (bar_start_x, y_pos), (bar_start_x + bar_width, y_pos + 30), bar_color, -1)
        
        # 텍스트
        text_on_bar = f"{action_korean[i]}: {prob*100:.1f}%"
        frame = draw_korean_text(frame, text_on_bar, (bar_start_x + 5, y_pos), font, (0, 0, 0))

    # 화면에 출력
    cv2.imshow('수정된 실시간 수어 인식', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close() 