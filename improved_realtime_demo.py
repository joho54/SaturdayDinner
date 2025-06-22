import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from PIL import ImageFont, ImageDraw, Image
from scipy.interpolate import interp1d

# --- 설정값 ---
TARGET_SEQ_LENGTH = 30  # 정규화된 시퀀스 길이
MODEL_SAVE_PATH = 'improved_transformer_model.keras'
ACTIONS = ["Fire", "Toilet", "None"]

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
    print(f"✅ 개선된 모델 로딩 성공: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    print("먼저 improved_main.py를 실행하여 개선된 모델을 학습하고 저장해주세요.")
    exit()

# 모델 구조 확인
print(model.summary())

# 입력 shape 확인
print("모델 입력 shape:", model.input_shape)

def draw_korean_text(img, text, pos, font, color=(0, 255, 0)):
    """Pillow를 사용하여 OpenCV 이미지에 한글 텍스트를 그립니다."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def normalize_sequence_length(sequence, target_length=30):
    """시퀀스 길이를 정규화합니다 (다운샘플링/업샘플링)."""
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    # 시간 축을 따라 보간
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    normalized_sequence = []
    for i in range(sequence.shape[1]):  # 각 특징 차원에 대해
        f = interp1d(x_old, sequence[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        normalized_sequence.append(f(x_new))
    
    return np.array(normalized_sequence).T

def extract_dynamic_features(sequence):
    """속도와 가속도 특징을 추출합니다."""
    # 속도 (이전 프레임 대비 변화량)
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    
    # 가속도 (속도의 변화율)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    
    # 원본 + 속도 + 가속도 결합
    dynamic_features = np.concatenate([sequence, velocity, acceleration], axis=1)
    
    return dynamic_features

def convert_to_relative_coordinates(landmarks_list):
    """절대 좌표를 어깨 중심 상대 좌표계로 변환합니다."""
    relative_landmarks = []
    
    for frame in landmarks_list:
        if not frame["pose"]:
            # 포즈 랜드마크가 없으면 원본 반환
            relative_landmarks.append(frame)
            continue
        
        pose_landmarks = frame["pose"].landmark
        
        # 어깨 중심점 계산 (왼쪽 어깨 + 오른쪽 어깨) / 2
        left_shoulder = pose_landmarks[11]  # 왼쪽 어깨
        right_shoulder = pose_landmarks[12]  # 오른쪽 어깨
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
        
        # 어깨 너비 계산 (정규화에 사용)
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        if shoulder_width == 0:
            shoulder_width = 1.0  # 0으로 나누기 방지
        
        # 새로운 프레임 데이터 생성
        new_frame = {}
        
        # 포즈 랜드마크 변환
        if frame["pose"]:
            relative_pose = []
            for landmark in pose_landmarks:
                rel_x = (landmark.x - shoulder_center_x) / shoulder_width
                rel_y = (landmark.y - shoulder_center_y) / shoulder_width
                rel_z = (landmark.z - shoulder_center_z) / shoulder_width
                relative_pose.append([rel_x, rel_y, rel_z])
            new_frame["pose"] = relative_pose
        
        # 손 랜드마크 변환
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
        # 빈 랜드마크 리스트인 경우 기본 시퀀스 반환
        return np.zeros((TARGET_SEQ_LENGTH, 675))  # 225*3 (원본+속도+가속도)
    
    # 1. 상대 좌표 변환
    relative_landmarks = convert_to_relative_coordinates(landmarks_list)
    
    # 2. 랜드마크 결합
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
            # 기본 크기로 0 채우기
            processed_frames.append(np.zeros(75 * 3))
    
    if not processed_frames:
        # 처리된 프레임이 없는 경우 기본 시퀀스 반환
        return np.zeros((TARGET_SEQ_LENGTH, 675))
    
    sequence = np.array(processed_frames)
    
    # 3. 시퀀스 길이 정규화
    if len(sequence) > 0:
        try:
            sequence = normalize_sequence_length(sequence, TARGET_SEQ_LENGTH)
            
            # 4. 동적 특징 추가
            sequence = extract_dynamic_features(sequence)
            
            # 5. 정규화 (0-1 범위로)
            sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence) + 1e-8)
            
            return sequence
        except Exception as e:
            print(f"⚠️ 시퀀스 처리 중 오류 발생: {e}")
            # 오류 발생 시 기본 시퀀스 반환
            return np.zeros((TARGET_SEQ_LENGTH, 675))
    
    return np.zeros((TARGET_SEQ_LENGTH, 675))

# 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

sequence = deque(maxlen=TARGET_SEQ_LENGTH)
current_prediction = ""
confidence = 0.0
pred_probs = np.zeros(len(ACTIONS))
pred_class_index = -1
prediction_counter = 0  # 예측 횟수 추적

print("🚀 개선된 실시간 수어 인식 시작!")
print("📝 사용법: 'q' 키를 눌러 종료")

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

    # 시퀀스가 충분히 쌓였을 때 예측 (매 5프레임마다)
    if len(sequence) >= TARGET_SEQ_LENGTH and prediction_counter % 5 == 0:
        try:
            print(f"🔍 예측 시도 #{prediction_counter}: 시퀀스 길이 {len(sequence)}")
            
            # 시퀀스 내용 확인 (디버깅용)
            if prediction_counter <= 35:  # 처음 몇 번만 확인
                pose_count = sum(1 for frame in sequence if frame["pose"] is not None)
                left_hand_count = sum(1 for frame in sequence if frame["left_hand"] is not None)
                right_hand_count = sum(1 for frame in sequence if frame["right_hand"] is not None)
                print(f"📊 랜드마크 통계: 포즈={pose_count}, 왼손={left_hand_count}, 오른손={right_hand_count}")
            
            processed_sequence = improved_preprocess_landmarks(list(sequence))
            print(f"📊 전처리된 시퀀스 형태: {processed_sequence.shape}")
            
            # 시퀀스 데이터 변화 확인 (디버깅용)
            if prediction_counter <= 35:
                seq_mean = np.mean(processed_sequence)
                seq_std = np.std(processed_sequence)
                print(f"📈 시퀀스 통계: 평균={seq_mean:.6f}, 표준편차={seq_std:.6f}")
            
            # 시퀀스 형태 확인
            if processed_sequence.shape == (TARGET_SEQ_LENGTH, 675):
                input_data = np.expand_dims(processed_sequence, axis=0)
                print(f"🎯 모델 입력 형태: {input_data.shape}")
                
                # 입력 shape 확인
                print("실제 입력 shape:", input_data.shape)
                
                # processed_sequence 일부 값 출력
                print("processed_sequence[0, :10]:", processed_sequence[0, :10])
                
                # model.predict() 호출 전후로 pred_probs 출력
                print("예측 전 pred_probs:", pred_probs)
                pred_probs = model.predict(input_data, verbose=0)[0]
                print("예측 후 pred_probs:", pred_probs)
                
                pred_class_index = np.argmax(pred_probs)
                
                current_prediction = ACTIONS[pred_class_index]
                confidence = pred_probs[pred_class_index]
                
                print(f"✅ 예측 #{prediction_counter}: {current_prediction} (신뢰도: {confidence:.3f})")
                print(f"📈 확률 분포: Fire={pred_probs[0]:.3f}, Toilet={pred_probs[1]:.3f}, None={pred_probs[2]:.3f}")
            else:
                print(f"⚠️ 시퀀스 형태 오류: {processed_sequence.shape}, 예상: ({TARGET_SEQ_LENGTH}, 675)")
                
        except Exception as e:
            print(f"❌ 예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    prediction_counter += 1

    # --- 결과 시각화 ---
    
    # 1. 예측 결과 텍스트
    display_label = {"Fire": "화재", "Toilet": "화장실", "None": "없음"}.get(current_prediction, "")
    if current_prediction == 'None' and confidence < 0.8:
        display_text = "..."
    else:
        display_text = f"예측: {display_label} (신뢰도: {confidence:.2f})"
        
    frame = draw_korean_text(frame, display_text, (20, 30), font, (0, 255, 0))

    # 2. 확률 막대그래프
    bar_start_x = frame.shape[1] - 300

    for i, prob in enumerate(pred_probs):
        action_korean = {"Fire": "화재", "Toilet": "화장실", "None": "없음"}.get(ACTIONS[i])
        y_pos = 50 + i * 40

        # 막대그래프 배경
        cv2.rectangle(frame, (bar_start_x, y_pos), (bar_start_x + 250, y_pos + 30), (200, 200, 200), -1)
        
        # 확률 막대
        bar_width = int(prob * 250)
        bar_color = (100, 100, 100) # 기본 회색
        if i == pred_class_index:
            bar_color = (0, 255, 0) # 예측된 클래스는 녹색으로 강조

        cv2.rectangle(frame, (bar_start_x, y_pos), (bar_start_x + bar_width, y_pos + 30), bar_color, -1)
        
        # 텍스트
        text_on_bar = f"{action_korean}: {prob*100:.1f}%"
        frame = draw_korean_text(frame, text_on_bar, (bar_start_x + 5, y_pos), font, (0, 0, 0))

    # 3. 디버그 정보 표시
    debug_text = f"시퀀스 길이: {len(sequence)}/{TARGET_SEQ_LENGTH}, 예측 횟수: {prediction_counter}"
    frame = draw_korean_text(frame, debug_text, (20, frame.shape[0] - 90), font, (255, 255, 255))

    # 4. 개선 사항 표시
    info_text = "개선된 모델: Transformer + 동적 특징"
    frame = draw_korean_text(frame, info_text, (20, frame.shape[0] - 60), font, (255, 255, 0))

    # 화면에 출력
    cv2.imshow('개선된 실시간 수어 인식 (Transformer)', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
print("🎉 개선된 실시간 수어 인식 종료") 