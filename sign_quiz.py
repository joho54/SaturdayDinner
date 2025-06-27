import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import random
import json
import sys
import os
from collections import deque
from PIL import ImageFont, ImageDraw, Image


def load_model_info(model_info_path):
    """모델 정보 파일을 로드합니다."""
    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 모델 정보 파일 로드 실패: {e}")
        return None


def validate_args():
    """명령행 인자를 검증합니다."""
    if len(sys.argv) != 2:
        print("사용법: python3 sign_quiz.py <model_info.json>")
        print("예시: python3 sign_quiz.py info/model-info-20250626_220849.json")
        sys.exit(1)

    model_info_path = sys.argv[1]
    if not os.path.exists(model_info_path):
        print(f"❌ 모델 정보 파일을 찾을 수 없습니다: {model_info_path}")
        sys.exit(1)

    return model_info_path


# --- 모델 정보 로드 ---
model_info_path = validate_args()
model_info = load_model_info(model_info_path)

if not model_info:
    print("❌ 모델 정보를 로드할 수 없습니다.")
    sys.exit(1)

# --- 설정값 ---
MAX_SEQ_LENGTH = model_info["input_shape"][0]  # JSON에서 시퀀스 길이 로드
MODEL_SAVE_PATH = model_info["model_path"]  # JSON에서 모델 경로 로드
ACTIONS = model_info["labels"]  # JSON에서 라벨 로드
QUIZ_LABELS = [a for a in ACTIONS if a != "None"]  # None 제외한 퀴즈 라벨

print(f"📋 로드된 라벨: {ACTIONS}")
print(f"🎯 퀴즈 라벨: {QUIZ_LABELS}")
print(f"📊 모델 경로: {MODEL_SAVE_PATH}")
print(f"⏱️ 시퀀스 길이: {MAX_SEQ_LENGTH}")

# --- 한글 폰트 설정 ---
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
try:
    font = ImageFont.truetype(FONT_PATH, 30)
except IOError:
    font = ImageFont.load_default()

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# 모델 로드
try:
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print(f"✅ 모델 로드 성공: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()


# --- 유틸 함수 ---
def draw_korean_text(img, text, pos, font, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cleanup_resources():
    """5문제마다 리소스를 정리합니다."""
    global sequence, hold_counter, feedback, feedback_timer

    # 시퀀스 버퍼 초기화
    sequence.clear()

    # 상태 변수 초기화
    hold_counter = 0
    feedback = ""
    feedback_timer = 0

    print(f"🧹 리소스 정리 완료 (퀴즈 {quiz_number})")


# --- 퀴즈 상태 변수 ---
quiz_index = 0
current_label = QUIZ_LABELS[quiz_index]
feedback = ""
feedback_timer = 0
FEEDBACK_DURATION = 30  # 프레임 단위
CORRECT_HOLD_FRAMES = 10  # 0.5초(30fps 기준)
hold_counter = 0
quiz_number = 1

# --- 웹캠 실행 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

sequence = deque(maxlen=MAX_SEQ_LENGTH)


# --- 전처리 함수들 (fixed_realtime_demo.py에서 복사) ---
def normalize_sequence_length(sequence, target_length=30):
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
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    dynamic_features = np.concatenate([sequence, velocity, acceleration], axis=1)
    return dynamic_features


def convert_to_relative_coordinates(landmarks_list):
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
                combined.extend([[0, 0, 0]] * num_points)
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
            sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)
            return sequence
        except Exception as e:
            return np.zeros((MAX_SEQ_LENGTH, 675))
    return np.zeros((MAX_SEQ_LENGTH, 675))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 랜드마크 추출
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    frame_data = {
        "pose": results.pose_landmarks,
        "left_hand": results.left_hand_landmarks,
        "right_hand": results.right_hand_landmarks,
        "face": results.face_landmarks,
    }
    sequence.append(frame_data)

    # 시퀀스가 꽉 찼을 때만 예측
    if len(sequence) == MAX_SEQ_LENGTH:
        processed_sequence = improved_preprocess_landmarks(list(sequence))
        input_data = np.expand_dims(processed_sequence, axis=0)
        pred_probs = model.predict(input_data, verbose=0)[0]
        pred_class_index = np.argmax(pred_probs)
        current_prediction = ACTIONS[pred_class_index]
        confidence = pred_probs[pred_class_index]

        # 정답 판정(1초 연속 유지 필요)
        if not feedback:  # 피드백 표시 중이 아닐 때만 판정
            if current_prediction == current_label and confidence > 0.7:
                hold_counter += 1
            else:
                hold_counter = 0

            if hold_counter >= CORRECT_HOLD_FRAMES:
                feedback = "정답!"
                feedback_timer = FEEDBACK_DURATION
                hold_counter = 0

    # --- UI 표시 ---
    # 1. 현재 문제 라벨 (문제 번호 포함, 검은색)
    frame = draw_korean_text(
        frame, f"퀴즈 {quiz_number}: {current_label}", (20, 30), font, (0, 0, 0)
    )
    # 2. 피드백
    if feedback:
        frame = draw_korean_text(frame, feedback, (20, 70), font, (0, 0, 0))
        feedback_timer -= 1
        if feedback_timer <= 0:
            feedback = ""
            quiz_index = (quiz_index + 1) % len(QUIZ_LABELS)
            current_label = QUIZ_LABELS[quiz_index]
            quiz_number += 1

            # 5문제마다 리소스 정리
            if quiz_number % 5 == 0:
                cleanup_resources()

    # 3. 모델 판정 확률 표시 (개발용, 검정색)
    if "pred_probs" in locals():
        for i, prob in enumerate(pred_probs):
            label = ACTIONS[i]
            text = f"{label}: {prob*100:.1f}%"
            frame = draw_korean_text(frame, text, (20, 110 + i * 30), font, (0, 0, 0))

    # 4. 모델 정보 표시 (우측 상단)
    info_text = f"모델: {model_info['model_type']}"
    frame = draw_korean_text(
        frame, info_text, (frame.shape[1] - 300, 30), font, (0, 0, 0)
    )

    # test_accuracy가 있을 때만 표시
    if (
        "training_stats" in model_info
        and "test_accuracy" in model_info["training_stats"]
    ):
        info_text2 = f"정확도: {model_info['training_stats']['test_accuracy']*100:.1f}%"
        frame = draw_korean_text(
            frame, info_text2, (frame.shape[1] - 300, 60), font, (0, 0, 0)
        )

    cv2.imshow("수어 퀴즈", frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n"):
        # 'n' 키로 정답 라벨 넘기기
        quiz_index = (quiz_index + 1) % len(QUIZ_LABELS)
        current_label = QUIZ_LABELS[quiz_index]
        feedback = ""
        feedback_timer = 0
        hold_counter = 0
        quiz_number += 1

cap.release()
cv2.destroyAllWindows()
holistic.close()

print(f"\n✅ 퀴즈 종료")
print(f"📊 사용된 모델: {model_info['model_type']}")
print(f"🎯 퀴즈 라벨 수: {len(QUIZ_LABELS)}")

# test_accuracy가 있을 때만 표시
if "training_stats" in model_info and "test_accuracy" in model_info["training_stats"]:
    print(f"📈 모델 정확도: {model_info['training_stats']['test_accuracy']*100:.1f}%")
else:
    print("📈 모델 정확도: 정보 없음")
