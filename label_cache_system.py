import os
import sys
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 설정
TARGET_SEQ_LENGTH = 30
AUGMENTATIONS_PER_VIDEO = 3
NONE_CLASS = "None"

# 캐시 디렉토리 설정
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 라벨별 캐시 파일 경로 생성 함수
def get_label_cache_path(label):
    """라벨별 캐시 파일 경로를 반환합니다."""
    safe_label = label.replace(" ", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_label}_data.pkl")

def save_label_cache(label, data):
    """라벨별 데이터를 캐시에 저장합니다."""
    cache_path = get_label_cache_path(label)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 {label} 라벨 데이터 캐시 저장: {cache_path}")

def load_label_cache(label):
    """라벨별 데이터를 캐시에서 로드합니다."""
    cache_path = get_label_cache_path(label)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"📂 {label} 라벨 데이터 캐시 로드: {cache_path}")
        return data
    return None

def extract_landmarks(video_path):
    """비디오에서 랜드마크를 추출합니다."""
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # MediaPipe 처리
        results = holistic.process(image)
        
        # 랜드마크 저장
        landmarks_list.append(results)
    
    cap.release()
    return landmarks_list

def improved_preprocess_landmarks(landmarks_list):
    """랜드마크를 전처리합니다."""
    if not landmarks_list:
        return None
    
    # 시퀀스 길이 정규화
    if len(landmarks_list) > TARGET_SEQ_LENGTH:
        # 더 긴 시퀀스는 균등하게 샘플링
        indices = np.linspace(0, len(landmarks_list) - 1, TARGET_SEQ_LENGTH, dtype=int)
        landmarks_list = [landmarks_list[i] for i in indices]
    elif len(landmarks_list) < TARGET_SEQ_LENGTH:
        # 더 짧은 시퀀스는 마지막 프레임으로 패딩
        last_frame = landmarks_list[-1]
        while len(landmarks_list) < TARGET_SEQ_LENGTH:
            landmarks_list.append(last_frame)
    
    # 랜드마크를 배열로 변환
    sequence = []
    for landmarks in landmarks_list:
        frame_features = []
        
        # Pose landmarks (33개)
        if landmarks.pose_landmarks:
            for lm in landmarks.pose_landmarks.landmark:
                frame_features.extend([lm.x, lm.y, lm.z])
        else:
            frame_features.extend([0, 0, 0] * 33)
        
        # Left hand landmarks (21개)
        if landmarks.left_hand_landmarks:
            for lm in landmarks.left_hand_landmarks.landmark:
                frame_features.extend([lm.x, lm.y, lm.z])
        else:
            frame_features.extend([0, 0, 0] * 21)
        
        # Right hand landmarks (21개)
        if landmarks.right_hand_landmarks:
            for lm in landmarks.right_hand_landmarks.landmark:
                frame_features.extend([lm.x, lm.y, lm.z])
        else:
            frame_features.extend([0, 0, 0] * 21)
        
        sequence.append(frame_features)
    
    return np.array(sequence)

def augment_sequence_improved(sequence, noise_level=0.05, scale_range=0.2, rotation_range=0.1):
    """시퀀스를 증강합니다."""
    augmented = sequence.copy()
    
    # 노이즈 추가
    noise = np.random.normal(0, noise_level, augmented.shape)
    augmented += noise
    
    # 스케일링
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    augmented *= scale_factor
    
    # 회전 (간단한 회전)
    angle = np.random.uniform(-rotation_range, rotation_range)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # x, y 좌표에만 회전 적용
    for i in range(augmented.shape[0]):
        for j in range(0, augmented.shape[1], 3):  # x, y, z 순서
            if j + 1 < augmented.shape[1]:
                xy = np.array([augmented[i, j], augmented[i, j + 1]])
                rotated_xy = rotation_matrix @ xy
                augmented[i, j] = rotated_xy[0]
                augmented[i, j + 1] = rotated_xy[1]
    
    return augmented

def extract_and_cache_label_data(file_mapping, label):
    """특정 라벨의 데이터를 추출하고 캐시에 저장합니다."""
    print(f"\n🔄 {label} 라벨 데이터 추출 중...")
    
    # 캐시 확인
    cached_data = load_label_cache(label)
    if cached_data:
        print(f"✅ {label} 라벨 캐시 데이터 사용: {len(cached_data)}개 샘플")
        return cached_data
    
    # 해당 라벨의 파일들만 필터링
    label_files = {filename: info for filename, info in file_mapping.items() 
                  if info['label'] == label}
    
    if not label_files:
        print(f"⚠️ {label} 라벨에 해당하는 파일이 없습니다.")
        return []
    
    label_data = []
    
    for filename, info in tqdm(label_files.items(), desc=f"{label} 데이터 추출"):
        file_path = info['path']
        
        try:
            landmarks = extract_landmarks(file_path)
            if not landmarks:
                print(f"⚠️ 랜드마크 추출 실패: {file_path}")
                continue

            processed_sequence = improved_preprocess_landmarks(landmarks)

            if processed_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                print(f"⚠️ 시퀀스 형태 오류: {processed_sequence.shape}")
                continue

            # 원본 데이터 추가
            label_data.append(processed_sequence)

            # 증강 데이터 추가
            for _ in range(AUGMENTATIONS_PER_VIDEO):
                try:
                    augmented = augment_sequence_improved(processed_sequence)
                    if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                        label_data.append(augmented)
                except Exception as e:
                    print(f"⚠️ 증강 중 오류: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ 파일 처리 중 오류: {filename}, 오류: {e}")
            continue
    
    print(f"✅ {label} 라벨 데이터 추출 완료: {len(label_data)}개 샘플")
    
    # 캐시에 저장
    save_label_cache(label, label_data)
    
    return label_data

def generate_none_class_data(file_mapping):
    """None 클래스 데이터를 생성하고 캐시에 저장합니다."""
    print(f"\n✨ '{NONE_CLASS}' 클래스 데이터 생성 중...")
    
    # 기존 캐시 확인
    cached_none_data = load_label_cache(NONE_CLASS)
    if cached_none_data:
        print(f"✅ {NONE_CLASS} 클래스 캐시 데이터 사용: {len(cached_none_data)}개 샘플")
        return cached_none_data
    
    none_samples = []
    source_videos = list(file_mapping.keys())

    for filename in tqdm(source_videos, desc="None 클래스 데이터 생성"):
        file_path = file_mapping[filename]['path']
        
        try:
            landmarks = extract_landmarks(file_path)
            if landmarks and len(landmarks) > 10:
                # 영상의 시작, 1/4, 1/2, 3/4, 끝 지점에서 프레임 추출
                frame_indices = [
                    0,
                    len(landmarks) // 4,
                    len(landmarks) // 2,
                    3 * len(landmarks) // 4,
                    -1,
                ]

                for idx in frame_indices:
                    static_landmarks = [landmarks[idx]] * TARGET_SEQ_LENGTH
                    static_sequence = improved_preprocess_landmarks(static_landmarks)

                    if static_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                        continue

                    # 정적 시퀀스 추가
                    none_samples.append(static_sequence)

                    # 미세한 움직임 추가 (노이즈)
                    for _ in range(3):
                        augmented = augment_sequence_improved(
                            static_sequence, noise_level=0.01
                        )
                        if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                            none_samples.append(augmented)

                # 느린 전환 데이터 생성
                start_frame_lm = landmarks[0]
                middle_frame_lm = landmarks[len(landmarks) // 2]

                transition_landmarks = []
                for i in range(TARGET_SEQ_LENGTH):
                    alpha = i / (TARGET_SEQ_LENGTH - 1)
                    interp_frame = {}
                    for key in ["pose", "left_hand", "right_hand"]:
                        if start_frame_lm.get(key) and middle_frame_lm.get(key):
                            interp_lm = []
                            start_lms = start_frame_lm[key].landmark
                            mid_lms = middle_frame_lm[key].landmark
                            for j in range(len(start_lms)):
                                new_x = (
                                    start_lms[j].x * (1 - alpha) + mid_lms[j].x * alpha
                                )
                                new_y = (
                                    start_lms[j].y * (1 - alpha) + mid_lms[j].y * alpha
                                )
                                new_z = (
                                    start_lms[j].z * (1 - alpha) + mid_lms[j].z * alpha
                                )
                                interp_lm.append(
                                    type(
                                        "obj",
                                        (object,),
                                        {"x": new_x, "y": new_y, "z": new_z},
                                    )
                                )
                            interp_frame[key] = type(
                                "obj", (object,), {"landmark": interp_lm}
                            )
                        else:
                            interp_frame[key] = None
                    transition_landmarks.append(interp_frame)

                transition_sequence = improved_preprocess_landmarks(
                    transition_landmarks
                )
                if transition_sequence.shape == (TARGET_SEQ_LENGTH, 675):
                    none_samples.append(transition_sequence)
        except Exception as e:
            print(f"⚠️ None 클래스 데이터 생성 중 오류: {filename}, 오류: {e}")
            continue

    print(f"✅ {NONE_CLASS} 클래스 데이터 생성 완료: {len(none_samples)}개 샘플")
    
    # 캐시에 저장
    save_label_cache(NONE_CLASS, none_samples)
    
    return none_samples

def get_action_index(label):
    """라벨의 인덱스를 반환합니다."""
    return ACTIONS.index(label)

def create_simple_model(input_shape, num_classes):
    """간단한 모델을 생성합니다."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def process_all_labels(file_mapping, actions):
    """모든 라벨의 데이터를 처리하고 캐시에 저장합니다."""
    print("🚀 라벨별 데이터 추출 및 캐싱 시작...")
    
    all_data = {}
    
    # 각 라벨별로 데이터 추출
    for label in actions:
        if label == NONE_CLASS:
            label_data = generate_none_class_data(file_mapping)
        else:
            label_data = extract_and_cache_label_data(file_mapping, label)
        
        all_data[label] = label_data
    
    return all_data

def combine_all_data(all_data, actions):
    """모든 라벨의 데이터를 결합합니다."""
    print("\n🔗 모든 라벨 데이터 결합 중...")
    
    X = []
    y = []
    
    for label in actions:
        if label in all_data and all_data[label]:
            label_data = all_data[label]
            label_index = get_action_index(label)
            
            X.extend(label_data)
            y.extend([label_index] * len(label_data))
            
            print(f"✅ {label}: {len(label_data)}개 샘플 추가")
    
    print(f"\n📊 최종 데이터 통계:")
    print(f"총 샘플 수: {len(X)}")
    
    # 클래스별 샘플 수 확인
    unique, counts = np.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        if 0 <= class_idx < len(actions):
            print(f"클래스 {class_idx} ({actions[class_idx]}): {count}개")
        else:
            print(f"클래스 {class_idx} (Unknown): {count}개")
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 이 파일은 라벨별 캐싱 시스템을 제공하는 모듈입니다.
    # fix_training_data.py에서 import하여 사용합니다.
    print("📦 라벨별 캐싱 시스템 모듈 로드 완료") 