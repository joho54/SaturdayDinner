#!/usr/bin/env python3
"""
수어 인식 모델 학습 파이프라인
사용법: python3 model_pipe.py spec.json
"""

import os
import sys
import json
import csv
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    LayerNormalization,
    MultiHeadAttention,
    Add,
)
from tensorflow.keras.utils import to_categorical
from scipy.interpolate import interp1d
import datetime
import random

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# 경로 설정 (변수로 생성하여 경로 변경 가능)
VIDEO_ROOT1 = "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/0001~3000(영상)"
VIDEO_ROOT2 = "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/3001~6000(영상)"
VIDEO_ROOT3 = "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/6001~8280(영상)"
VIDEO_ROOT4 = "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/8381~9000(영상)"
VIDEO_ROOT5 = "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/9001~9600(영상)"

# CSV 파일 경로 (변수로 생성)
LABEL_CSV_PATH = "./labels.csv"

def load_spec(spec_path):
    """spec.json 파일을 로드합니다."""
    try:
        with open(spec_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: {spec_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ 오류: {spec_path} 파일의 JSON 형식이 올바르지 않습니다.")
        sys.exit(1)

def load_label_csv(csv_path):
    """label.csv 파일을 로드하여 딕셔너리로 변환합니다."""
    label_dict = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_dict[row['filename']] = row['label']
        print(f"✅ {len(label_dict)}개의 라벨 데이터를 로드했습니다.")
        return label_dict
    except FileNotFoundError:
        print(f"❌ 오류: {csv_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)

def filter_labels_by_spec(label_dict, target_labels):
    """spec.json에 명시된 라벨과 일치하는 영상:라벨 쌍만 필터링합니다."""
    filtered_dict = {}
    for filename, label in label_dict.items():
        if label in target_labels:
            filtered_dict[filename] = label
    
    print(f"✅ 필터링 결과: {len(filtered_dict)}개의 영상이 선택되었습니다.")
    for label in target_labels:
        count = sum(1 for l in filtered_dict.values() if l == label)
        print(f"   - {label}: {count}개")
    
    return filtered_dict

def get_video_root_and_path(filename):
    """파일명에서 번호를 추출해 올바른 VIDEO_ROOT 경로와 실제 파일 경로를 반환합니다."""
    try:
        num_str = filename.split("_")[-1].split(".")[0]
        num = int(num_str)
    except Exception:
        return None

    if 1 <= num <= 3000:
        root = VIDEO_ROOT1
    elif 3001 <= num <= 6000:
        root = VIDEO_ROOT2
    elif 6001 <= num <= 8280:
        root = VIDEO_ROOT3
    elif 8381 <= num <= 9000:
        root = VIDEO_ROOT4
    elif 9001 <= num <= 9600:
        root = VIDEO_ROOT5
    else:
        return None

    base_name = "_".join(filename.split("_")[:-1]) + f"_{num_str}"
    for ext in [".MOV", ".MTS", ".AVI", ".MP4"]:
        # 대문자와 소문자 모두 시도
        for case_ext in [ext, ext.lower()]:
            candidate = os.path.join(root, base_name + case_ext)
            if os.path.exists(candidate):
                return candidate
    return None

def normalize_sequence_length(sequence, target_length=30):
    """시퀀스 길이를 정규화합니다."""
    current_length = len(sequence)

    if current_length == target_length:
        return sequence

    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)

    normalized_sequence = []
    for i in range(sequence.shape[1]):
        f = interp1d(
            x_old,
            sequence[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        normalized_sequence.append(f(x_new))

    return np.array(normalized_sequence).T

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

def extract_landmarks(video_path):
    """비디오에서 랜드마크를 추출합니다."""
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        frame_data = {
            "pose": results.pose_landmarks,
            "left_hand": results.left_hand_landmarks,
            "right_hand": results.right_hand_landmarks,
        }
        landmarks_list.append(frame_data)

    cap.release()
    return landmarks_list

def extract_dynamic_features(sequence):
    """속도와 가속도 특징을 추출합니다."""
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    dynamic_features = np.concatenate([sequence, velocity, acceleration], axis=1)
    return dynamic_features

def improved_preprocess_landmarks(landmarks_list, target_seq_length=30):
    """개선된 랜드마크 전처리 함수."""
    if not landmarks_list:
        return np.zeros((target_seq_length, 675))

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
        return np.zeros((target_seq_length, 675))

    sequence = np.array(processed_frames)

    if len(sequence) > 0:
        try:
            sequence = normalize_sequence_length(sequence, target_seq_length)
            sequence = extract_dynamic_features(sequence)

            # 정규화 개선: 더 강한 정규화
            sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)

            return sequence
        except Exception as e:
            print(f"⚠️ 시퀀스 처리 중 오류 발생: {e}")
            return np.zeros((target_seq_length, 675))

    return np.zeros((target_seq_length, 675))

def create_model(input_shape, num_classes):
    """모델을 생성합니다."""
    inputs = Input(shape=input_shape)
    
    # 1D Convolutional layers
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    
    # LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def augment_sequence(sequence, noise_level=0.05, scale_range=0.2):
    """시퀀스를 증강합니다."""
    augmented = sequence.copy()
    
    # 노이즈 추가
    noise = np.random.normal(0, noise_level, augmented.shape)
    augmented += noise
    
    # 스케일링
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    augmented *= scale_factor
    
    return augmented

def get_model_number():
    """현재 시간을 기반으로 모델 번호를 생성합니다."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def generate_cache_filename(labels, target_seq_length, augmentations_per_video):
    """캐시 파일명을 생성합니다."""
    labels_str = "_".join(sorted(labels))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"landmarks_cache_{labels_str}_seq{target_seq_length}_aug{augmentations_per_video}_{timestamp}.npz"

def find_latest_cache(labels, target_seq_length, augmentations_per_video):
    """가장 최근의 캐시 파일을 찾습니다."""
    npz_dir = "npzs"
    if not os.path.exists(npz_dir):
        return None
    
    labels_str = "_".join(sorted(labels))
    pattern = f"landmarks_cache_{labels_str}_seq{target_seq_length}_aug{augmentations_per_video}_*.npz"
    
    import glob
    cache_files = glob.glob(os.path.join(npz_dir, pattern))
    
    if not cache_files:
        return None
    
    # 가장 최근 파일 반환
    latest_cache = max(cache_files, key=os.path.getctime)
    return latest_cache

def load_cached_landmarks(cache_path):
    """캐시된 랜드마크 데이터를 로드합니다."""
    try:
        if os.path.exists(cache_path):
            file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"📂 캐시된 데이터 로드 중: {cache_path} ({file_size_mb:.1f}MB)")
            data = np.load(cache_path, allow_pickle=True)
            X = data['X']
            y = data['y']
            filenames = data['filenames']
            
            # 메타데이터 출력
            if 'metadata' in data:
                metadata = data['metadata'].item()
                print(f"📊 캐시 메타데이터:")
                print(f"   - 라벨: {metadata.get('labels', [])}")
                print(f"   - 시퀀스 길이: {metadata.get('sequence_length', 'N/A')}")
                print(f"   - 증강 수: {metadata.get('augmentations_per_video', 'N/A')}")
                print(f"   - 생성일: {metadata.get('created_at', 'N/A')}")
                print(f"   - 총 샘플: {metadata.get('total_samples', 'N/A')}")
            
            print(f"✅ 캐시 로드 완료: {len(X)}개 샘플")
            return X, y, filenames
        return None, None, None
    except Exception as e:
        print(f"⚠️ 캐시 로드 실패: {e}")
        return None, None, None

def save_cached_landmarks(X, y, filenames, cache_path, spec, target_seq_length, augmentations_per_video):
    """랜드마크 데이터를 캐시에 저장합니다."""
    try:
        print(f"💾 캐시 저장 중: {cache_path}")
        
        # 메타데이터 추가
        metadata = {
            'labels': spec.get('labels', []),
            'sequence_length': target_seq_length,
            'augmentations_per_video': augmentations_per_video,
            'created_at': datetime.datetime.now().isoformat(),
            'total_samples': len(X),
            'model_name': spec.get('model_name', 'custom_model')
        }
        
        np.savez_compressed(
            cache_path, 
            X=X, 
            y=y, 
            filenames=filenames,
            metadata=metadata
        )
        print(f"✅ 캐시 저장 완료: {len(X)}개 샘플")
    except Exception as e:
        print(f"⚠️ 캐시 저장 실패: {e}")

def save_model_info(model, model_path, spec, model_number):
    """모델 정보를 JSON 파일로 저장합니다."""
    model_info = {
        "name": spec.get("model_name", "custom_model"),
        "type": "Functional",
        "total_params": model.count_params(),
        "trainable_params": model.count_params(),
        "input_shape": [None] + list(model.input_shape[1:]),
        "output_shape": [None] + list(model.output_shape[1:]),
        "layers_count": len(model.layers),
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "labels": spec.get("labels", []),
        "model_number": model_number,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    # models 디렉토리에 저장
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    info_path = os.path.join(models_dir, f"model-info-{model_number}.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 모델 정보 저장: {info_path}")

def main():
    if len(sys.argv) < 2:
        print("사용법: python3 model_pipe.py spec.json [--no-cache]")
        print("  --no-cache: 캐시를 무시하고 새로 추출")
        sys.exit(1)
    
    spec_path = sys.argv[1]
    use_cache = "--no-cache" not in sys.argv
    
    print(f"🚀 모델 학습 파이프라인 시작")
    print(f"📋 명세 파일: {spec_path}")
    print(f"💾 캐시 사용: {'예' if use_cache else '아니오'}")
    
    # 1. spec.json 로드
    spec = load_spec(spec_path)
    target_labels = spec.get("labels", [])
    print(f"🎯 학습할 라벨: {target_labels}")
    
    # 2. label.csv 로드
    label_dict = load_label_csv(LABEL_CSV_PATH)
    
    # 3. spec에 맞는 라벨 필터링
    filtered_dict = filter_labels_by_spec(label_dict, target_labels)
    
    if len(filtered_dict) == 0:
        print("❌ 오류: 필터링된 데이터가 없습니다.")
        sys.exit(1)
    
    # 4. 설정값 가져오기
    training_config = spec.get("training_config", {})
    model_config = spec.get("model_config", {})
    
    TARGET_SEQ_LENGTH = training_config.get("sequence_length", 30)
    AUGMENTATIONS_PER_VIDEO = training_config.get("augmentations_per_video", 20)
    TEST_SPLIT = training_config.get("test_split", 0.2)
    RANDOM_STATE = training_config.get("random_state", 42)
    
    LEARNING_RATE = model_config.get("learning_rate", 0.001)
    BATCH_SIZE = model_config.get("batch_size", 32)
    EPOCHS = model_config.get("epochs", 100)
    EARLY_STOPPING_PATIENCE = model_config.get("early_stopping_patience", 10)
    
    # 5. 데이터 추출 및 전처리
    print("\n📊 데이터 추출 및 전처리 중...")
    
    # 캐시 파일 경로 설정
    npz_dir = "npzs"
    os.makedirs(npz_dir, exist_ok=True)
    
    # 최신 캐시 파일 찾기 (캐시 사용이 활성화된 경우에만)
    latest_cache_path = None
    if use_cache:
        latest_cache_path = find_latest_cache(target_labels, TARGET_SEQ_LENGTH, AUGMENTATIONS_PER_VIDEO)
        if latest_cache_path:
            print(f"📂 최신 캐시 파일 발견: {os.path.basename(latest_cache_path)}")
    
    # 캐시된 데이터 확인
    X, y, cached_filenames = load_cached_landmarks(latest_cache_path) if latest_cache_path else (None, None, None)
    
    if X is not None and use_cache:
        # 캐시된 데이터가 있으면 사용
        print(f"✅ 캐시된 데이터 사용: {len(X)}개 샘플")
        y_one_hot = to_categorical(y, num_classes=len(target_labels))
    else:
        # 캐시된 데이터가 없거나 캐시 사용이 비활성화된 경우 새로 추출
        if not use_cache:
            print("🔄 캐시 무효화 옵션이 활성화되어 새로 추출합니다...")
        else:
            print("🔄 캐시된 데이터가 없어 새로 추출합니다...")
            
        cache_filename = generate_cache_filename(target_labels, TARGET_SEQ_LENGTH, AUGMENTATIONS_PER_VIDEO)
        cache_path = os.path.join(npz_dir, cache_filename)
        
        X = []
        y = []
        filenames = []
        
        for filename, label in tqdm(filtered_dict.items(), desc="데이터 추출"):
            actual_path = get_video_root_and_path(filename)
            if actual_path is None or not os.path.exists(actual_path):
                print(f"⚠️ 파일 없음: {filename}")
                continue

            landmarks = extract_landmarks(actual_path)
            if not landmarks:
                print(f"⚠️ 랜드마크 추출 실패: {filename}")
                continue

            processed_sequence = improved_preprocess_landmarks(landmarks, TARGET_SEQ_LENGTH)
            if processed_sequence is None or processed_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                print(f"⚠️ 시퀀스 형태 오류: {filename}")
                continue

            # 원본 데이터 추가
            X.append(processed_sequence)
            y.append(target_labels.index(label))
            filenames.append(filename)

            # 증강 데이터 추가
            for _ in range(AUGMENTATIONS_PER_VIDEO):
                try:
                    augmented = augment_sequence(processed_sequence)
                    if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                        X.append(augmented)
                        y.append(target_labels.index(label))
                        filenames.append(f"{filename}_aug_{_}")
                except Exception as e:
                    print(f"⚠️ 증강 중 오류: {e}")
                    continue
        
        if len(X) == 0:
            print("❌ 오류: 처리된 데이터가 없습니다.")
            sys.exit(1)
        
        # 캐시에 저장
        X = np.array(X)
        y = np.array(y)
        filenames = np.array(filenames)
        save_cached_landmarks(X, y, filenames, cache_path, spec, TARGET_SEQ_LENGTH, AUGMENTATIONS_PER_VIDEO)
        
        y_one_hot = to_categorical(y, num_classes=len(target_labels))
    
    # 6. 데이터 준비
    print(f"📊 최종 데이터 통계:")
    print(f"총 샘플 수: {len(X)}")
    print(f"입력 형태: {X.shape}")
    print(f"출력 형태: {y_one_hot.shape}")
    
    # 클래스별 샘플 수 확인
    unique, counts = np.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        if 0 <= class_idx < len(target_labels):
            print(f"클래스 {class_idx} ({target_labels[class_idx]}): {count}개")
    
    # 7. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y_one_hot
    )
    
    # 8. 모델 생성
    print("\n🏗️ 모델 생성 중...")
    model = create_model(
        input_shape=(X.shape[1], X.shape[2]), 
        num_classes=len(target_labels)
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    print("\n--- 모델 구조 ---")
    model.summary()
    
    # 9. 모델 학습
    print("\n🏋️‍♀️ 모델 학습 시작...")
    
    # Early stopping 콜백
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=0.0001,  # 최소 개선 임계값 (0.0001 = 0.01%)
        restore_best_weights=True,
        verbose=1  # Early stopping 발생 시 메시지 출력
    )
    
    # 체크포인트 콜백
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model-epoch-{epoch:02d}.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_loss'
    )
    
    # 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint_callback],
        verbose=1
    )
    
    # 10. 모델 평가
    print("\n📈 모델 평가...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 손실: {test_loss:.4f}")
    
    # 11. 모델 저장
    model_number = get_model_number()
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"model_{spec.get('model_name', 'custom')}_{model_number}.keras"
    model_path = os.path.join(models_dir, model_filename)
    
    model.save(model_path)
    print(f"✅ 모델 저장: {model_path}")
    
    # 12. 모델 정보 저장
    save_model_info(model, model_path, spec, model_number)
    
    print(f"\n🎉 모델 학습 완료!")
    print(f"📁 모델 파일: {model_path}")
    print(f"📊 최종 정확도: {test_accuracy:.4f}")

if __name__ == "__main__":
    main() 