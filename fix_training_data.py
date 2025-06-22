import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional, 
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from scipy.interpolate import interp1d
import sys

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# 경로 및 상수 설정
VIDEO_ROOT = "/Volumes/Sub_Storage/수어 데이터셋/0001~3000(영상)"
TARGET_SEQ_LENGTH = 30
AUGMENTATIONS_PER_VIDEO = 20  # 증강 횟수 증가
DATA_CACHE_PATH = 'fixed_preprocessed_data.npz'
MODEL_SAVE_PATH = 'fixed_transformer_model.keras'
ACTIONS = ["Fire", "Toilet", "None"]

label_dict = {
    # Fire
    "KETI_SL_0000000419.MOV": "Fire",
    "KETI_SL_0000000838.MTS": "Fire",
    "KETI_SL_0000001255.MTS": "Fire",
    "KETI_SL_0000001674.MTS": "Fire",
    "KETI_SL_0000002032.MOV": "Fire",
    "KETI_SL_0000002451.MP4": "Fire",
    "KETI_SL_0000002932.MOV": "Fire",
    # Toilet
    "KETI_SL_0000000418.MOV": "Toilet",
    "KETI_SL_0000000837.MTS": "Toilet",
    "KETI_SL_0000001254.MTS": "Toilet",
    "KETI_SL_0000001673.MTS": "Toilet",
    "KETI_SL_0000002031.MOV": "Toilet",
    "KETI_SL_0000002450.MP4": "Toilet",
    "KETI_SL_0000002931.MOV": "Toilet"
}

def normalize_sequence_length(sequence, target_length=30):
    """시퀀스 길이를 정규화합니다."""
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    normalized_sequence = []
    for i in range(sequence.shape[1]):
        f = interp1d(x_old, sequence[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        normalized_sequence.append(f(x_new))
    
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
        return np.zeros((TARGET_SEQ_LENGTH, 675))
    
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
        return np.zeros((TARGET_SEQ_LENGTH, 675))
    
    sequence = np.array(processed_frames)
    
    if len(sequence) > 0:
        try:
            sequence = normalize_sequence_length(sequence, TARGET_SEQ_LENGTH)
            sequence = extract_dynamic_features(sequence)
            
            # 정규화 개선: 더 강한 정규화
            sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)
            
            return sequence
        except Exception as e:
            print(f"⚠️ 시퀀스 처리 중 오류 발생: {e}")
            return np.zeros((TARGET_SEQ_LENGTH, 675))
    
    return np.zeros((TARGET_SEQ_LENGTH, 675))

def create_simple_model(input_shape, num_classes):
    """간단하고 효과적인 모델을 생성합니다."""
    inputs = Input(shape=input_shape)
    
    # 1D CNN
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # LSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def augment_sequence_improved(sequence, noise_level=0.05, scale_range=0.2, rotation_range=0.1):
    """개선된 시퀀스 증강."""
    augmented = sequence.copy()
    
    # 노이즈 추가
    noise = np.random.normal(0, noise_level, augmented.shape)
    augmented += noise
    
    # 스케일링
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    augmented *= scale_factor
    
    # 시간축에서의 회전 (시프트)
    shift = np.random.randint(-3, 4)
    if shift > 0:
        augmented = np.roll(augmented, shift, axis=0)
    elif shift < 0:
        augmented = np.roll(augmented, shift, axis=0)
    
    return augmented

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

if __name__ == "__main__":
    print("🔧 학습 데이터 문제 해결 및 모델 재학습 시작")
    
    # 데이터 추출
    X = []
    y = []

    for filename, label in tqdm(label_dict.items(), desc="데이터 추출"):
        file_id = filename.split(".")[0]
        actual_path = os.path.join(VIDEO_ROOT, f"{file_id}.avi")
        
        if not os.path.exists(actual_path):
            print(f"⚠️ 파일 없음: {actual_path}")
            continue
        
        landmarks = extract_landmarks(actual_path)
        if not landmarks:
            print(f"⚠️ 랜드마크 추출 실패: {actual_path}")
            continue
            
        processed_sequence = improved_preprocess_landmarks(landmarks)
        
        if processed_sequence.shape != (TARGET_SEQ_LENGTH, 675):
            print(f"⚠️ 시퀀스 형태 오류: {processed_sequence.shape}")
            continue
        
        # 원본 데이터 추가
        X.append(processed_sequence)
        y.append(ACTIONS.index(label))

        # 더 많은 증강 데이터 추가
        for _ in range(AUGMENTATIONS_PER_VIDEO):
            try:
                augmented = augment_sequence_improved(processed_sequence)
                if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                    X.append(augmented)
                    y.append(ACTIONS.index(label))
            except Exception as e:
                print(f"⚠️ 증강 중 오류: {e}")
                continue

    # 'None' 클래스 데이터 생성 (더 다양하게)
    print("\n✨ 'None' 클래스 데이터 대폭 강화 중...")
    none_samples = []
    
    # 전략 1: 더 많은 비디오에서, 더 다양한 프레임을 소스로 사용
    source_videos = list(label_dict.keys())
    
    for filename in source_videos:
        file_id = filename.split(".")[0]
        actual_path = os.path.join(VIDEO_ROOT, f"{file_id}.avi")
        
        if os.path.exists(actual_path):
            landmarks = extract_landmarks(actual_path)
            if landmarks and len(landmarks) > 10: # 충분한 길이의 영상만 사용
                # 영상의 시작, 1/4, 1/2, 3/4, 끝 지점에서 프레임 추출
                frame_indices = [0, len(landmarks)//4, len(landmarks)//2, 3*len(landmarks)//4, -1]
                
                for idx in frame_indices:
                    static_landmarks = [landmarks[idx]] * TARGET_SEQ_LENGTH
                    static_sequence = improved_preprocess_landmarks(static_landmarks)
                    
                    if static_sequence.shape != (TARGET_SEQ_LENGTH, 675): continue
                    
                    # 전략 1-1: 정적 시퀀스 자체를 추가
                    none_samples.append(static_sequence)

                    # 전략 1-2: 미세한 움직임 추가 (노이즈)
                    for _ in range(3):
                        augmented = augment_sequence_improved(static_sequence, noise_level=0.01)
                        if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                            none_samples.append(augmented)

                # 전략 2: 느린 전환 데이터 생성 (두 프레임 보간)
                start_frame_lm = landmarks[0]
                middle_frame_lm = landmarks[len(landmarks)//2]
                
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
                                new_x = start_lms[j].x * (1-alpha) + mid_lms[j].x * alpha
                                new_y = start_lms[j].y * (1-alpha) + mid_lms[j].y * alpha
                                new_z = start_lms[j].z * (1-alpha) + mid_lms[j].z * alpha
                                # MediaPipe Landmark-like object
                                interp_lm.append(type('obj', (object,), {'x': new_x, 'y': new_y, 'z': new_z}))
                            # MediaPipe LandmarkList-like object
                            interp_frame[key] = type('obj', (object,), {'landmark': interp_lm})
                        else:
                            interp_frame[key] = None
                    transition_landmarks.append(interp_frame)

                transition_sequence = improved_preprocess_landmarks(transition_landmarks)
                if transition_sequence.shape == (TARGET_SEQ_LENGTH, 675):
                    none_samples.append(transition_sequence)

    # 전략 3: 완전한 정지(zero) 데이터 추가
    for _ in range(len(source_videos) * 5): # 다른 클래스와 수량 맞추기
        none_samples.append(np.zeros((TARGET_SEQ_LENGTH, 675)))

    # None 클래스 데이터 추가
    none_label_index = ACTIONS.index("None")
    for sample in none_samples:
        X.append(sample)
        y.append(none_label_index)

    print(f"📊 최종 데이터 통계:")
    print(f"총 샘플 수: {len(X)}")
    
    # 클래스별 샘플 수 확인
    unique, counts = np.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"클래스 {class_idx} ({ACTIONS[class_idx]}): {count}개")

    X_padded = np.array(X)
    y_one_hot = to_categorical(y, num_classes=len(ACTIONS))
    
    # 데이터 저장
    print(f"💾 수정된 데이터 저장: {DATA_CACHE_PATH}")
    np.savez(DATA_CACHE_PATH, X=X_padded, y=y_one_hot)

    # 모델 학습
    print("\n🏋️‍♀️ 간단한 모델 학습 시작")
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
    )

    model = create_simple_model(
        input_shape=(X_padded.shape[1], X_padded.shape[2]), 
        num_classes=len(ACTIONS)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- 모델 구조 ---")
    model.summary()

    # 학습
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train, 
        epochs=200, 
        batch_size=8, 
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"🧠 수정된 모델 저장: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    # 평가
    print("\n--- 모델 평가 ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"🚀 테스트 정확도: {accuracy * 100:.2f}%")

    # 예측 결과 확인
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\n--- 클래스별 정확도 ---")
    for i in range(len(ACTIONS)):
        class_mask = y_true_classes == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(y_pred_classes[class_mask] == y_true_classes[class_mask])
            print(f"{ACTIONS[i]}: {class_accuracy:.4f}")

    holistic.close() 