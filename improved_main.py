import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
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
TARGET_SEQ_LENGTH = 30  # 정규화된 시퀀스 길이 (리포트 권장사항)
AUGMENTATIONS_PER_VIDEO = 9
DATA_CACHE_PATH = 'improved_preprocessed_data.npz'
MODEL_SAVE_PATH = 'improved_transformer_model.keras'
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

def create_transformer_model(input_shape, num_classes):
    """Transformer 기반 모델을 생성합니다."""
    inputs = Input(shape=input_shape)
    
    # 1D CNN으로 공간 패턴 추출
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Transformer Encoder Layer
    def transformer_encoder_block(x, num_heads=8, ff_dim=256, dropout=0.1):
        # Multi-Head Self-Attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=128, dropout=dropout
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward Network
        ffn_output = Dense(ff_dim, activation='relu')(x)
        ffn_output = Dense(128)(ffn_output)
        ffn_output = Dropout(dropout)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        return x
    
    # Transformer 블록 적용
    x = transformer_encoder_block(x)
    x = transformer_encoder_block(x)
    
    # BiLSTM으로 시간적 패턴 분석
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32))(x)
    
    # 분류 레이어
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_hybrid_model(input_shape, num_classes):
    """CNN + LSTM 하이브리드 모델을 생성합니다."""
    model = Sequential([
        # 1D CNN으로 공간 패턴 추출
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        
        # BiLSTM으로 시간적 패턴 분석
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        
        # 분류 레이어
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def augment_sequence_improved(sequence, noise_level=0.01, scale_range=0.1, rotation_range=0.05):
    """개선된 시퀀스 증강 함수."""
    augmented_sequence = sequence.copy()
    
    # 1. 노이즈 추가
    noise = np.random.normal(0, noise_level, augmented_sequence.shape)
    augmented_sequence += noise
    
    # 2. 크기 조절
    scale_factor = 1.0 + np.random.uniform(-scale_range, scale_range)
    augmented_sequence *= scale_factor
    
    # 3. 시간적 변형 (프레임 순서 약간 변경)
    if len(augmented_sequence) > 3:
        # 랜덤하게 몇 개 프레임을 교환
        for _ in range(2):
            i, j = np.random.choice(len(augmented_sequence), 2, replace=False)
            augmented_sequence[i], augmented_sequence[j] = augmented_sequence[j].copy(), augmented_sequence[i].copy()
    
    return augmented_sequence

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

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # --- 1. 데이터 로딩 또는 추출 ---
    if os.path.exists(DATA_CACHE_PATH):
        print(f"💾 캐시에서 개선된 전처리 데이터 로딩: {DATA_CACHE_PATH}")
        cached_data = np.load(DATA_CACHE_PATH)
        X_padded = cached_data['X']
        y_one_hot = cached_data['y']
    else:
        print("✨ 개선된 데이터 캐시 없음. 비디오에서 랜드마크 추출 및 증강을 시작합니다.")
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
            
            # 시퀀스 길이 확인
            if processed_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                print(f"⚠️ 시퀀스 형태 오류: {processed_sequence.shape}, 예상: ({TARGET_SEQ_LENGTH}, 675)")
                continue
            
            # 원본 데이터 추가
            X.append(processed_sequence)
            y.append(ACTIONS.index(label))

            # 증강 데이터 추가
            for _ in range(AUGMENTATIONS_PER_VIDEO):
                try:
                    augmented = augment_sequence_improved(processed_sequence)
                    # 증강 후에도 형태 확인
                    if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                        X.append(augmented)
                        y.append(ACTIONS.index(label))
                except Exception as e:
                    print(f"⚠️ 증강 중 오류 발생: {e}")
                    continue

        # 'None' 클래스 데이터 생성
        print("\n✨ 'None' 클래스 데이터 생성 중...")
        base_video_path = os.path.join(VIDEO_ROOT, f"{list(label_dict.keys())[0].split('.')[0]}.avi")
        if os.path.exists(base_video_path):
            landmarks = extract_landmarks(base_video_path)
            if landmarks:
                first_frame_landmarks = improved_preprocess_landmarks([landmarks[0]])
                if first_frame_landmarks.shape == (TARGET_SEQ_LENGTH, 675):
                    still_sequence = np.tile(first_frame_landmarks, (TARGET_SEQ_LENGTH, 1))
                    
                    none_label_index = ACTIONS.index("None")
                    for _ in range(10 * len(ACTIONS)):
                        try:
                            augmented = augment_sequence_improved(still_sequence)
                            if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                                X.append(augmented)
                                y.append(none_label_index)
                        except Exception as e:
                            print(f"⚠️ None 클래스 증강 중 오류: {e}")
                            continue

        # 최종 데이터 형태 확인
        if not X:
            print("❌ 처리된 데이터가 없습니다.")
            print("데이터 경로와 파일들을 확인해주세요.")
            sys.exit(1)
        
        print(f"📊 처리된 시퀀스 수: {len(X)}")
        print(f"📊 첫 번째 시퀀스 형태: {X[0].shape}")
        
        # 모든 시퀀스가 같은 형태인지 확인
        shapes = [seq.shape for seq in X]
        if len(set(shapes)) > 1:
            print(f"⚠️ 시퀀스 형태가 일정하지 않습니다: {set(shapes)}")
            # 가장 일반적인 형태로 필터링
            most_common_shape = max(set(shapes), key=shapes.count)
            X = [seq for seq in X if seq.shape == most_common_shape]
            print(f"📊 필터링 후 시퀀스 수: {len(X)}")

        X_padded = np.array(X)
        y_one_hot = to_categorical(y, num_classes=len(ACTIONS))
        
        print(f"💾 개선된 데이터 캐시 저장: {DATA_CACHE_PATH}")
        np.savez(DATA_CACHE_PATH, X=X_padded, y=y_one_hot)

    print(f"✅ 개선된 데이터 준비 완료: {X_padded.shape[0]}개 샘플, 시퀀스 길이: {X_padded.shape[1]}")

    # --- 2. 모델 학습 또는 로딩 ---
    if X_padded.shape[0] < 2:
        print("⚠️ 데이터가 부족하여 모델을 학습할 수 없습니다.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
        )

        if os.path.exists(MODEL_SAVE_PATH):
            print(f"🧠 저장된 개선 모델 로딩: {MODEL_SAVE_PATH}")
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        else:
            print("🏋️‍♀️ 저장된 모델 없음. 개선된 Transformer 모델 학습을 시작합니다.")
            
            # 모델 선택 (Transformer 또는 하이브리드)
            use_transformer = True  # True: Transformer, False: 하이브리드
            
            if use_transformer:
                model = create_transformer_model(
                    input_shape=(X_padded.shape[1], X_padded.shape[2]), 
                    num_classes=len(ACTIONS)
                )
            else:
                model = create_hybrid_model(
                    input_shape=(X_padded.shape[1], X_padded.shape[2]), 
                    num_classes=len(ACTIONS)
                )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            print("\n--- 개선된 모델 학습 시작 ---")
            model.summary()

            # Early stopping과 체크포인트 추가
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]

            history = model.fit(
                X_train, y_train, 
                epochs=100, 
                batch_size=16, 
                validation_data=(X_test, y_test),
                callbacks=callbacks
            )
            
            print(f"🧠 학습된 개선 모델 저장: {MODEL_SAVE_PATH}")
            model.save(MODEL_SAVE_PATH)

        # --- 3. 모델 평가 ---
        print("\n--- 개선된 모델 평가 ---")
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"🚀 개선된 테스트 정확도: {accuracy * 100:.2f}%")

        # 예측 결과 확인
        print("\n--- Test Sample Predictions ---")
        y_pred_prob = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_prob, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        correct_predictions = 0
        for i, (pred_class, true_class) in enumerate(zip(y_pred_classes, y_true_classes)):
            pred_label = ACTIONS[pred_class]
            actual_label = ACTIONS[true_class]
            result = "✅" if pred_class == true_class else "❌"
            confidence = y_pred_prob[i][pred_class]
            print(f"Sample {i+1}: Prediction={pred_label} (Confidence: {confidence:.2f}), Actual={actual_label} {result}")
            if pred_class == true_class:
                correct_predictions += 1
        
        print(f"\n📊 정확도 요약: {correct_predictions}/{len(y_test)} ({correct_predictions/len(y_test)*100:.2f}%)")

    holistic.close() 