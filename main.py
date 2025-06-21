import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# 경로 및 상수 설정
VIDEO_ROOT = "/Volumes/Sub_Storage/수어 데이터셋/0001~3000(영상)"
MAX_SEQ_LENGTH = 100  # 최대 프레임 길이 (패딩에 사용)
AUGMENTATIONS_PER_VIDEO = 9 # 원본 1개당 생성할 증강 데이터 수
DATA_CACHE_PATH = 'preprocessed_data_multiclass.npz'
MODEL_SAVE_PATH = 'lstm_model_multiclass.keras'
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

def augment_sequence(sequence, noise_level=0.005, scale_range=0.05):
    """랜드마크 시퀀스에 노이즈 추가 및 크기 조절을 적용하여 증강합니다."""
    augmented_sequence = sequence.copy()

    # 1. 노이즈 추가
    noise = np.random.normal(0, noise_level, augmented_sequence.shape)
    augmented_sequence += noise

    # 2. 크기 조절
    scale_factor = 1.0 + np.random.uniform(-scale_range, scale_range)
    augmented_sequence *= scale_factor

    return augmented_sequence

def extract_landmarks(video_path):
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

# --- 1. 데이터 로딩 또는 추출 ---
if os.path.exists(DATA_CACHE_PATH):
    print(f"💾 캐시에서 전처리된 데이터 로딩: {DATA_CACHE_PATH}")
    cached_data = np.load(DATA_CACHE_PATH)
    X_padded = cached_data['X']
    y_one_hot = cached_data['y']
else:
    print("✨ 데이터 캐시 없음. 비디오에서 랜드마크 추출 및 증강을 시작합니다.")
    X = []
    y = []

    for filename, label in tqdm(label_dict.items()):
        file_id = filename.split(".")[0]
        actual_path = os.path.join(VIDEO_ROOT, f"{file_id}.avi")
        
        if not os.path.exists(actual_path):
            print(f"⚠️ 파일 없음: {actual_path}")
            continue
        
        landmarks = extract_landmarks(actual_path)
        if not landmarks:
            continue
            
        processed_sequence = preprocess_landmarks(landmarks)
        
        # 원본 데이터 추가
        X.append(processed_sequence)
        y.append(ACTIONS.index(label))

        # 증강 데이터 추가
        for _ in range(AUGMENTATIONS_PER_VIDEO):
            augmented = augment_sequence(processed_sequence)
            X.append(augmented)
            y.append(ACTIONS.index(label))

    # 2. 'None' 데이터 인공적으로 생성
    print("\n✨ 'None' 클래스 데이터 생성 중...")
    # '화재' 첫번째 영상을 기반으로 '가만히 있는' 데이터 생성
    base_video_path = os.path.join(VIDEO_ROOT, f"{list(label_dict.keys())[0].split('.')[0]}.avi")
    if os.path.exists(base_video_path):
        landmarks = extract_landmarks(base_video_path)
        if landmarks:
            # 첫 프레임만 사용
            first_frame_landmarks = preprocess_landmarks([landmarks[0]])
            # 100프레임 동안 가만히 있는 시퀀스 생성
            still_sequence = np.tile(first_frame_landmarks, (MAX_SEQ_LENGTH, 1))
            
            # 원본 및 증강 데이터 추가 (총 10 * 7 = 70개)
            none_label_index = ACTIONS.index("None")
            for _ in range(10 * len(ACTIONS)): # 다른 클래스 수와 비슷하게 생성
                augmented = augment_sequence(still_sequence)
                X.append(augmented)
                y.append(none_label_index)

    X_padded = pad_sequences(X, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post', dtype='float32')
    y_one_hot = to_categorical(y, num_classes=len(ACTIONS))
    
    print(f"💾 다중 클래스 데이터 캐시 저장: {DATA_CACHE_PATH}")
    np.savez(DATA_CACHE_PATH, X=X_padded, y=y_one_hot)

print(f"✅ 데이터 준비 완료: {X_padded.shape[0]}개 샘플")

# --- 2. 모델 학습 또는 로딩 ---
if X_padded.shape[0] < 2:
    print("⚠️ 데이터가 부족하여 모델을 학습할 수 없습니다.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
    )

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🧠 저장된 모델 로딩: {MODEL_SAVE_PATH}")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    else:
        print("🏋️‍♀️ 저장된 모델 없음. 다중 클래스 분류 모델 학습을 시작합니다.")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(MAX_SEQ_LENGTH, X_padded.shape[2])),
            Dropout(0.5),
            LSTM(32),
            Dropout(0.5),
            Dense(16, activation='relu'),
            Dense(len(ACTIONS), activation='softmax') # 3개의 클래스, softmax 활성화
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', # 다중 클래스 손실 함수
                      metrics=['accuracy'])

        print("\n--- 모델 학습 시작 ---")
        model.summary()

        history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
        
        print(f"🧠 학습된 모델 저장: {MODEL_SAVE_PATH}")
        model.save(MODEL_SAVE_PATH)

    # --- 3. 모델 평가 ---
    print("\n--- 모델 평가 ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"🚀 테스트 정확도: {accuracy * 100:.2f}%")

    # 예측 결과 확인
    print("\n--- Test Sample Predictions ---")
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    for i, (pred_class, true_class) in enumerate(zip(y_pred_classes, y_true_classes)):
        pred_label = ACTIONS[pred_class]
        actual_label = ACTIONS[true_class]
        result = "✅" if pred_class == true_class else "❌"
        confidence = y_pred_prob[i][pred_class]
        print(f"Sample {i+1}: Prediction={pred_label} (Confidence: {confidence:.2f}), Actual={actual_label} {result}")