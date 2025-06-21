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

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# 경로 및 상수 설정
VIDEO_ROOT = "/Volumes/Sub_Storage/수어 데이터셋/0001~3000(영상)"
MAX_SEQ_LENGTH = 100  # 최대 프레임 길이 (패딩에 사용)

label_dict = {
    # 화재
    "KETI_SL_0000000419.MOV": "화재",
    "KETI_SL_0000000838.MTS": "화재",
    "KETI_SL_0000001255.MTS": "화재",
    "KETI_SL_0000001674.MTS": "화재",
    "KETI_SL_0000002032.MOV": "화재",
    "KETI_SL_0000002451.MP4": "화재",
    "KETI_SL_0000002932.MOV": "화재",
    # 화장실
    "KETI_SL_0000000418.MOV": "화장실",
    "KETI_SL_0000000837.MTS": "화장실",
    "KETI_SL_0000001254.MTS": "화장실",
    "KETI_SL_0000001673.MTS": "화장실",
    "KETI_SL_0000002031.MOV": "화장실",
    "KETI_SL_0000002450.MP4": "화장실",
    "KETI_SL_0000002931.MOV": "화장실"
}

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
            "face": results.face_landmarks
        }
        landmarks_list.append(frame_data)
    
    cap.release()
    return landmarks_list

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
    
    X.append(processed_sequence)
    y.append(1 if label == "화재" else 0)

print(f"✅ 추출 완료: {len(X)}개 샘플")

if len(X) < 2:
    print("⚠️ 데이터가 부족하여 모델을 학습할 수 없습니다.")
else:
    # 데이터 패딩
    X_padded = pad_sequences(X, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post', dtype='float32')
    y_np = np.array(y)

    # 데이터셋 차원 확인
    print(f"Padded X shape: {X_padded.shape}")
    print(f"y shape: {y_np.shape}")
    
    if X_padded.shape[0] == 0:
        print("⚠️ 처리된 데이터가 없어 모델 학습을 건너뜁니다.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_np, test_size=0.2, random_state=42, stratify=y_np
        )

        # LSTM 모델 정의
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(MAX_SEQ_LENGTH, X_padded.shape[2])),
            Dropout(0.5),
            LSTM(32),
            Dropout(0.5),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        print("\n--- 모델 학습 시작 ---")
        model.summary()

        history = model.fit(X_train, y_train, epochs=30, batch_size=4, validation_split=0.2)
        
        print("\n--- 모델 평가 ---")
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"🚀 테스트 정확도: {accuracy * 100:.2f}%")

        # 예측 결과 확인
        print("\n--- 테스트 샘플 예측 결과 ---")
        y_pred_prob = model.predict(X_test)
        for i, (pred_prob, actual) in enumerate(zip(y_pred_prob, y_test)):
            pred_label = "화재" if pred_prob[0] > 0.5 else "화장실"
            actual_label = "화재" if actual == 1 else "화장실"
            result = "✅" if (pred_prob[0] > 0.5) == actual else "❌"
            print(f"샘플 {i+1}: 예측={pred_label} (신뢰도: {pred_prob[0]:.2f}), 실제={actual_label} {result}")