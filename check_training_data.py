import numpy as np
import tensorflow as tf

# 학습 데이터 로드
print("=== 학습 데이터 분석 ===")
data = np.load('improved_preprocessed_data.npz')
X = data['X']
y = data['y']

print(f"데이터 형태: {X.shape}")
print(f"레이블 형태: {y.shape}")

# 클래스 분포 확인
class_counts = np.sum(y, axis=0)
print(f"\n클래스 분포:")
for i, count in enumerate(class_counts):
    print(f"클래스 {i}: {count}개 ({count/len(y)*100:.1f}%)")

# 데이터 통계 확인
print(f"\n데이터 통계:")
print(f"X 평균: {np.mean(X):.6f}")
print(f"X 표준편차: {np.std(X):.6f}")
print(f"X 최소값: {np.min(X):.6f}")
print(f"X 최대값: {np.max(X):.6f}")

# 각 클래스별 데이터 통계
print(f"\n클래스별 데이터 통계:")
for i in range(len(class_counts)):
    class_mask = y[:, i] == 1
    class_data = X[class_mask]
    if len(class_data) > 0:
        print(f"클래스 {i}:")
        print(f"  평균: {np.mean(class_data):.6f}")
        print(f"  표준편차: {np.std(class_data):.6f}")
        print(f"  최소값: {np.min(class_data):.6f}")
        print(f"  최대값: {np.max(class_data):.6f}")

# 모델 로드 및 테스트
print(f"\n=== 모델 테스트 ===")
model = tf.keras.models.load_model('improved_transformer_model.keras')

# 학습 데이터로 예측
print("학습 데이터로 예측 테스트:")
y_pred = model.predict(X, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y, axis=1)

# 정확도 계산
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"학습 데이터 정확도: {accuracy:.4f}")

# 각 클래스별 정확도
print(f"\n클래스별 정확도:")
for i in range(len(class_counts)):
    class_mask = y_true_classes == i
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(y_pred_classes[class_mask] == y_true_classes[class_mask])
        print(f"클래스 {i}: {class_accuracy:.4f}")

# 예측 확률 분포 확인
print(f"\n예측 확률 분포:")
for i in range(len(class_counts)):
    class_mask = y_true_classes == i
    if np.sum(class_mask) > 0:
        class_probs = y_pred[class_mask]
        print(f"클래스 {i} 예측 확률:")
        print(f"  평균: {np.mean(class_probs, axis=0)}")
        print(f"  표준편차: {np.std(class_probs, axis=0)}")

# 몇 개 샘플의 상세 예측 확인
print(f"\n=== 샘플 예측 상세 분석 ===")
for i in range(min(5, len(X))):
    print(f"\n샘플 {i+1}:")
    print(f"  실제 클래스: {y_true_classes[i]}")
    print(f"  예측 클래스: {y_pred_classes[i]}")
    print(f"  예측 확률: {y_pred[i]}")
    print(f"  입력 데이터 통계: 평균={np.mean(X[i]):.6f}, 표준편차={np.std(X[i]):.6f}") 