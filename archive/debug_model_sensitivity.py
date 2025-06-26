import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('improved_transformer_model.keras')

print("=== 모델 민감도 테스트 ===")

# 1. 완전히 랜덤한 입력으로 테스트
print("\n1. 랜덤 입력 테스트:")
for i in range(5):
    random_input = np.random.randn(1, 30, 675)
    pred = model.predict(random_input, verbose=0)[0]
    print(f"랜덤 {i+1}: {pred}")

# 2. 미세하게 다른 입력으로 테스트
print("\n2. 미세한 변화 테스트:")
base_input = np.random.randn(1, 30, 675)
for i in range(5):
    # 미세한 노이즈 추가
    noisy_input = base_input + np.random.normal(0, 0.01, (1, 30, 675))
    pred = model.predict(noisy_input, verbose=0)[0]
    print(f"노이즈 {i+1}: {pred}")

# 3. 실제 수어와 유사한 패턴으로 테스트
print("\n3. 수어 패턴 테스트:")
# 포즈는 고정, 손만 변화
base_input = np.random.randn(1, 30, 675)
for i in range(5):
    # 손 부분만 변화 (인덱스 99~140: 왼손, 141~182: 오른손)
    test_input = base_input.copy()
    test_input[0, :, 99:182] += np.random.normal(0, 0.1, (30, 83))
    pred = model.predict(test_input, verbose=0)[0]
    print(f"손 변화 {i+1}: {pred}")

# 4. 극단적인 변화 테스트
print("\n4. 극단적 변화 테스트:")
for i in range(3):
    # 모든 값을 0으로
    zero_input = np.zeros((1, 30, 675))
    pred = model.predict(zero_input, verbose=0)[0]
    print(f"모두 0: {pred}")
    
    # 모든 값을 1로
    ones_input = np.ones((1, 30, 675))
    pred = model.predict(ones_input, verbose=0)[0]
    print(f"모두 1: {pred}") 