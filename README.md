# 수어 인식 모델 훈련 및 퀴즈 시스템

## 데이터 크기 최적화 기능

### 1. 실험 모드 (다양한 데이터 크기 테스트)
```bash
# 다양한 데이터 크기로 실험하여 최적 설정 찾기
python3 model_pipe.py spec.json --experiment
```

### 2. 자동 튜닝 모드 (자동으로 최적 크기 찾기)
```bash
# 자동으로 최적 데이터 크기를 찾아서 학습
python3 model_pipe.py spec.json --auto-tune
```

### 3. 고정 데이터 크기 모드
```bash
# 클래스당 5개 비디오로 고정하여 학습
python3 model_pipe.py spec.json --data-size 5
```

### 4. 동적 설정 모드 (기본값)
```bash
# 데이터셋 크기에 따라 자동으로 최적 설정 적용
python3 model_pipe.py spec.json
```

## 권장 사용 순서

1. **실험 모드로 최적 크기 찾기**
   ```bash
   python3 model_pipe.py spec.json --experiment
   ```

2. **발견된 최적 크기로 학습**
   ```bash
   python3 model_pipe.py spec.json --data-size [최적크기]
   ```

3. **또는 자동 튜닝으로 한 번에 처리**
   ```bash
   python3 model_pipe.py spec.json --auto-tune
   ```

## 실험 결과 해석

- **작은 데이터 (2-5개)**: 빠른 학습, 과적합 위험
- **중간 데이터 (5-10개)**: 균형잡힌 성능
- **큰 데이터 (10개 이상)**: 안정적 성능, 학습 시간 증가

## 퀴즈 실행

```bash
python3 sign_quiz.py models/model-info-[timestamp].json
``` 