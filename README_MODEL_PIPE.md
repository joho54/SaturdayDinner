# 모델 학습 파이프라인 사용법

## 개요
이 파이프라인은 수어 인식 모델을 학습시키기 위한 자동화된 시스템입니다. `spec.json` 파일에 원하는 라벨을 지정하면 해당 라벨에 맞는 데이터만 필터링하여 모델을 학습시킵니다.

## 파일 구조
```
SaturdayDinner/
├── model_pipe.py          # 메인 파이프라인 스크립트
├── spec.json             # 모델 명세 파일 (사용자가 작성)
├── label.csv             # 전체 라벨 데이터
├── models/               # 학습된 모델 저장 디렉토리
└── model-info-*.json     # 모델 정보 파일
```

## 1. 모델 명세 작성 (spec.json)

### 기본 구조
```json
{
  "model_name": "custom_model",
  "description": "사용자 정의 수어 인식 모델",
  "labels": [
    "화재",
    "지진", 
    "태풍",
    "폭우",
    "대피"
  ],
  "training_config": {
    "sequence_length": 30,
    "augmentations_per_video": 20,
    "test_split": 0.2,
    "random_state": 42
  },
  "model_config": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10
  }
}
```

### 필드 설명

#### 필수 필드
- `model_name`: 모델의 이름 (파일명에 사용됨)
- `labels`: 학습할 라벨 목록 (최대 5개)

#### 선택 필드
- `description`: 모델에 대한 설명
- `training_config`: 학습 관련 설정
  - `sequence_length`: 시퀀스 길이 (기본값: 30)
  - `augmentations_per_video`: 비디오당 증강 횟수 (기본값: 20)
  - `test_split`: 테스트 데이터 비율 (기본값: 0.2)
  - `random_state`: 랜덤 시드 (기본값: 42)
- `model_config`: 모델 학습 설정
  - `learning_rate`: 학습률 (기본값: 0.001)
  - `batch_size`: 배치 크기 (기본값: 32)
  - `epochs`: 최대 에포크 수 (기본값: 100)
  - `early_stopping_patience`: 조기 종료 인내심 (기본값: 10)

### 사용 가능한 라벨
현재 `label.csv`에 포함된 라벨들:
- 화재
- 화장실
- 화요일
- 화약
- 화상

## 2. 파이프라인 실행

### 기본 사용법
```bash
python3 model_pipe.py spec.json
```

### 실행 예시
```bash
# 1. spec.json 파일 작성
# 2. 파이프라인 실행
python3 model_pipe.py spec.json
```

### 실행 과정
1. **명세 파일 로드**: `spec.json`에서 설정과 라벨 목록을 읽어옴
2. **라벨 데이터 로드**: `label.csv`에서 전체 라벨 데이터를 로드
3. **데이터 필터링**: 명세에 명시된 라벨과 일치하는 영상만 선택
4. **데이터 전처리**: 비디오에서 랜드마크 추출 및 전처리
5. **데이터 증강**: 각 비디오에 대해 증강 데이터 생성
6. **모델 학습**: 필터링된 데이터로 모델 학습
7. **모델 저장**: 학습된 모델을 `models/` 디렉토리에 저장
8. **정보 저장**: 모델 정보를 `model-info-*.json` 파일로 저장

## 3. 출력 파일

### 모델 파일
- 위치: `models/model_{model_name}_{timestamp}.keras`
- 예시: `models/model_custom_20241201_143022.keras`

### 모델 정보 파일
- 위치: `model-info-{timestamp}.json`
- 예시: `model-info-20241201_143022.json`

### 모델 정보 파일 내용
```json
{
  "name": "custom_model",
  "type": "Functional",
  "total_params": 300838,
  "trainable_params": 300838,
  "input_shape": [null, 30, 675],
  "output_shape": [null, 5],
  "layers_count": 16,
  "model_size_mb": 1.15,
  "labels": ["화재", "지진", "태풍", "폭우", "대피"],
  "model_number": "20241201_143022",
  "created_at": "2024-12-01T14:30:22.123456"
}
```

## 4. 예시

### 예시 1: 재난 관련 수어 모델
```json
{
  "model_name": "disaster_signs",
  "description": "재난 상황 관련 수어 인식 모델",
  "labels": [
    "화재",
    "지진",
    "태풍",
    "폭우",
    "대피"
  ]
}
```

### 예시 2: 요일 관련 수어 모델
```json
{
  "model_name": "weekday_signs",
  "description": "요일 관련 수어 인식 모델",
  "labels": [
    "화요일"
  ],
  "training_config": {
    "augmentations_per_video": 30
  },
  "model_config": {
    "learning_rate": 0.0005,
    "epochs": 150
  }
}
```

## 5. 주의사항

1. **라벨 제한**: 현재 `label.csv`에 있는 라벨만 사용 가능
2. **데이터 경로**: 비디오 파일 경로가 올바르게 설정되어 있어야 함
3. **메모리 사용량**: 대용량 데이터 처리 시 충분한 메모리 확보 필요
4. **GPU 사용**: TensorFlow GPU 버전 설치 시 자동으로 GPU 사용

## 6. 문제 해결

### 일반적인 오류
- **파일 없음**: 비디오 파일 경로 확인
- **메모리 부족**: `batch_size` 줄이기
- **학습 시간 오래**: `epochs` 줄이거나 `early_stopping_patience` 조정

### 로그 확인
실행 중 상세한 로그가 출력되므로 문제 발생 시 로그를 확인하세요. 