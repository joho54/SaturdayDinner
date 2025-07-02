# Motion Extraction and Clustering System 🎬🤖

수어 영상에서 MediaPipe 키포인트를 추출하고 동작 유사성에 따라 클러스터링하는 완전한 시스템

## 📋 시스템 개요

이 시스템은 다음 4가지 핵심 기능을 수행합니다:

1. **라벨 추출**: `labels.csv`에서 유니크한 라벨과 첫 번째 파일명 쌍 추출
2. **키포인트 추출**: MediaPipe를 사용하여 영상에서 포즈, 손, 얼굴 키포인트 시퀸스 추출
3. **데이터 저장**: 추출된 시퀸스를 `extracted-src/` 디렉토리에 저장하고 `extracted_labels.csv` 생성
4. **동작 클러스터링**: 동적 특성을 분석하여 유사한 동작끼리 클러스터링

## 🏗️ 시스템 구조

### 핵심 클래스들

- **`MotionExtractor`**: MediaPipe 키포인트 추출 및 저장
- **`MotionEmbedder`**: 동적 특성 추출을 위한 고급 임베딩 시스템
- **`MotionClusterer`**: 동작 클러스터링 및 분석

### 파일 구조
```
📁 SaturdayDinner/
├── 📄 extractor.py          # 메인 시스템
├── 📄 test_extractor.py     # 테스트 스크립트  
├── 📄 video_path_utils.py   # 비디오 경로 유틸리티
├── 📄 labels.csv            # 원본 라벨 데이터 (43,889개 파일, 657개 유니크 라벨)
├── 📁 extracted-src/        # 추출된 키포인트 시퀸스 저장소
└── 📄 extracted_labels.csv  # 추출된 시퀸스와 라벨 매핑
```

## 🔧 사용 방법

### 1. 테스트 실행 (권장)
```bash
python test_extractor.py
```
소수의 영상으로 시스템이 정상 작동하는지 확인합니다.

### 2. 전체 시스템 실행
```bash
python extractor.py
```
모든 유니크한 라벨의 영상을 처리합니다 (657개 영상).

### 3. 개별 컴포넌트 사용
```python
from extractor import MotionExtractor, MotionEmbedder, MotionClusterer

# 1. 키포인트 추출
extractor = MotionExtractor()
labels_dict = extractor.extract_unique_labels_with_first_files("labels.csv")
sequences = extractor.extract_all_sequences(labels_dict)

# 2. 동작 클러스터링
embedder = MotionEmbedder()
clusterer = MotionClusterer(embedder)
results = clusterer.cluster_motions(sequences)
```

## 🎯 고급 동적 임베딩 특성

시스템은 동작의 동적 특성을 포착하기 위해 다음과 같은 고급 특성들을 추출합니다:

### 1. 운동학적 특성
- **속도 (Velocity)**: 프레임 간 키포인트 이동 속도
- **가속도 (Acceleration)**: 속도 변화율
- **각속도 (Angular Velocity)**: 관절 각도 변화율

### 2. 궤적 특성
- **통계적 특성**: 평균, 표준편차, 최대/최소값, 변동 범위
- **움직임 패턴**: 각 랜드마크의 이동 궤적 분석

### 3. 주파수 도메인 특성
- **FFT 기반 분석**: 동작의 주파수 성분 분석
- **리듬/템포 특성**: 동작의 주기성 및 리듬 패턴

### 4. 신체 부위별 상대적 움직임
- **손-몸통 비율**: 손 움직임 대비 몸통 움직임
- **얼굴-포즈 관계**: 얼굴 표정과 전체 포즈의 상대적 움직임
- **좌우 균형**: 양손의 움직임 비교

## 📊 출력 데이터

### 1. extracted_labels.csv
```csv
sequence_path,label
extracted-src/KETI_SL_0000000419_MOV_화재.pkl,화재
extracted-src/KETI_SL_0000000415_MOV_화상.pkl,화상
...
```

### 2. 키포인트 시퀸스 파일 (.pkl)
각 파일에는 `KeypointSequence` 객체가 포함되어 있습니다:
- `pose_landmarks`: 포즈 키포인트 (33개 점)
- `left_hand_landmarks`: 왼손 키포인트 (21개 점)  
- `right_hand_landmarks`: 오른손 키포인트 (21개 점)
- `face_landmarks`: 얼굴 키포인트 (468개 점)
- `sequence`: 전체 결합된 시퀸스
- `frame_count`, `fps`: 메타데이터

### 3. 클러스터링 결과
```python
clustering_results = {
    'cluster_labels': [...],      # 각 동작의 클러스터 라벨
    'motion_labels': [...],       # 원본 한국어 라벨  
    'features': [...],            # PCA 후 특성 벡터
    'n_clusters': 15,             # 자동 선택된 클러스터 수
    'sequences': [...]            # 원본 시퀸스 객체들
}
```

## 🎯 클러스터링 자동 최적화

시스템은 다음 방법으로 최적의 클러스터 수를 자동 선택합니다:

1. **실루엣 스코어**: 클러스터 내 응집도와 클러스터 간 분리도 평가
2. **PCA 차원 축소**: 95% 분산을 유지하면서 차원 축소
3. **표준화**: 모든 특성을 동일한 스케일로 정규화

## 🔍 특성 추출 세부사항

### MediaPipe 키포인트
- **포즈**: 33개 랜드마크 (어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목 등)
- **손**: 각 손당 21개 랜드마크 (손가락 관절들)
- **얼굴**: 468개 랜드마크 (눈, 코, 입, 얼굴 윤곽)

### 동적 특성 벡터
```python
features = [
    velocity_features,      # 속도 관련 특성
    acceleration_features,  # 가속도 특성  
    angular_features,       # 각속도 특성
    trajectory_features,    # 궤적 통계 특성
    frequency_features,     # 주파수 도메인 특성
    relative_motion         # 상대적 움직임 특성
]
```

## 🚀 성능 및 확장성

- **병렬 처리**: 가능한 부분에서 병렬 처리 지원
- **메모리 효율성**: 대용량 비디오 처리를 위한 스트리밍 방식
- **오류 복구**: 개별 파일 처리 실패가 전체 시스템에 영향을 주지 않음
- **진행률 표시**: tqdm을 통한 상세한 진행률 표시

## 📈 예상 결과

657개의 유니크한 라벨로부터:
- 비슷한 의미의 동작들이 같은 클러스터로 그룹화
- 동적 특성이 유사한 동작들의 자동 발견
- 수어 동작의 세밀한 차이점과 공통점 분석

## 🛠️ 요구사항

시스템은 `requirements_improved.txt`의 패키지들을 사용합니다:
- `mediapipe>=0.10.0`: 키포인트 추출
- `opencv-python>=4.8.0`: 비디오 처리
- `scikit-learn>=1.3.0`: 클러스터링
- `numpy>=1.24.0`, `pandas>=2.0.0`: 데이터 처리
- `tqdm>=4.65.0`: 진행률 표시

## 🎉 결론

이 시스템은 수어 동작의 복잡한 동적 특성을 포착하고 분석할 수 있는 완전한 솔루션을 제공합니다. MediaPipe의 정확한 키포인트 추출과 고급 동적 임베딩 기법을 결합하여 수어 동작의 미묘한 차이까지 분석할 수 있습니다. 