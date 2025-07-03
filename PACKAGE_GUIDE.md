# Saturday Dinner 패키지 사용 가이드

## 📦 패키지 구조

```
saturday_dinner/
├── __init__.py                  # 메인 패키지 (핵심 함수들 자동 노출)
├── utils/                       # 🛠️ 유틸리티 모듈
│   ├── __init__.py
│   ├── video_path_utils.py      # 비디오 경로 관리
│   └── config.py               # 설정 관리
├── core/                        # 🧠 핵심 기능
│   ├── __init__.py
│   ├── main.py                 # 메인 학습 로직
│   ├── sign_quiz.py            # 퀴즈 시스템
│   └── label_cache_system.py   # 라벨 캐시 시스템
├── categorizer/                 # 📊 데이터 분류 및 클러스터링
│   ├── __init__.py
│   ├── VideoCluster.py         # 비디오 클러스터링
│   ├── LabelCluster.py         # 라벨 클러스터링
│   ├── Categorizer.py          # 기본 분류기
│   ├── CrossCategorizer.py     # 교차 분류기
│   └── extracted-src/          # 추출된 데이터
├── scripts/                     # 🔧 유틸리티 스크립트
│   ├── __init__.py
│   └── label_picker.py         # 라벨 추출 스크립트
└── data/                        # 📁 데이터 파일
    ├── __init__.py
    └── labels.csv              # 라벨 데이터
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 개발 모드 설치 (권장)
pip install -e .

# 또는 일반 설치
pip install saturday-dinner
```

### 2. 기본 사용법

```python
# 🎯 가장 간단한 사용법
import saturday_dinner as sd

# 비디오 경로 찾기
video_path = sd.find_video("KETI_SL_0000000419.MOV")
print(f"📹 비디오 경로: {video_path}")

# 모든 비디오 경로 가져오기  
all_paths = sd.get_all_video_paths()
print(f"📊 총 비디오 개수: {len(all_paths)}")

# 비디오 경로 통계 출력
sd.print_video_path_stats()
```

## 📋 주요 기능별 사용법

### 🔍 비디오 경로 관리

```python
# 방법 1: 메인 패키지에서 직접 사용
import saturday_dinner as sd

video_path = sd.get_video_root_and_path("KETI_SL_0000000419.MOV")
video_number = sd.get_video_number("KETI_SL_0000000419.MOV")  # 419
root_path = sd.get_root_path_for_number(419)

# 방법 2: utils 모듈에서 직접 import
from saturday_dinner.utils import get_video_root_and_path, VIDEO_ROOTS
from saturday_dinner.utils.video_path_utils import validate_video_paths

# 방법 3: 별칭 사용 (편의성)
path1 = sd.find_video("filename.MOV")     # get_video_root_and_path의 별칭
path2 = sd.get_video_path("filename.MOV") # 동일한 별칭
```

### ⚙️ 설정 관리

```python
# 방법 1: 메인 패키지에서
import saturday_dinner as sd
print(f"모델 디렉토리: {sd.MODELS_DIR}")
print(f"캐시 디렉토리: {sd.CACHE_DIR}")

# 방법 2: 설정 모듈에서 직접
from saturday_dinner.utils.config import (
    TARGET_SEQ_LENGTH, BATCH_SIZE, LEARNING_RATE,
    MODELS_DIR, CACHE_DIR, VIDEO_EXTENSIONS
)

print(f"시퀀스 길이: {TARGET_SEQ_LENGTH}")
print(f"배치 크기: {BATCH_SIZE}")
```

### 📊 데이터 분류 및 클러스터링

```python
# categorizer 모듈 사용
from saturday_dinner.categorizer import MotionExtractor, LabelClusterer

# 모션 추출기 생성
extractor = MotionExtractor(output_dir="my_extracted_data")

# 라벨 클러스터링
clusterer = LabelClusterer()
```

### 🎯 모델 학습 및 추론

```python
# 메인 학습 실행
from saturday_dinner.core.main import main
main()

# 또는 스크립트로 실행
# python -m saturday_dinner.core.main
```

## 🔧 명령줄 도구

### 라벨 추출 도구

```bash
# 기본 사용법
python -m saturday_dinner.scripts.label_picker

# 커스텀 설정
python -m saturday_dinner.scripts.label_picker \
    --input my_labels.csv \
    --output my_spec.json \
    --chapter chapter_100 \
    --column "라벨컬럼명"

# None 라벨 제외
python -m saturday_dinner.scripts.label_picker --no-none
```

### 퀴즈 시스템

```bash
# 퀴즈 실행
python -m saturday_dinner.core.sign_quiz model_info.json
```

## 🛠️ 개발자용 고급 사용법

### 1. 커스텀 비디오 경로 설정

```python
# saturday_dinner/utils/video_path_utils.py 수정
VIDEO_ROOTS = {
    (1, 1000): "/your/custom/path/videos1",
    (1001, 2000): "/your/custom/path/videos2",
    # ...
}
```

### 2. 설정 오버라이드

```python
# saturday_dinner/utils/config.py 수정 또는
# 런타임에 설정 변경
import saturday_dinner.utils.config as config
config.BATCH_SIZE = 16
config.LEARNING_RATE = 0.0005
```

### 3. 패키지 확장

```python
# 새로운 모듈 추가시 __init__.py 업데이트
# saturday_dinner/__init__.py에 새 함수 추가
from .your_module import your_function

__all__.append('your_function')
```

## 🐛 문제 해결

### Import 에러

```python
# Saturday Dinner 패키지가 설치되지 않은 경우
pip install -e .

# 개별 모듈 import 실패
try:
    from saturday_dinner import find_video
except ImportError:
    from saturday_dinner.utils.video_path_utils import get_video_root_and_path as find_video
```

### 비디오 파일 경로 문제

```python
# 비디오 경로 진단
import saturday_dinner as sd
sd.print_video_path_stats()

# 특정 파일 확인
path = sd.find_video("KETI_SL_0000000001.avi", verbose=True)
if not path:
    print("비디오 파일이 존재하지 않거나 경로 설정을 확인하세요.")
```

## 📚 예제 코드

### 완전한 워크플로우 예제

```python
#!/usr/bin/env python3
"""
Saturday Dinner 패키지를 사용한 완전한 워크플로우 예제
"""
import saturday_dinner as sd

def main():
    print("=== Saturday Dinner 워크플로우 ===")
    
    # 1. 시스템 상태 확인
    print("\n1️⃣ 시스템 상태 확인")
    sd.print_video_path_stats()
    
    # 2. 특정 비디오 찾기
    print("\n2️⃣ 비디오 파일 검색")
    test_files = ["KETI_SL_0000000001.avi", "KETI_SL_0000000002.avi"]
    for filename in test_files:
        path = sd.find_video(filename)
        if path:
            print(f"✅ {filename} -> Found")
        else:
            print(f"❌ {filename} -> Not found")
    
    # 3. 설정 정보 확인
    print("\n3️⃣ 설정 정보")
    print(f"모델 디렉토리: {sd.MODELS_DIR}")
    print(f"캐시 디렉토리: {sd.CACHE_DIR}")
    print(f"지원 확장자: {sd.VIDEO_EXTENSIONS}")
    
    # 4. 라벨 추출 (데이터가 있는 경우)
    print("\n4️⃣ 라벨 데이터 처리")
    try:
        from saturday_dinner.scripts.label_picker import extract_unique_labels
        labels = extract_unique_labels("saturday_dinner/data/labels.csv")
        print(f"추출된 라벨 수: {len(labels)}")
    except Exception as e:
        print(f"라벨 추출 실패: {e}")
    
    print("\n🎉 워크플로우 완료!")

if __name__ == "__main__":
    main()
```

## 🔄 마이그레이션 가이드

기존 코드를 새 패키지 구조로 마이그레이션하는 방법:

### Before (기존 코드)
```python
# 기존 방식
from video_path_utils import get_video_root_and_path
from config import MODELS_DIR, BATCH_SIZE
import sys
sys.path.append("...")
```

### After (새 패키지 구조)
```python
# 새로운 방식
import saturday_dinner as sd

# 비디오 경로
path = sd.find_video("filename.MOV")

# 설정
models_dir = sd.MODELS_DIR
batch_size = sd.get_config_value("BATCH_SIZE")  # 또는 직접 import

# 또는 세밀한 제어가 필요한 경우
from saturday_dinner.utils import get_video_root_and_path, MODELS_DIR
from saturday_dinner.utils.config import BATCH_SIZE
```

---

**📞 지원**: 문제가 있으시면 GitHub Issues를 통해 알려주세요!  
**📖 더 많은 예제**: [Wiki](링크) 참조 