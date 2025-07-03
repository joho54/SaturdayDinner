"""
Saturday Dinner - 수어 인식 및 처리 라이브러리

이 패키지는 수어 데이터셋 처리, 비디오 경로 관리, 모델 학습 및 추론을 위한 
통합 라이브러리입니다.

주요 모듈:
- utils: 비디오 경로 처리 및 설정 관리
- core: 메인 학습 로직 및 퀴즈 시스템
- categorizer: 데이터 분류 및 클러스터링
- models: 학습된 모델 관리
- scripts: 유틸리티 스크립트
"""

__version__ = "1.0.0"
__author__ = "Saturday Dinner Team"

# 핵심 유틸리티 함수들을 top-level에서 접근 가능하게 함
from .utils.video_path_utils import (
    get_video_root_and_path,
    get_video_number,
    get_root_path_for_number,
    get_all_video_paths,
    validate_video_paths,
    print_video_path_stats,
    VIDEO_ROOTS,
    VIDEO_EXTENSIONS
)

from .utils.config import (
    IDENTICAL_VIDEO_ROOT,
    MODELS_DIR,
    CHECKPOINT_DIR,
    CACHE_DIR,
    get_action_index
)

# 편의를 위한 별칭
find_video = get_video_root_and_path
get_video_path = get_video_root_and_path

__all__ = [
    # Video utilities
    'get_video_root_and_path',
    'find_video',
    'get_video_path',
    'get_video_number',
    'get_root_path_for_number',
    'get_all_video_paths',
    'validate_video_paths',
    'print_video_path_stats',
    'VIDEO_ROOTS',
    'VIDEO_EXTENSIONS',
    
    # Config
    'IDENTICAL_VIDEO_ROOT',
    'MODELS_DIR',
    'CHECKPOINT_DIR',
    'CACHE_DIR',
    'get_action_index',
    
    # Package info
    '__version__',
    '__author__'
] 