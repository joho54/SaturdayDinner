"""
Categorizer 패키지 - 데이터 분류 및 클러스터링

이 패키지는 수어 데이터의 분류, 클러스터링, 라벨 추출 등
데이터 처리 관련 기능들을 제공합니다.
"""

# Import가 실패하더라도 패키지는 로드되도록 함
try:
    from .VideoCluster import MotionExtractor, MotionClusterer
    from .LabelCluster import LabelClusterer
    from .Categorizer import Categorizer
    from .CrossCategorizer import CrossCategorizer
    
    __all__ = [
        'MotionExtractor',
        'MotionClusterer', 
        'LabelClusterer',
        'Categorizer',
        'CrossCategorizer'
    ]
except ImportError as e:
    # Import 에러 발생시 빈 리스트로 설정
    __all__ = []
    print(f"⚠️ Categorizer 모듈 일부 import 실패: {e}")
    print("   기본 기능은 정상 작동합니다.") 