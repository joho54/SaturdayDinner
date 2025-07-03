"""
Core 패키지 - 핵심 기능 모듈들

이 패키지는 수어 인식 모델의 학습, 추론, 퀴즈 시스템 등
핵심 기능들을 제공합니다.
"""

# 핵심 기능들을 import (에러가 발생해도 패키지는 로드되도록 함)
try:
    # main.py에서 주요 함수들 import (있다면)
    # sign_quiz.py에서 주요 함수들 import (있다면)
    
    __all__ = [
        # 여기에 main.py와 sign_quiz.py에서 export할 함수들 추가
    ]
except ImportError as e:
    __all__ = []
    print(f"⚠️ Core 모듈 일부 import 실패: {e}")
    print("   기본 기능은 정상 작동합니다.") 