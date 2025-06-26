import tensorflow as tf
from tensorflow import keras

def quick_model_check(model_path):
    """Keras 모델 파일을 빠르게 확인하는 함수"""
    print(f"🔍 모델 확인: {model_path}")
    
    try:
        # 모델 로드
        model = keras.models.load_model(model_path)
        
        # 기본 정보 출력
        print(f"✅ 모델 타입: {type(model).__name__}")
        print(f"✅ 모델 이름: {model.name}")
        print(f"✅ 입력 형태: {model.input_shape}")
        print(f"✅ 출력 형태: {model.output_shape}")
        print(f"✅ 총 파라미터: {model.count_params():,}")
        print(f"✅ 레이어 수: {len(model.layers)}")
        
        # 간단한 구조 요약
        print("\n📋 모델 구조:")
        model.summary(show_trainable=True)
        
        return model
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return None

if __name__ == "__main__":
    # 간단한 사용 예시
    model_file = "models/fixed_transformer_model.keras"
    quick_model_check(model_file) 