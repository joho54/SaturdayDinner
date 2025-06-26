import tensorflow as tf
from tensorflow import keras
import numpy as np
import json

def convert_numpy_types(obj):
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    return obj

def inspect_keras_model(model_path):
    """
    Keras 모델 파일의 상세한 명세를 확인하는 함수
    """
    print(f"=== 모델 파일 분석: {model_path} ===\n")
    
    try:
        # 1. 모델 로드
        model = keras.models.load_model(model_path)
        print("✅ 모델 로드 성공!")
        
        # 2. 기본 모델 정보
        print("\n📋 기본 모델 정보:")
        print(f"모델 타입: {type(model).__name__}")
        print(f"모델 이름: {model.name}")
        
        # 3. 모델 구조 요약
        print("\n🏗️ 모델 구조 요약:")
        model.summary()
        
        # 4. 레이어별 상세 정보 (수정된 버전)
        print("\n🔍 레이어별 상세 정보:")
        for i, layer in enumerate(model.layers):
            print(f"\n레이어 {i}: {layer.name}")
            print(f"  - 타입: {type(layer).__name__}")
            
            # 출력 형태 안전하게 가져오기
            try:
                if hasattr(layer, 'output_shape'):
                    print(f"  - 출력 형태: {layer.output_shape}")
                elif hasattr(layer, 'output'):
                    print(f"  - 출력 형태: {layer.output.shape}")
            except:
                print(f"  - 출력 형태: 확인 불가")
            
            print(f"  - 파라미터 수: {layer.count_params():,}")
            
            # 특정 레이어 타입에 대한 추가 정보
            if hasattr(layer, 'units'):
                print(f"  - 유닛 수: {layer.units}")
            if hasattr(layer, 'filters'):
                print(f"  - 필터 수: {layer.filters}")
            if hasattr(layer, 'kernel_size'):
                print(f"  - 커널 크기: {layer.kernel_size}")
            if hasattr(layer, 'activation'):
                print(f"  - 활성화 함수: {layer.activation}")
            if hasattr(layer, 'rate'):
                print(f"  - Dropout 비율: {layer.rate}")
            if hasattr(layer, 'num_heads'):
                print(f"  - 어텐션 헤드 수: {layer.num_heads}")
        
        # 5. 모델 입력/출력 정보
        print("\n📥📤 입력/출력 정보:")
        print(f"입력 형태: {model.input_shape}")
        print(f"출력 형태: {model.output_shape}")
        
        # 6. 모델 가중치 정보
        print("\n⚖️ 가중치 정보:")
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"총 파라미터 수: {total_params:,}")
        print(f"훈련 가능한 파라미터: {trainable_params:,}")
        print(f"훈련 불가능한 파라미터: {non_trainable_params:,}")
        
        # 7. 모델 크기 추정 (MB)
        model_size_mb = total_params * 4 / (1024 * 1024)  # float32 기준
        print(f"예상 모델 크기: {model_size_mb:.2f} MB")
        
        # 8. 컴파일 정보 (가능한 경우)
        print("\n⚙️ 컴파일 정보:")
        if hasattr(model, 'optimizer') and model.optimizer:
            print(f"옵티마이저: {type(model.optimizer).__name__}")
        if hasattr(model, 'loss'):
            print(f"손실 함수: {model.loss}")
        if hasattr(model, 'metrics'):
            print(f"메트릭: {model.metrics}")
        
        return {
            'name': model.name,
            'type': type(model).__name__,
            'total_params': convert_numpy_types(total_params),
            'trainable_params': convert_numpy_types(trainable_params),
            'input_shape': convert_numpy_types(model.input_shape),
            'output_shape': convert_numpy_types(model.output_shape),
            'layers_count': len(model.layers),
            'model_size_mb': convert_numpy_types(model_size_mb)
        }
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return None

def compare_models(model_paths):
    """
    여러 모델을 비교하는 함수
    """
    print("=== 모델 비교 ===\n")
    
    models_info = {}
    
    for path in model_paths:
        try:
            info = inspect_keras_model(path)
            if info:
                models_info[path] = info
        except Exception as e:
            print(f"❌ {path} 분석 실패: {str(e)}")
    
    # 비교 테이블 출력
    if models_info:
        print(f"{'모델명':<35} {'타입':<15} {'총 파라미터':<12} {'훈련가능':<10} {'레이어수':<8} {'크기(MB)':<10}")
        print("-" * 100)
        for path, info in models_info.items():
            model_name = path.split('/')[-1]
            print(f"{model_name:<35} {info['type']:<15} {info['total_params']:<12,} {info['trainable_params']:<10,} {info['layers_count']:<8} {info['model_size_mb']:<10.2f}")
    
    return models_info

def export_model_info(model_path, output_file):
    """
    모델 정보를 JSON 파일로 내보내는 함수
    """
    try:
        info = inspect_keras_model(model_path)
        if info:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            print(f"✅ 모델 정보가 {output_file}에 저장되었습니다.")
        else:
            print("❌ 모델 정보를 가져올 수 없습니다.")
    except Exception as e:
        print(f"❌ JSON 내보내기 실패: {str(e)}")

if __name__ == "__main__":
    # 사용 예시
    model_files = [
        "models/fixed_transformer_model.keras",
        "models/lstm_model_multiclass.keras", 
        "models/improved_transformer_model.keras"
    ]
    
    print("🤖 Keras 모델 명세 확인 도구\n")
    
    # 1. 개별 모델 상세 분석
    for model_file in model_files:
        inspect_keras_model(model_file)
        print("\n" + "="*60 + "\n")
    
    # 2. 모델 비교
    compare_models(model_files)
    
    # 3. 모델 정보 JSON 내보내기 (선택사항)
    print("\n" + "="*60)
    print("JSON 내보내기 예시:")
    export_model_info("models/fixed_transformer_model.keras", "model_info.json") 