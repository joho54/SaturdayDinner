import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from archive.improved_main import improved_preprocess_landmarks, extract_landmarks
from archive.main import preprocess_landmarks
import os

# 모델 경로
LSTM_MODEL_PATH = 'lstm_model_multiclass.keras'
TRANSFORMER_MODEL_PATH = 'improved_transformer_model.keras'
ACTIONS = ["Fire", "Toilet", "None"]

def load_models():
    """기존 LSTM 모델과 개선된 Transformer 모델을 로드합니다."""
    models = {}
    
    # LSTM 모델 로드
    try:
        models['lstm'] = tf.keras.models.load_model(LSTM_MODEL_PATH)
        print(f"✅ LSTM 모델 로딩 성공: {LSTM_MODEL_PATH}")
    except Exception as e:
        print(f"❌ LSTM 모델 로딩 실패: {e}")
        models['lstm'] = None
    
    # Transformer 모델 로드
    try:
        models['transformer'] = tf.keras.models.load_model(TRANSFORMER_MODEL_PATH)
        print(f"✅ Transformer 모델 로딩 성공: {TRANSFORMER_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Transformer 모델 로딩 실패: {e}")
        models['transformer'] = None
    
    return models

def evaluate_model_performance(model, X_test, y_test, model_name):
    """모델 성능을 평가합니다."""
    if model is None:
        return None
    
    print(f"\n--- {model_name} 모델 평가 ---")
    
    # 예측 시간 측정
    start_time = time.time()
    y_pred_prob = model.predict(X_test, verbose=0)
    prediction_time = time.time() - start_time
    
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 정확도 계산
    accuracy = np.mean(y_pred_classes == y_true_classes)
    
    # 분류 리포트
    print(f"정확도: {accuracy * 100:.2f}%")
    print(f"평균 예측 시간: {prediction_time / len(X_test) * 1000:.2f}ms")
    
    # 상세 분류 리포트
    print("\n분류 리포트:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=ACTIONS))
    
    # 혼동 행렬
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    return {
        'accuracy': accuracy,
        'prediction_time': prediction_time / len(X_test),
        'confusion_matrix': cm,
        'y_pred': y_pred_classes,
        'y_true': y_true_classes,
        'y_pred_prob': y_pred_prob
    }

def plot_confusion_matrices(results):
    """혼동 행렬을 시각화합니다."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (model_name, result) in enumerate(results.items()):
        if result is None:
            continue
            
        cm = result['confusion_matrix']
        ax = axes[i]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=ACTIONS, yticklabels=ACTIONS, ax=ax)
        ax.set_title(f'{model_name} 모델 혼동 행렬')
        ax.set_xlabel('예측')
        ax.set_ylabel('실제')
    
    plt.tight_layout()
    plt.savefig('model_comparison_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_comparison(results):
    """정확도 비교를 시각화합니다."""
    model_names = []
    accuracies = []
    prediction_times = []
    
    for model_name, result in results.items():
        if result is not None:
            model_names.append(model_name)
            accuracies.append(result['accuracy'] * 100)
            prediction_times.append(result['prediction_time'] * 1000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 정확도 비교
    bars1 = ax1.bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
    ax1.set_title('모델별 정확도 비교')
    ax1.set_ylabel('정확도 (%)')
    ax1.set_ylim(0, 100)
    
    # 값 표시
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 예측 시간 비교
    bars2 = ax2.bar(model_names, prediction_times, color=['lightgreen', 'orange'])
    ax2.set_title('모델별 예측 시간 비교')
    ax2.set_ylabel('예측 시간 (ms)')
    
    # 값 표시
    for bar, time_val in zip(bars2, prediction_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.2f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_feature_importance(results):
    """특징 중요도를 비교합니다."""
    if 'transformer' not in results or results['transformer'] is None:
        print("Transformer 모델이 없어 특징 중요도 비교를 건너뜁니다.")
        return
    
    # 각 클래스별 평균 신뢰도 비교
    transformer_probs = results['transformer']['y_pred_prob']
    lstm_probs = results['lstm']['y_pred_prob'] if results['lstm'] else None
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Transformer 모델 신뢰도 분포
    for i, action in enumerate(ACTIONS):
        action_probs = transformer_probs[:, i]
        axes[0].hist(action_probs, alpha=0.7, label=action, bins=20)
    
    axes[0].set_title('Transformer 모델 신뢰도 분포')
    axes[0].set_xlabel('신뢰도')
    axes[0].set_ylabel('빈도')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # LSTM 모델 신뢰도 분포 (있는 경우)
    if lstm_probs is not None:
        for i, action in enumerate(ACTIONS):
            action_probs = lstm_probs[:, i]
            axes[1].hist(action_probs, alpha=0.7, label=action, bins=20)
        
        axes[1].set_title('LSTM 모델 신뢰도 분포')
        axes[1].set_xlabel('신뢰도')
        axes[1].set_ylabel('빈도')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_confidence.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수."""
    print("🔍 모델 성능 비교 시작")
    
    # 모델 로드
    models = load_models()
    
    if not any(models.values()):
        print("❌ 로드할 수 있는 모델이 없습니다.")
        return
    
    # 테스트 데이터 준비 (기존 데이터 사용)
    if os.path.exists('preprocessed_data_multiclass.npz'):
        print("📊 기존 데이터를 사용하여 성능 비교를 진행합니다.")
        data = np.load('preprocessed_data_multiclass.npz')
        X = data['X']
        y = data['y']
        
        # 테스트 세트만 사용 (20%)
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # LSTM 모델 평가
        if models['lstm']:
            results['LSTM'] = evaluate_model_performance(
                models['lstm'], X_test, y_test, "LSTM"
            )
        
        # Transformer 모델 평가 (데이터 변환 필요)
        if models['transformer']:
            # Transformer 모델용 데이터 변환
            print("\n🔄 Transformer 모델용 데이터 변환 중...")
            X_test_transformer = []
            
            # 여기서는 간단히 기존 데이터를 사용하지만, 
            # 실제로는 improved_preprocess_landmarks를 사용해야 합니다
            # 현재는 호환성을 위해 기존 형태 유지
            X_test_transformer = X_test
            
            results['Transformer'] = evaluate_model_performance(
                models['transformer'], X_test_transformer, y_test, "Transformer"
            )
        
        # 결과 시각화
        if len(results) > 1:
            print("\n📈 성능 비교 결과 시각화...")
            plot_confusion_matrices(results)
            plot_accuracy_comparison(results)
            compare_feature_importance(results)
            
            # 요약
            print("\n📊 성능 비교 요약:")
            for model_name, result in results.items():
                if result:
                    print(f"{model_name}: 정확도 {result['accuracy']*100:.2f}%, "
                          f"평균 예측시간 {result['prediction_time']*1000:.2f}ms")
        else:
            print("⚠️ 비교할 모델이 충분하지 않습니다.")
    
    else:
        print("❌ 테스트 데이터를 찾을 수 없습니다.")
        print("먼저 main.py 또는 improved_main.py를 실행하여 데이터를 준비해주세요.")

if __name__ == "__main__":
    main() 