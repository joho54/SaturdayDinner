import numpy as np
import matplotlib.pyplot as plt

def analyze_training_logs():
    """학습 로그 분석"""
    
    print("🔍 학습 로그 분석")
    print("=" * 60)
    
    # 실제 학습 로그 데이터
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    train_acc = [0.1562, 0.1641, 0.2240, 0.2396, 0.2943, 0.2891, 0.3333, 0.4141, 0.4479, 0.5182, 0.5286, 0.0]
    train_loss = [1.9694, 1.9581, 1.8989, 1.8814, 1.7715, 1.7260, 1.6305, 1.4263, 1.3442, 1.2573, 1.0911, 0.0]
    val_acc = [0.2708, 0.3229, 0.3229, 0.3229, 0.3438, 0.4583, 0.5208, 0.5729, 0.5938, 0.6354, 0.7188, 0.0]
    val_loss = [1.7547, 1.7363, 1.6388, 1.5051, 1.4812, 1.3853, 1.1979, 1.0446, 0.9084, 0.7802, 0.6449, 0.0]
    
    # 마지막 에폭 제거 (중단됨)
    epochs = epochs[:-1]
    train_acc = train_acc[:-1]
    train_loss = train_loss[:-1]
    val_acc = val_acc[:-1]
    val_loss = val_loss[:-1]
    
    print("📊 학습 진행 상황 분석:")
    print(f"   총 에폭: {len(epochs)}")
    print(f"   최종 훈련 정확도: {train_acc[-1]:.4f}")
    print(f"   최종 검증 정확도: {val_acc[-1]:.4f}")
    print(f"   최종 훈련 손실: {train_loss[-1]:.4f}")
    print(f"   최종 검증 손실: {val_loss[-1]:.4f}")
    
    # 개선률 계산
    print(f"\n📈 개선률 분석:")
    train_acc_improvement = (train_acc[-1] - train_acc[0]) / train_acc[0] * 100
    val_acc_improvement = (val_acc[-1] - val_acc[0]) / val_acc[0] * 100
    train_loss_improvement = (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
    val_loss_improvement = (val_loss[0] - val_loss[-1]) / val_loss[0] * 100
    
    print(f"   훈련 정확도 개선: {train_acc_improvement:.1f}%")
    print(f"   검증 정확도 개선: {val_acc_improvement:.1f}%")
    print(f"   훈련 손실 개선: {train_loss_improvement:.1f}%")
    print(f"   검증 손실 개선: {val_loss_improvement:.1f}%")
    
    return epochs, train_acc, train_loss, val_acc, val_loss

def identify_problems():
    """문제점 식별"""
    
    print(f"\n{'='*60}")
    print("⚠️ 문제점 식별")
    print(f"{'='*60}")
    
    problems = [
        {
            "issue": "과적합 징후",
            "description": "검증 정확도가 훈련 정확도보다 높음",
            "evidence": "Epoch 11: train_acc=0.5286, val_acc=0.7188",
            "severity": "높음"
        },
        {
            "issue": "학습 불안정성",
            "description": "훈련 정확도가 일정하지 않음",
            "evidence": "Epoch 5→6: 0.2943→0.2891 (감소)",
            "severity": "중간"
        },
        {
            "issue": "데이터 불균형",
            "description": "초기 정확도가 0.1562로 매우 낮음",
            "evidence": "3개 클래스에서 랜덤 정확도는 0.3333",
            "severity": "높음"
        },
        {
            "issue": "모델 복잡도 부족",
            "description": "현재 CNN+LSTM 구조가 데이터에 부족할 수 있음",
            "evidence": "675차원 입력에 비해 모델이 단순함",
            "severity": "중간"
        }
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"{i}. {problem['issue']} ({problem['severity']})")
        print(f"   설명: {problem['description']}")
        print(f"   증거: {problem['evidence']}")
        print()

def suggest_solutions():
    """해결책 제안"""
    
    print(f"{'='*60}")
    print("💡 해결책 제안")
    print(f"{'='*60}")
    
    solutions = [
        {
            "category": "데이터 관련",
            "solutions": [
                "데이터 불균형 확인 및 해결",
                "증강 데이터 품질 개선",
                "데이터 전처리 강화"
            ]
        },
        {
            "category": "모델 관련",
            "solutions": [
                "모델 복잡도 증가",
                "Attention 메커니즘 추가",
                "더 깊은 네트워크 구조"
            ]
        },
        {
            "category": "학습 관련",
            "solutions": [
                "배치 크기 조정",
                "다른 옵티마이저 시도",
                "학습률 스케줄링 개선"
            ]
        },
        {
            "category": "정규화 관련",
            "solutions": [
                "Dropout 비율 조정",
                "L2 정규화 강화",
                "배치 정규화 추가"
            ]
        }
    ]
    
    for category in solutions:
        print(f"📋 {category['category']}:")
        for solution in category['solutions']:
            print(f"   • {solution}")
        print()

def analyze_data_distribution():
    """데이터 분포 분석"""
    
    print(f"{'='*60}")
    print("📊 데이터 분포 분석")
    print(f"{'='*60}")
    
    # 현재 설정 기반 분석
    print("현재 설정:")
    print(f"   LABEL_MAX_SAMPLES_PER_CLASS: 20")
    print(f"   MIN_SAMPLES_PER_CLASS: 20")
    print(f"   AUGMENTATIONS_PER_VIDEO: 3")
    
    # 예상 데이터 분포
    print(f"\n예상 데이터 분포:")
    print(f"   원본 데이터: 20개/클래스")
    print(f"   증강 후: 20 × (1+3) = 80개/클래스")
    print(f"   총 샘플: 80 × 3클래스 = 240개")
    
    # 문제점 분석
    print(f"\n잠재적 문제점:")
    print(f"   1. 데이터가 너무 적음 (240개)")
    print(f"   2. 증강 데이터의 품질 문제")
    print(f"   3. 클래스 간 불균형")
    
    return 240  # 총 샘플 수

def recommend_immediate_actions():
    """즉시 실행 가능한 해결책"""
    
    print(f"{'='*60}")
    print("🚀 즉시 실행 가능한 해결책")
    print(f"{'='*60}")
    
    actions = [
        {
            "action": "데이터 개수 증가",
            "description": "LABEL_MAX_SAMPLES_PER_CLASS를 20에서 50으로 증가",
            "expected_effect": "더 많은 데이터로 안정적 학습",
            "priority": "높음"
        },
        {
            "action": "증강 강화",
            "description": "AUGMENTATIONS_PER_VIDEO를 3에서 5로 증가",
            "expected_effect": "더 다양한 증강 데이터",
            "priority": "높음"
        },
        {
            "action": "배치 크기 조정",
            "description": "BATCH_SIZE를 8에서 16으로 증가",
            "expected_effect": "더 안정적인 그래디언트",
            "priority": "중간"
        },
        {
            "action": "모델 복잡도 증가",
            "description": "LSTM 유닛 수 증가 (128→256, 64→128)",
            "expected_effect": "더 강력한 특징 학습",
            "priority": "중간"
        }
    ]
    
    for i, action in enumerate(actions, 1):
        print(f"{i}. {action['action']} ({action['priority']})")
        print(f"   설명: {action['description']}")
        print(f"   예상 효과: {action['expected_effect']}")
        print()

if __name__ == "__main__":
    epochs, train_acc, train_loss, val_acc, val_loss = analyze_training_logs()
    identify_problems()
    suggest_solutions()
    total_samples = analyze_data_distribution()
    recommend_immediate_actions()
    
    print(f"{'='*60}")
    print("✅ 분석 완료!")
    print(f"{'='*60}")
    
    print(f"\n🎯 핵심 문제점:")
    print(f"1. 데이터가 너무 적음 ({total_samples}개)")
    print(f"2. 과적합 징후 (검증 정확도 > 훈련 정확도)")
    print(f"3. 학습 불안정성")
    
    print(f"\n🚀 권장 해결책:")
    print(f"1. 데이터 개수 증가 (20 → 50개/클래스)")
    print(f"2. 증강 강화 (3 → 5개/비디오)")
    print(f"3. 모델 복잡도 증가") 