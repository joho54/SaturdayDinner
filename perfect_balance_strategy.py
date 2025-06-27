import numpy as np
from collections import defaultdict

def calculate_adaptive_augmentations(label_counts, target_count=100, min_aug=1, max_aug=10):
    """
    라벨별 적응적 증강 수 계산
    
    Args:
        label_counts: 라벨별 원본 파일 수
        target_count: 목표 샘플 수
        min_aug: 최소 증강 수
        max_aug: 최대 증강 수
    
    Returns:
        라벨별 증강 수 딕셔너리
    """
    augmentations = {}
    
    for label, count in label_counts.items():
        if count == 0:
            augmentations[label] = 0
            continue
            
        if count >= target_count:
            # 충분한 데이터가 있으면 최소 증강만
            augmentations[label] = min_aug
        else:
            # 부족한 만큼 증강 (최소/최대 제한 적용)
            needed_aug = max(1, (target_count - count) // count)
            augmentations[label] = min(max_aug, max(min_aug, needed_aug))
    
    return augmentations

def calculate_perfect_balance_augmentations(label_counts, target_count=100):
    """
    완전 균형을 위한 증강 수 계산
    
    Args:
        label_counts: 라벨별 원본 파일 수
        target_count: 목표 샘플 수
    
    Returns:
        라벨별 증강 수 딕셔너리
    """
    augmentations = {}
    
    for label, count in label_counts.items():
        if count == 0:
            # None 클래스는 특별 처리
            augmentations[label] = 0
            continue
            
        if count >= target_count:
            # 충분한 데이터가 있으면 증강 없음
            augmentations[label] = 0
        else:
            # 정확히 목표 수에 맞춤
            needed_aug = (target_count - count) // count
            if (target_count - count) % count > 0:
                needed_aug += 1  # 올림 처리
            augmentations[label] = needed_aug
    
    return augmentations

def calculate_synthetic_generation(label_counts, target_count=100):
    """
    합성 샘플 생성으로 완전 균형 달성
    
    Args:
        label_counts: 라벨별 원본 파일 수
        target_count: 목표 샘플 수
    
    Returns:
        라벨별 합성 샘플 수 딕셔너리
    """
    synthetic_counts = {}
    
    for label, count in label_counts.items():
        if count == 0:
            synthetic_counts[label] = target_count  # None 클래스는 모두 합성
        elif count >= target_count:
            synthetic_counts[label] = 0  # 증강 불필요
        else:
            synthetic_counts[label] = target_count - count  # 부족한 만큼 합성
    
    return synthetic_counts

def analyze_perfect_balance_strategies():
    """완전 균형 전략들 분석"""
    
    # 예시 데이터
    example_counts = {
        "화재": 15,
        "화요일": 8, 
        "화약": 12,
        "화상": 5,
        "팔": 20,
        "목": 18,
        "등": 10,
        "배": 7,
        "손목": 25,
        "None": 0
    }
    
    target_count = 80
    
    print("🔍 완전 균형 전략 분석")
    print("=" * 60)
    
    # 전략 1: 적응적 증강
    adaptive_aug = calculate_adaptive_augmentations(example_counts, target_count)
    adaptive_final = {}
    for label, count in example_counts.items():
        adaptive_final[label] = count * (1 + adaptive_aug[label])
    
    # 전략 2: 완전 균형 증강
    perfect_aug = calculate_perfect_balance_augmentations(example_counts, target_count)
    perfect_final = {}
    for label, count in example_counts.items():
        perfect_final[label] = count * (1 + perfect_aug[label])
    
    # 전략 3: 합성 샘플 생성
    synthetic_counts = calculate_synthetic_generation(example_counts, target_count)
    synthetic_final = {}
    for label, count in example_counts.items():
        synthetic_final[label] = count + synthetic_counts[label]
    
    # 결과 출력
    strategies = {
        "적응적 증강": adaptive_final,
        "완전 균형 증강": perfect_final,
        "합성 샘플 생성": synthetic_final
    }
    
    for strategy_name, final_counts in strategies.items():
        print(f"\n📋 {strategy_name}:")
        
        total_samples = sum(final_counts.values())
        min_samples = min([c for c in final_counts.values() if c > 0]) if any(c > 0 for c in final_counts.values()) else 0
        max_samples = max(final_counts.values())
        
        print(f"   총 샘플 수: {total_samples}")
        print(f"   최소 샘플 수: {min_samples}")
        print(f"   최대 샘플 수: {max_samples}")
        
        if min_samples > 0:
            imbalance_ratio = max_samples / min_samples
            print(f"   불균형 비율: {imbalance_ratio:.2f}:1")
        else:
            print(f"   불균형 비율: 무한대")
        
        # 각 라벨별 상세 정보
        for label, count in final_counts.items():
            if label == "None":
                print(f"   {label}: {count}개 (합성)")
            else:
                original = example_counts[label]
                if count > original:
                    aug_count = count - original
                    print(f"   {label}: {count}개 (원본 {original} + 증강 {aug_count})")
                else:
                    print(f"   {label}: {count}개 (원본만)")
    
    return strategies

def compare_strategy_efficiency():
    """전략별 효율성 비교"""
    
    print(f"\n{'='*60}")
    print("📈 전략별 효율성 비교")
    print(f"{'='*60}")
    
    # 효율성 지표 계산
    example_counts = {
        "화재": 15, "화요일": 8, "화약": 12, "화상": 5,
        "팔": 20, "목": 18, "등": 10, "배": 7, "손목": 25, "None": 0
    }
    
    target_count = 80
    
    # 각 전략의 효율성 계산
    strategies = {
        "고정 증강": {"aug_per_video": 3},
        "적응적 증강": {"target": target_count, "min_aug": 1, "max_aug": 10},
        "완전 균형 증강": {"target": target_count},
        "합성 샘플 생성": {"target": target_count}
    }
    
    efficiency_metrics = {}
    
    for strategy_name, params in strategies.items():
        if strategy_name == "고정 증강":
            # 고정 증강
            total_original = sum(example_counts.values())
            total_augmented = total_original * (1 + params["aug_per_video"])
            balance_score = 0  # 균형 점수 (낮을수록 좋음)
            
        elif strategy_name == "적응적 증강":
            # 적응적 증강
            aug_counts = calculate_adaptive_augmentations(
                example_counts, params["target"], params["min_aug"], params["max_aug"]
            )
            final_counts = {}
            for label, count in example_counts.items():
                final_counts[label] = count * (1 + aug_counts[label])
            
            total_augmented = sum(final_counts.values())
            counts_list = [c for c in final_counts.values() if c > 0]
            balance_score = np.std(counts_list) / np.mean(counts_list) if counts_list else float('inf')
            
        elif strategy_name == "완전 균형 증강":
            # 완전 균형 증강
            aug_counts = calculate_perfect_balance_augmentations(example_counts, params["target"])
            final_counts = {}
            for label, count in example_counts.items():
                final_counts[label] = count * (1 + aug_counts[label])
            
            total_augmented = sum(final_counts.values())
            counts_list = [c for c in final_counts.values() if c > 0]
            balance_score = np.std(counts_list) / np.mean(counts_list) if counts_list else float('inf')
            
        elif strategy_name == "합성 샘플 생성":
            # 합성 샘플 생성
            synthetic_counts = calculate_synthetic_generation(example_counts, params["target"])
            final_counts = {}
            for label, count in example_counts.items():
                final_counts[label] = count + synthetic_counts[label]
            
            total_augmented = sum(final_counts.values())
            counts_list = [c for c in final_counts.values() if c > 0]
            balance_score = np.std(counts_list) / np.mean(counts_list) if counts_list else float('inf')
        
        efficiency_metrics[strategy_name] = {
            "total_samples": total_augmented,
            "balance_score": balance_score,
            "efficiency": total_augmented / sum(example_counts.values())  # 원본 대비 증가율
        }
    
    # 결과 출력
    print(f"{'전략':<15} {'총 샘플':<10} {'균형 점수':<12} {'효율성':<10}")
    print("-" * 50)
    
    for strategy, metrics in efficiency_metrics.items():
        balance_str = f"{metrics['balance_score']:.3f}" if metrics['balance_score'] != float('inf') else "∞"
        print(f"{strategy:<15} {metrics['total_samples']:<10} {balance_str:<12} {metrics['efficiency']:.2f}x")
    
    # 최적 전략 추천
    print(f"\n🏆 전략 추천:")
    
    # 균형 점수 기준 (낮을수록 좋음)
    valid_metrics = {k: v for k, v in efficiency_metrics.items() 
                    if v['balance_score'] != float('inf')}
    
    if valid_metrics:
        best_balance = min(valid_metrics.keys(), 
                          key=lambda x: valid_metrics[x]['balance_score'])
        print(f"   최고 균형: {best_balance} (균형 점수: {valid_metrics[best_balance]['balance_score']:.3f})")
    
    # 효율성 기준 (낮을수록 좋음)
    best_efficiency = min(efficiency_metrics.keys(),
                         key=lambda x: efficiency_metrics[x]['efficiency'])
    print(f"   최고 효율성: {best_efficiency} (효율성: {efficiency_metrics[best_efficiency]['efficiency']:.2f}x)")

if __name__ == "__main__":
    # 완전 균형 전략 분석
    strategies = analyze_perfect_balance_strategies()
    
    # 효율성 비교
    compare_strategy_efficiency()
    
    print(f"\n✅ 분석 완료!")
    print("\n📋 핵심 결론:")
    print("   1. 적응적 증강: 불균형 완화하지만 완전 균형은 아님")
    print("   2. 완전 균형 증강: 균형 달성하지만 과도한 증강 가능성")
    print("   3. 합성 샘플 생성: 완전 균형 + 효율성, 하지만 품질 이슈")
    print("   4. None 클래스: 특별 처리 필요 (합성 샘플 또는 별도 로직)") 