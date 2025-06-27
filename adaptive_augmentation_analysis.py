import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

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

def simulate_data_distribution(original_counts, augmentation_strategy="fixed", target_count=100):
    """
    다양한 증강 전략에 따른 데이터 분포 시뮬레이션
    
    Args:
        original_counts: 라벨별 원본 파일 수
        augmentation_strategy: "fixed", "adaptive", "balanced"
        target_count: 목표 샘플 수 (adaptive 전략에서 사용)
    
    Returns:
        라벨별 최종 샘플 수
    """
    if augmentation_strategy == "fixed":
        # 현재 코드의 고정 증강
        aug_per_video = 3
        final_counts = {}
        for label, count in original_counts.items():
            final_counts[label] = count * (1 + aug_per_video)
    
    elif augmentation_strategy == "adaptive":
        # 적응적 증강
        aug_counts = calculate_adaptive_augmentations(original_counts, target_count)
        final_counts = {}
        for label, count in original_counts.items():
            final_counts[label] = count * (1 + aug_counts[label])
    
    elif augmentation_strategy == "balanced":
        # 완전 균형 (모든 라벨을 동일하게 맞춤)
        max_original = max(original_counts.values())
        final_counts = {}
        for label, count in original_counts.items():
            if count == 0:
                final_counts[label] = 0
            else:
                # 가장 많은 라벨에 맞춰서 증강
                needed_aug = max(1, (max_original - count) // count)
                final_counts[label] = count * (1 + needed_aug)
    
    return final_counts

def analyze_strategies():
    """다양한 증강 전략 분석"""
    
    # 예시 데이터 (실제 데이터와 유사한 분포)
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
        "None": 0  # None은 특별 처리
    }
    
    print("📊 원본 라벨별 파일 수:")
    for label, count in example_counts.items():
        print(f"   {label}: {count}개")
    
    print(f"\n{'='*60}")
    print("🔍 다양한 증강 전략 비교")
    print(f"{'='*60}")
    
    strategies = ["fixed", "adaptive", "balanced"]
    results = {}
    
    for strategy in strategies:
        print(f"\n📋 {strategy.upper()} 전략:")
        
        if strategy == "adaptive":
            final_counts = simulate_data_distribution(example_counts, strategy, target_count=80)
        else:
            final_counts = simulate_data_distribution(example_counts, strategy)
        
        results[strategy] = final_counts
        
        total_samples = sum(final_counts.values())
        min_samples = min(final_counts.values()) if final_counts.values() else 0
        max_samples = max(final_counts.values()) if final_counts.values() else 0
        
        print(f"   총 샘플 수: {total_samples}")
        print(f"   최소 샘플 수: {min_samples}")
        print(f"   최대 샘플 수: {max_samples}")
        print(f"   불균형 비율: {max_samples/min_samples:.2f}:1" if min_samples > 0 else "   불균형 비율: 무한대")
        
        for label, count in final_counts.items():
            print(f"   {label}: {count}개")
    
    return example_counts, results

def visualize_comparison(original_counts, results):
    """결과 시각화"""
    
    # 데이터 준비
    labels = list(original_counts.keys())
    x = np.arange(len(labels))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 원본 vs 전략별 비교
    ax1.bar(x - width, [original_counts[label] for label in labels], 
            width, label='원본', alpha=0.8)
    ax1.bar(x, [results['fixed'][label] for label in labels], 
            width, label='고정 증강', alpha=0.8)
    ax1.bar(x + width, [results['adaptive'][label] for label in labels], 
            width, label='적응적 증강', alpha=0.8)
    
    ax1.set_xlabel('라벨')
    ax1.set_ylabel('샘플 수')
    ax1.set_title('증강 전략별 라벨별 샘플 수 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 불균형 지수 비교
    strategies = ['fixed', 'adaptive', 'balanced']
    imbalance_ratios = []
    
    for strategy in strategies:
        counts = list(results[strategy].values())
        if min(counts) > 0:
            ratio = max(counts) / min(counts)
        else:
            ratio = float('inf')
        imbalance_ratios.append(ratio)
    
    ax2.bar(strategies, imbalance_ratios, alpha=0.8, color=['red', 'orange', 'green'])
    ax2.set_ylabel('불균형 비율 (최대/최소)')
    ax2.set_title('전략별 불균형 지수 비교')
    ax2.grid(True, alpha=0.3)
    
    # 값 표시
    for i, ratio in enumerate(imbalance_ratios):
        if ratio != float('inf'):
            ax2.text(i, ratio + 0.1, f'{ratio:.1f}', ha='center', va='bottom')
        else:
            ax2.text(i, 10, '∞', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('augmentation_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_metrics(original_counts, results):
    """정량적 지표 계산"""
    
    print(f"\n{'='*60}")
    print("📈 정량적 지표 비교")
    print(f"{'='*60}")
    
    metrics = {}
    
    for strategy, final_counts in results.items():
        counts = list(final_counts.values())
        counts = [c for c in counts if c > 0]  # 0 제외
        
        if not counts:
            continue
            
        total = sum(counts)
        mean = np.mean(counts)
        std = np.std(counts)
        cv = std / mean if mean > 0 else 0  # 변동계수
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        metrics[strategy] = {
            'total': total,
            'mean': mean,
            'std': std,
            'cv': cv,
            'min': min_count,
            'max': max_count,
            'imbalance_ratio': imbalance_ratio
        }
        
        print(f"\n📊 {strategy.upper()} 전략:")
        print(f"   총 샘플 수: {total}")
        print(f"   평균: {mean:.1f}")
        print(f"   표준편차: {std:.1f}")
        print(f"   변동계수: {cv:.3f}")
        print(f"   최소/최대: {min_count}/{max_count}")
        print(f"   불균형 비율: {imbalance_ratio:.2f}:1" if imbalance_ratio != float('inf') else "   불균형 비율: 무한대")
    
    return metrics

def recommend_strategy(metrics):
    """최적 전략 추천"""
    
    print(f"\n{'='*60}")
    print("🎯 전략 추천")
    print(f"{'='*60}")
    
    # 불균형 비율 기준으로 정렬
    valid_metrics = {k: v for k, v in metrics.items() if v['imbalance_ratio'] != float('inf')}
    
    if not valid_metrics:
        print("❌ 모든 전략에서 극단적 불균형 발생")
        return
    
    # 가장 균형잡힌 전략 찾기
    best_strategy = min(valid_metrics.keys(), 
                       key=lambda x: valid_metrics[x]['imbalance_ratio'])
    
    print(f"🏆 추천 전략: {best_strategy.upper()}")
    print(f"   불균형 비율: {valid_metrics[best_strategy]['imbalance_ratio']:.2f}:1")
    print(f"   변동계수: {valid_metrics[best_strategy]['cv']:.3f}")
    
    # 각 전략의 장단점
    print(f"\n📋 전략별 특징:")
    for strategy, metric in metrics.items():
        print(f"\n   {strategy.upper()}:")
        if strategy == 'fixed':
            print("     ✅ 구현 간단, 일관된 증강")
            print("     ❌ 불균형 해결 안됨")
        elif strategy == 'adaptive':
            print("     ✅ 불균형 완화, 효율적 증강")
            print("     ❌ 구현 복잡도 증가")
        elif strategy == 'balanced':
            print("     ✅ 완전 균형 달성")
            print("     ❌ 과도한 증강 가능성")

if __name__ == "__main__":
    print("🔍 적응적 증강 전략 분석")
    print("=" * 60)
    
    # 분석 실행
    original_counts, results = analyze_strategies()
    
    # 시각화
    try:
        visualize_comparison(original_counts, results)
    except Exception as e:
        print(f"⚠️ 시각화 실패: {e}")
    
    # 정량적 지표
    metrics = calculate_metrics(original_counts, results)
    
    # 전략 추천
    recommend_strategy(metrics)
    
    print(f"\n✅ 분석 완료!")
    print("📊 결과 요약:")
    print("   - 고정 증강: 모든 라벨에 동일한 증강 적용")
    print("   - 적응적 증강: 부족한 라벨에 더 많은 증강 적용")
    print("   - 균형 증강: 모든 라벨을 동일한 수로 맞춤") 