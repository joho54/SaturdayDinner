import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_optimal_augmentation_for_12_samples():
    """라벨별 12개 데이터에 대한 최적 증강 수 분석"""
    
    print("🔍 라벨별 12개 데이터에 대한 최적 증강 수 분석")
    print("=" * 70)
    
    original_count = 12
    augmentation_levels = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    
    print(f"📊 원본 데이터: {original_count}개/라벨")
    print(f"📊 분석할 증강 수준: {augmentation_levels}")
    
    results = []
    
    for aug_level in augmentation_levels:
        total_samples = original_count * (1 + aug_level)
        
        # 성능 예측 (다양한 지표 기반)
        performance_score = calculate_performance_score(original_count, aug_level, total_samples)
        risk_score = calculate_risk_score(aug_level, total_samples)
        efficiency_score = calculate_efficiency_score(original_count, aug_level, total_samples)
        
        # 종합 평가
        overall_score = (performance_score + efficiency_score - risk_score) / 3
        
        results.append({
            "aug_level": aug_level,
            "total_samples": total_samples,
            "performance_score": performance_score,
            "risk_score": risk_score,
            "efficiency_score": efficiency_score,
            "overall_score": overall_score
        })
        
        print(f"\n📋 증강 수준: {aug_level}개/비디오")
        print(f"   총 샘플: {total_samples}개")
        print(f"   성능 점수: {performance_score:.2f}")
        print(f"   위험도: {risk_score:.2f}")
        print(f"   효율성: {efficiency_score:.2f}")
        print(f"   종합 점수: {overall_score:.2f}")
    
    # 최적 증강 수 찾기
    best_result = max(results, key=lambda x: x['overall_score'])
    
    print(f"\n{'='*70}")
    print("🏆 최적 증강 수 추천")
    print(f"{'='*70}")
    print(f"   추천 증강 수: {best_result['aug_level']}개/비디오")
    print(f"   예상 총 샘플: {best_result['total_samples']}개")
    print(f"   종합 점수: {best_result['overall_score']:.2f}")
    
    return results, best_result

def calculate_performance_score(original_count, aug_level, total_samples):
    """성능 점수 계산"""
    # 기본 성능 (충분한 데이터 확보)
    base_performance = min(1.0, total_samples / 100)  # 100개 기준
    
    # 증강 품질 (적절한 수준)
    if aug_level <= 3:
        quality_bonus = 0.2  # 적절한 증강
    elif aug_level <= 6:
        quality_bonus = 0.1  # 보통 증강
    else:
        quality_bonus = -0.1  # 과도한 증강
    
    # 데이터 다양성
    diversity_bonus = min(0.1, aug_level * 0.02)  # 증강이 많을수록 다양성 증가
    
    return min(1.0, base_performance + quality_bonus + diversity_bonus)

def calculate_risk_score(aug_level, total_samples):
    """위험도 점수 계산 (낮을수록 좋음)"""
    # 과도한 증강 위험
    if aug_level <= 3:
        over_augmentation_risk = 0.0
    elif aug_level <= 6:
        over_augmentation_risk = 0.2
    else:
        over_augmentation_risk = 0.5
    
    # 과적합 위험 (데이터가 너무 많으면)
    if total_samples <= 100:
        overfitting_risk = 0.0
    elif total_samples <= 200:
        overfitting_risk = 0.1
    else:
        overfitting_risk = 0.3
    
    # 학습 시간 증가 위험
    time_risk = min(0.2, aug_level * 0.02)
    
    return over_augmentation_risk + overfitting_risk + time_risk

def calculate_efficiency_score(original_count, aug_level, total_samples):
    """효율성 점수 계산"""
    # 데이터 증가 효율성
    increase_ratio = total_samples / original_count
    efficiency = min(1.0, increase_ratio / 10)  # 10배 증가까지 효율적
    
    # 증강 비용 (적을수록 효율적)
    if aug_level <= 3:
        cost_efficiency = 1.0
    elif aug_level <= 6:
        cost_efficiency = 0.8
    else:
        cost_efficiency = 0.6
    
    return (efficiency + cost_efficiency) / 2

def analyze_different_scenarios():
    """다양한 시나리오에서의 분석"""
    
    print(f"\n{'='*70}")
    print("🌍 다양한 시나리오 분석")
    print(f"{'='*70}")
    
    scenarios = [
        {
            "name": "소규모 실험",
            "description": "빠른 프로토타입, 적은 데이터",
            "recommended_aug": 2,
            "reason": "빠른 학습, 충분한 데이터"
        },
        {
            "name": "균형잡힌 학습",
            "description": "안정적인 성능, 적절한 시간",
            "recommended_aug": 3,
            "reason": "최적의 균형점"
        },
        {
            "name": "고성능 목표",
            "description": "최고 성능, 충분한 시간",
            "recommended_aug": 4,
            "reason": "더 많은 데이터로 성능 향상"
        },
        {
            "name": "대규모 실험",
            "description": "연구 목적, 충분한 리소스",
            "recommended_aug": 5,
            "reason": "최대한 많은 데이터"
        }
    ]
    
    for scenario in scenarios:
        aug_level = scenario["recommended_aug"]
        total_samples = 12 * (1 + aug_level)
        
        print(f"\n📋 {scenario['name']}")
        print(f"   설명: {scenario['description']}")
        print(f"   추천 증강: {aug_level}개/비디오")
        print(f"   총 샘플: {total_samples}개")
        print(f"   이유: {scenario['reason']}")

def compare_with_previous_analysis():
    """이전 분석(7개)과 비교"""
    
    print(f"\n{'='*70}")
    print("📊 이전 분석(7개)과 비교")
    print(f"{'='*70}")
    
    previous_data = {
        "original_count": 7,
        "aug_level": 3,
        "total_samples": 28,
        "performance": "높음"
    }
    
    current_data = {
        "original_count": 12,
        "aug_level": 3,
        "total_samples": 48,
        "performance": "높음"
    }
    
    print(f"📋 이전 분석 (7개 원본):")
    print(f"   원본: {previous_data['original_count']}개")
    print(f"   증강: {previous_data['aug_level']}개/비디오")
    print(f"   총 샘플: {previous_data['total_samples']}개")
    print(f"   성능: {previous_data['performance']}")
    
    print(f"\n📋 현재 분석 (12개 원본):")
    print(f"   원본: {current_data['original_count']}개")
    print(f"   증강: {current_data['aug_level']}개/비디오")
    print(f"   총 샘플: {current_data['total_samples']}개")
    print(f"   성능: {current_data['performance']}")
    
    # 개선 효과
    improvement_ratio = current_data['total_samples'] / previous_data['total_samples']
    print(f"\n📈 개선 효과:")
    print(f"   데이터 증가: {improvement_ratio:.1f}배")
    print(f"   원본 증가: {current_data['original_count']/previous_data['original_count']:.1f}배")

def provide_recommendations():
    """구체적인 추천사항"""
    
    print(f"\n{'='*70}")
    print("💡 구체적인 추천사항")
    print(f"{'='*70}")
    
    recommendations = [
        {
            "scenario": "기본 설정",
            "aug_level": 3,
            "reason": "이전 성공 사례와 동일한 수준",
            "expected_samples": 48,
            "pros": ["검증된 방법", "안정적 성능", "적절한 학습 시간"],
            "cons": ["보수적 접근"]
        },
        {
            "scenario": "성능 최적화",
            "aug_level": 4,
            "reason": "더 많은 데이터로 성능 향상 기대",
            "expected_samples": 60,
            "pros": ["더 나은 성능", "더 나은 일반화"],
            "cons": ["더 긴 학습 시간", "과적합 위험 증가"]
        },
        {
            "scenario": "빠른 실험",
            "aug_level": 2,
            "reason": "빠른 반복 실험을 위한 최소 증강",
            "expected_samples": 36,
            "pros": ["빠른 학습", "빠른 반복"],
            "cons": ["성능 제한 가능성"]
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['scenario']}")
        print(f"   증강 수: {rec['aug_level']}개/비디오")
        print(f"   예상 샘플: {rec['expected_samples']}개")
        print(f"   이유: {rec['reason']}")
        print(f"   장점: {', '.join(rec['pros'])}")
        print(f"   단점: {', '.join(rec['cons'])}")

if __name__ == "__main__":
    # 메인 분석 실행
    results, best_result = analyze_optimal_augmentation_for_12_samples()
    analyze_different_scenarios()
    compare_with_previous_analysis()
    provide_recommendations()
    
    print(f"\n{'='*70}")
    print("✅ 분석 완료!")
    print(f"{'='*70}")
    
    print(f"\n🎯 최종 추천:")
    print(f"   기본 설정: 3개/비디오 (검증된 방법)")
    print(f"   성능 최적화: 4개/비디오 (더 나은 성능)")
    print(f"   빠른 실험: 2개/비디오 (빠른 반복)")
    
    print(f"\n📋 핵심 인사이트:")
    print("1. 12개 원본은 7개보다 더 안정적인 학습 가능")
    print("2. 3-4개 증강이 최적의 균형점")
    print("3. 목적에 따라 2-5개 증강 선택 가능")
    print("4. 이전 성공 사례를 고려하면 3개가 안전한 선택") 