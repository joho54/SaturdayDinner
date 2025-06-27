import numpy as np
from collections import Counter

def test_none_class_balance():
    """None 클래스 균형 생성 기능 테스트"""
    
    print("🧪 None 클래스 균형 생성 테스트")
    print("=" * 50)
    
    # 테스트 시나리오
    test_scenarios = [
        {
            "name": "균등 분포",
            "file_mapping": {
                "video1": {"label": "화재", "path": "path1"},
                "video2": {"label": "화재", "path": "path2"},
                "video3": {"label": "화재", "path": "path3"},
                "video4": {"label": "화요일", "path": "path4"},
                "video5": {"label": "화요일", "path": "path5"},
                "video6": {"label": "화요일", "path": "path6"},
                "video7": {"label": "화약", "path": "path7"},
                "video8": {"label": "화약", "path": "path8"},
                "video9": {"label": "화약", "path": "path9"},
            }
        },
        {
            "name": "불균등 분포",
            "file_mapping": {
                "video1": {"label": "화재", "path": "path1"},
                "video2": {"label": "화재", "path": "path2"},
                "video3": {"label": "화요일", "path": "path3"},
                "video4": {"label": "화요일", "path": "path4"},
                "video5": {"label": "화요일", "path": "path5"},
                "video6": {"label": "화요일", "path": "path6"},
                "video7": {"label": "화약", "path": "path7"},
                "video8": {"label": "화약", "path": "path8"},
                "video9": {"label": "화약", "path": "path9"},
                "video10": {"label": "화약", "path": "path10"},
                "video11": {"label": "화약", "path": "path11"},
            }
        },
        {
            "name": "극단적 불균등",
            "file_mapping": {
                "video1": {"label": "화재", "path": "path1"},
                "video2": {"label": "화요일", "path": "path2"},
                "video3": {"label": "화요일", "path": "path3"},
                "video4": {"label": "화약", "path": "path4"},
                "video5": {"label": "화약", "path": "path5"},
                "video6": {"label": "화약", "path": "path6"},
                "video7": {"label": "화약", "path": "path7"},
                "video8": {"label": "화약", "path": "path8"},
                "video9": {"label": "화약", "path": "path9"},
                "video10": {"label": "화약", "path": "path10"},
                "video11": {"label": "화약", "path": "path11"},
                "video12": {"label": "화약", "path": "path12"},
                "video13": {"label": "화약", "path": "path13"},
                "video14": {"label": "화약", "path": "path14"},
                "video15": {"label": "화약", "path": "path15"},
            }
        }
    ]
    
    # 설정값 (config.py와 동일)
    AUGMENTATIONS_PER_VIDEO = 3
    NONE_CLASS = "None"
    
    for scenario in test_scenarios:
        print(f"\n📋 시나리오: {scenario['name']}")
        print("-" * 30)
        
        file_mapping = scenario['file_mapping']
        
        # 기존 방식 (모든 비디오 사용)
        total_videos = len(file_mapping)
        old_none_count = total_videos * 21  # 비디오당 약 21개
        
        # 새로운 방식 (균형 생성)
        other_class_counts = {}
        for filename, info in file_mapping.items():
            if info['label'] != NONE_CLASS:
                label = info['label']
                other_class_counts[label] = other_class_counts.get(label, 0) + 1
        
        if other_class_counts:
            avg_other_class_count = sum(other_class_counts.values()) / len(other_class_counts)
            new_none_count = int(avg_other_class_count * (1 + AUGMENTATIONS_PER_VIDEO))
        else:
            new_none_count = 100  # 기본값
        
        # 결과 출력
        print(f"📊 라벨별 원본 개수:")
        for label, count in other_class_counts.items():
            print(f"   {label}: {count}개")
        
        print(f"\n📈 None 클래스 생성량 비교:")
        print(f"   기존 방식: {old_none_count}개 (모든 비디오 사용)")
        print(f"   새로운 방식: {new_none_count}개 (균형 생성)")
        print(f"   개선 효과: {old_none_count/new_none_count:.1f}배 감소")
        
        # 균형 지수 계산
        if other_class_counts:
            other_class_avg = sum(other_class_counts.values()) / len(other_class_counts)
            other_class_std = np.std(list(other_class_counts.values()))
            balance_ratio_old = old_none_count / other_class_avg
            balance_ratio_new = new_none_count / other_class_avg
            
            print(f"\n⚖️ 균형 지수:")
            print(f"   기존 방식: {balance_ratio_old:.1f}:1 (None이 {balance_ratio_old:.1f}배 많음)")
            print(f"   새로운 방식: {balance_ratio_new:.1f}:1 (None이 {balance_ratio_new:.1f}배 많음)")
            print(f"   개선 효과: {balance_ratio_old/balance_ratio_new:.1f}배 균형 개선")

def simulate_real_world_scenario():
    """실제 상황 시뮬레이션"""
    
    print(f"\n{'='*60}")
    print("🌍 실제 상황 시뮬레이션")
    print(f"{'='*60}")
    
    # 실제 데이터와 유사한 분포
    real_scenario = {
        "화재": 15,
        "화요일": 8,
        "화약": 12,
        "화상": 5,
        "팔": 20,
        "목": 18,
        "등": 10,
        "배": 7,
        "손목": 25
    }
    
    print("📊 실제 라벨별 분포:")
    for label, count in real_scenario.items():
        print(f"   {label}: {count}개")
    
    # 기존 방식 계산
    total_videos = sum(real_scenario.values())
    old_none_count = total_videos * 21
    
    # 새로운 방식 계산
    avg_other_class_count = sum(real_scenario.values()) / len(real_scenario)
    new_none_count = int(avg_other_class_count * (1 + 3))  # AUGMENTATIONS_PER_VIDEO = 3
    
    print(f"\n📈 None 클래스 생성량:")
    print(f"   기존 방식: {old_none_count}개")
    print(f"   새로운 방식: {new_none_count}개")
    print(f"   개선 효과: {old_none_count/new_none_count:.1f}배 감소")
    
    # 증강 후 예상 분포
    print(f"\n🔮 증강 후 예상 분포:")
    for label, count in real_scenario.items():
        augmented_count = count * (1 + 3)  # AUGMENTATIONS_PER_VIDEO = 3
        print(f"   {label}: {augmented_count}개")
    
    print(f"   None (기존): {old_none_count}개")
    print(f"   None (새로운): {new_none_count}개")
    
    # 불균형 지수
    avg_augmented = sum([count * 4 for count in real_scenario.values()]) / len(real_scenario)
    imbalance_old = old_none_count / avg_augmented
    imbalance_new = new_none_count / avg_augmented
    
    print(f"\n⚖️ 최종 불균형 지수:")
    print(f"   기존 방식: {imbalance_old:.1f}:1")
    print(f"   새로운 방식: {imbalance_new:.1f}:1")
    print(f"   개선 효과: {imbalance_old/imbalance_new:.1f}배 균형 개선")

if __name__ == "__main__":
    test_none_class_balance()
    simulate_real_world_scenario()
    
    print(f"\n✅ 테스트 완료!")
    print("\n📋 핵심 개선사항:")
    print("   1. None 클래스가 다른 클래스와 균형있게 생성됨")
    print("   2. 과도한 None 샘플로 인한 모델 편향 방지")
    print("   3. 예측 가능한 데이터 분포")
    print("   4. 효율적인 리소스 사용") 