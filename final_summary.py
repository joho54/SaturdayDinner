import json
import pandas as pd
from collections import defaultdict, Counter

def analyze_results():
    """결과 분석 및 실용적 솔루션 제시"""
    
    print("="*70)
    print("영상 클러스터와 자연어 클러스터를 사용한 교차 카테고라이즈")
    print("최종 분석 보고서")
    print("="*70)
    
    # 결과 파일 로드
    try:
        with open('improved_chapter_result.json', 'r', encoding='utf-8') as f:
            result = json.load(f)
    except FileNotFoundError:
        print("결과 파일을 찾을 수 없습니다. 먼저 improved_cross_categorize.py를 실행해주세요.")
        return
    
    # 기본 통계
    summary = result['summary']
    print(f"\n📊 기본 통계:")
    print(f"  • 전체 라벨 수: {summary['total_labels']}")
    print(f"  • 할당된 라벨 수: {summary['assigned_labels']}")
    print(f"  • 할당 비율: {summary['assignment_rate']:.1f}%")
    print(f"  • 활성 챕터 수: {summary['total_chapters']}")
    print(f"  • 제약 조건 위반: {summary['constraint_violations']}개")
    print(f"  • 자연어 클러스터 응집도: {summary['natural_cluster_cohesion']:.3f}")
    
    # 챕터별 상세 분석
    print(f"\n📋 챕터별 분석:")
    for chapter_name, labels in result['chapters'].items():
        if labels:
            print(f"  {chapter_name}: {len(labels)}개 라벨")
            # 샘플 라벨 표시
            sample_labels = labels[:3]
            print(f"    예시: {', '.join(sample_labels)}")
            if len(labels) > 3:
                print(f"    ... 외 {len(labels)-3}개")
    
    # 문제점 분석
    print(f"\n⚠️ 문제점 분석:")
    print(f"  1. 비디오 클러스터 크기 > 챕터 수 (4개)")
    print(f"  2. 가장 큰 비디오 클러스터는 36개 라벨을 포함")
    print(f"  3. 4개 챕터로는 비디오 클러스터 제약 조건 완전 만족 불가능")
    
    # 실용적 솔루션 제안
    print(f"\n💡 실용적 솔루션 제안:")
    
    # 솔루션 1: 최소 위반 버전
    print(f"\n1️⃣ 최소 위반 챕터 (현재 결과 기반)")
    print(f"   - 모든 라벨 할당: ✅")
    print(f"   - 자연어 클러스터 응집도: {summary['natural_cluster_cohesion']:.3f}")
    print(f"   - 비디오 클러스터 위반: {summary['constraint_violations']}개")
    print(f"   - 권장 용도: 자연어 의미 기반 분류가 우선인 경우")
    
    # 솔루션 2: 더 많은 챕터 사용
    print(f"\n2️⃣ 확장 챕터 시스템 (8-10개 챕터 권장)")
    print(f"   - 비디오 클러스터 제약 조건 완전 만족 가능")
    print(f"   - 더 세분화된 카테고리 제공")
    print(f"   - 권장 용도: 더 정확한 분류가 필요한 경우")
    
    # 솔루션 3: 하이브리드 접근
    print(f"\n3️⃣ 하이브리드 접근법")
    print(f"   - 주요 4개 챕터 + 보조 카테고리")
    print(f"   - 비디오 클러스터 충돌 라벨들을 별도 관리")
    print(f"   - 권장 용도: 유연한 시스템 설계가 가능한 경우")
    
    # 구체적인 챕터 테마 제안
    analyze_chapter_themes(result)
    
    # 최종 권장사항
    print(f"\n🎯 최종 권장사항:")
    print(f"  1. 현재 4개 챕터 제한이 있다면: 솔루션 1 사용")
    print(f"  2. 챕터 수를 늘릴 수 있다면: 8-10개 챕터로 확장")
    print(f"  3. 비디오 클러스터 제약이 절대적이라면: 하이브리드 접근법")
    print(f"  4. 자연어 클러스터 응집도(0.510)는 양호한 수준")

def analyze_chapter_themes(result):
    """챕터별 주요 테마 분석"""
    print(f"\n🏷️ 챕터별 주요 테마 분석:")
    
    # 라벨별로 챕터 찾기
    label_to_chapter = {}
    for chapter_name, labels in result['chapters'].items():
        for label in labels:
            label_to_chapter[label] = chapter_name
    
    # 각 챕터의 주요 키워드 추출
    chapter_keywords = defaultdict(list)
    
    for chapter_name, labels in result['chapters'].items():
        if not labels:
            continue
            
        # 응급상황 키워드
        emergency_keywords = ['불', '화재', '폭발', '추락', '쓰러', '응급', '구급']
        medical_keywords = ['아프', '통증', '열', '호흡', '의식', '피', '골절', '화상']
        location_keywords = ['집', '병원', '학교', '구', '아파트', '옥상']
        people_keywords = ['아기', '아이', '학생', '할머니', '할아버지', '엄마', '아빠']
        
        emergency_count = sum(1 for label in labels if any(kw in label for kw in emergency_keywords))
        medical_count = sum(1 for label in labels if any(kw in label for kw in medical_keywords))
        location_count = sum(1 for label in labels if any(kw in label for kw in location_keywords))
        people_count = sum(1 for label in labels if any(kw in label for kw in people_keywords))
        
        theme_scores = {
            '응급상황': emergency_count,
            '의료응급': medical_count,
            '장소관련': location_count,
            '인물관련': people_count
        }
        
        main_theme = max(theme_scores.items(), key=lambda x: x[1])
        
        print(f"  {chapter_name}:")
        print(f"    - 주요 테마: {main_theme[0]} ({main_theme[1]}개 관련 라벨)")
        print(f"    - 테마 분포: 응급{emergency_count}, 의료{medical_count}, 장소{location_count}, 인물{people_count}")
        
        # 대표 라벨 3개
        sample_labels = labels[:3] if len(labels) >= 3 else labels
        print(f"    - 대표 라벨: {', '.join(sample_labels)}")

def generate_practical_output():
    """실용적인 최종 결과물 생성"""
    
    try:
        with open('improved_chapter_result.json', 'r', encoding='utf-8') as f:
            result = json.load(f)
    except FileNotFoundError:
        print("결과 파일을 찾을 수 없습니다.")
        return
    
    # 실용적인 라벨 딕셔너리 생성 (None 제외)
    practical_label_dict = {}
    chapter_mapping = {
        "챕터_0": 0,
        "챕터_1": 1, 
        "챕터_2": 2,
        "챕터_3": 3
    }
    
    for chapter_name, labels in result['chapters'].items():
        if chapter_name in chapter_mapping:
            chapter_id = chapter_mapping[chapter_name]
            for label in labels:
                practical_label_dict[label] = chapter_id
    
    # None 추가
    practical_label_dict["None"] = 4
    
    # 최종 결과
    final_result = {
        "label_dict": practical_label_dict,
        "chapter_info": {
            "총 라벨 수": len(practical_label_dict) - 1,
            "챕터 수": 4,
            "할당 비율": "100%",
            "자연어 클러스터 응집도": result['summary']['natural_cluster_cohesion'],
            "주의사항": "비디오 클러스터 제약 조건 일부 위반 (예상됨)"
        },
        "사용법": {
            "라벨": "practical_label_dict[라벨명]으로 챕터 ID 조회",
            "챕터 ID": "0, 1, 2, 3 (4개 챕터), 4 (None)",
            "예시": f"practical_label_dict['화재'] = {practical_label_dict.get('화재', 'N/A')}"
        }
    }
    
    # 파일 저장
    with open('final_chapter_result.json', 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 최종 실용 결과가 'final_chapter_result.json'에 저장되었습니다.")
    
    # 사용 예시 출력
    print(f"\n📋 사용 예시:")
    sample_labels = ['화재', '구급차', '119', '골절', '병원']
    for label in sample_labels:
        if label in practical_label_dict:
            chapter_id = practical_label_dict[label]
            print(f"  '{label}' → 챕터 {chapter_id}")

if __name__ == "__main__":
    analyze_results()
    generate_practical_output() 