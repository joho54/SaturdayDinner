import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class CrossCategorizer:
    def __init__(self, max_labels_per_chapter=4):
        self.max_labels_per_chapter = max_labels_per_chapter
        self.label_cluster_data = None
        self.video_cluster_data = None
        self.merged_data = None
        self.chapters = {}
        
    def load_data(self, label_cluster_file, video_cluster_file):
        """데이터 로드 및 전처리"""
        # 라벨 자연어 임베딩 클러스터 데이터 로드
        self.label_cluster_data = pd.read_csv(label_cluster_file)
        print(f"라벨 클러스터 데이터 로드: {len(self.label_cluster_data)} 행")
        
        # 비디오 임베딩 기반 클러스터 데이터 로드
        self.video_cluster_data = pd.read_csv(video_cluster_file)
        print(f"비디오 클러스터 데이터 로드: {len(self.video_cluster_data)} 행")
        
        # 데이터 병합을 위해 파일명을 기준으로 매칭
        self._merge_data()
        
    def _merge_data(self):
        """두 데이터셋을 병합"""
        # 파일명에서 확장자 제거하여 매칭
        label_data = self.label_cluster_data.copy()
        video_data = self.video_cluster_data.copy()
        
        # 파일명 정규화 (확장자 제거)
        label_data['filename_normalized'] = label_data['파일명'].str.replace(r'\.(MOV|AVI|MP4|MTS)$', '', regex=True)
        video_data['filename_normalized'] = video_data['filename'].str.replace(r'\.(MOV|AVI|MP4|MTS)$', '', regex=True)
        
        # 병합
        self.merged_data = pd.merge(
            label_data[['filename_normalized', '한국어', '클러스터_ID', '클러스터_크기']],
            video_data[['filename_normalized', 'cluster_id']],
            on='filename_normalized',
            how='inner'
        )
        
        print(f"병합된 데이터: {len(self.merged_data)} 행")
        
        # 라벨별로 그룹화하여 클러스터 정보 정리
        self.label_info = {}
        for _, row in self.merged_data.iterrows():
            label = row['한국어']
            if label not in self.label_info:
                self.label_info[label] = {
                    'natural_cluster': row['클러스터_ID'],
                    'video_cluster': row['cluster_id']
                }
        
        print(f"고유 라벨 수: {len(self.label_info)}")
        
    def analyze_constraints(self):
        """제약 조건 분석"""
        # 자연어 클러스터별 라벨 그룹화
        natural_clusters = defaultdict(list)
        video_clusters = defaultdict(list)
        
        for label, info in self.label_info.items():
            natural_clusters[info['natural_cluster']].append(label)
            video_clusters[info['video_cluster']].append(label)
        
        print(f"\n자연어 클러스터 수: {len(natural_clusters)}")
        print(f"비디오 클러스터 수: {len(video_clusters)}")
        
        # 비디오 클러스터 크기 분석
        video_cluster_sizes = [len(labels) for labels in video_clusters.values()]
        print(f"비디오 클러스터 크기 분포: min={min(video_cluster_sizes)}, max={max(video_cluster_sizes)}, avg={np.mean(video_cluster_sizes):.2f}")
        
        # 자연어 클러스터 크기 분석
        natural_cluster_sizes = [len(labels) for labels in natural_clusters.values()]
        print(f"자연어 클러스터 크기 분포: min={min(natural_cluster_sizes)}, max={max(natural_cluster_sizes)}, avg={np.mean(natural_cluster_sizes):.2f}")
        
        # 최대 자연어 클러스터 크기가 챕터 최대 라벨 수를 초과하는지 확인
        max_natural_cluster_size = max(natural_cluster_sizes)
        if max_natural_cluster_size > self.max_labels_per_chapter:
            print(f"⚠️ 경고: 최대 자연어 클러스터 크기({max_natural_cluster_size})가 챕터당 최대 라벨 수({self.max_labels_per_chapter})를 초과합니다.")
        
        return natural_clusters, video_clusters
    
    def create_chapters(self):
        """새로운 챕터 생성 알고리즘 - 각 챕터당 최대 4개 라벨"""
        natural_clusters, video_clusters = self.analyze_constraints()
        
        # 모든 라벨 수집
        all_labels = list(self.label_info.keys())
        
        # 챕터 리스트 (각 챕터는 라벨 리스트)
        chapters = []
        
        # 라벨별 할당 상태 추적
        label_to_chapter = {}
        
        print("\n1단계: 자연어 클러스터 우선 배치...")
        
        # 자연어 클러스터를 크기 순으로 정렬 (큰 것부터)
        sorted_natural_clusters = sorted(natural_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        for natural_cluster_id, natural_labels in sorted_natural_clusters:
            print(f"자연어 클러스터 {natural_cluster_id}: {len(natural_labels)}개 라벨 처리")
            
            # 이미 할당된 라벨들 제외
            unassigned_labels = [label for label in natural_labels if label not in label_to_chapter]
            
            if not unassigned_labels:
                continue
            
            # 자연어 클러스터의 라벨들을 가능한 한 같은 챕터에 배치
            while unassigned_labels:
                # 새 챕터 생성
                new_chapter = []
                labels_to_assign = []
                
                # 최대 4개까지 라벨 선택 (비디오 클러스터 제약 조건 고려)
                for label in unassigned_labels[:]:
                    if len(new_chapter) >= self.max_labels_per_chapter:
                        break
                    
                    # 비디오 클러스터 제약 조건 확인
                    video_cluster_id = self.label_info[label]['video_cluster']
                    
                    # 현재 챕터에 같은 비디오 클러스터의 다른 라벨이 있는지 확인
                    conflict = False
                    for existing_label in new_chapter:
                        if self.label_info[existing_label]['video_cluster'] == video_cluster_id:
                            conflict = True
                            break
                    
                    if not conflict:
                        new_chapter.append(label)
                        labels_to_assign.append(label)
                
                # 챕터에 라벨 할당
                if new_chapter:
                    chapter_id = len(chapters)
                    chapters.append(new_chapter)
                    
                    for label in labels_to_assign:
                        label_to_chapter[label] = chapter_id
                        unassigned_labels.remove(label)
                        
                    print(f"  새 챕터 {chapter_id} 생성: {len(new_chapter)}개 라벨")
                else:
                    # 할당할 수 있는 라벨이 없으면 중단
                    break
        
        print(f"\n2단계: 남은 라벨들 개별 배치...")
        
        # 아직 할당되지 않은 라벨들 처리
        unassigned_labels = [label for label in all_labels if label not in label_to_chapter]
        print(f"남은 라벨 수: {len(unassigned_labels)}")
        
        for label in unassigned_labels:
            video_cluster_id = self.label_info[label]['video_cluster']
            
            # 기존 챕터 중에서 배치 가능한 곳 찾기
            placed = False
            
            for chapter_id, chapter_labels in enumerate(chapters):
                # 챕터가 가득 찼으면 스킵
                if len(chapter_labels) >= self.max_labels_per_chapter:
                    continue
                
                # 비디오 클러스터 충돌 확인
                conflict = False
                for existing_label in chapter_labels:
                    if self.label_info[existing_label]['video_cluster'] == video_cluster_id:
                        conflict = True
                        break
                
                if not conflict:
                    chapters[chapter_id].append(label)
                    label_to_chapter[label] = chapter_id
                    placed = True
                    print(f"  라벨 '{label}' -> 챕터 {chapter_id}")
                    break
            
            # 기존 챕터에 배치할 수 없으면 새 챕터 생성
            if not placed:
                new_chapter_id = len(chapters)
                chapters.append([label])
                label_to_chapter[label] = new_chapter_id
                print(f"  라벨 '{label}' -> 새 챕터 {new_chapter_id}")
        
        # 결과를 딕셔너리 형태로 변환
        self.chapters = {}
        for i, chapter_labels in enumerate(chapters):
            chapter_name = f"chapter_{i+1}"
            self.chapters[chapter_name] = {
                "label_dict": {label: idx for idx, label in enumerate(chapter_labels)},
                "labels": chapter_labels
            }
        
        # 모든 챕터에 None 라벨을 마지막 요소로 추가
        for chapter_name, chapter_data in self.chapters.items():
            none_idx = len(chapter_data["labels"])
            chapter_data["label_dict"]["None"] = none_idx
            chapter_data["labels"].append("None")
        
        return self.chapters
    
    def validate_constraints(self):
        """제약 조건 검증"""
        print("\n제약 조건 검증...")
        
        # 챕터별 라벨 딕셔너리 생성
        label_to_chapter = {}
        for chapter_name, chapter_data in self.chapters.items():
            for label in chapter_data["labels"]:
                if label != "None":
                    label_to_chapter[label] = chapter_name
        
        # 비디오 클러스터 제약 조건 검증
        natural_clusters, video_clusters = self.analyze_constraints()
        
        violations = []
        
        # 1. 비디오 클러스터 제약 조건 확인 (반드시 다른 챕터에 배치)
        for video_cluster_id, video_labels in video_clusters.items():
            assigned_chapters = set()
            for label in video_labels:
                if label in label_to_chapter:
                    assigned_chapters.add(label_to_chapter[label])
            
            if len(assigned_chapters) < len(video_labels):
                violations.append(f"비디오 클러스터 {video_cluster_id}: {len(video_labels)}개 라벨이 {len(assigned_chapters)}개 챕터에만 할당됨 (같은 챕터에 배치된 라벨 존재)")
        
        # 2. 챕터당 최대 라벨 수 확인
        for chapter_name, chapter_data in self.chapters.items():
            if len(chapter_data["labels"]) > self.max_labels_per_chapter:
                violations.append(f"{chapter_name}: {len(chapter_data['labels'])}개 라벨 (최대 {self.max_labels_per_chapter}개 초과)")
        
        if violations:
            print("제약 조건 위반:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("✅ 모든 제약 조건이 만족됩니다.")
        
        # 자연어 클러스터 응집도 분석
        natural_cohesion = []
        for natural_cluster_id, natural_labels in natural_clusters.items():
            chapter_distribution = defaultdict(int)
            for label in natural_labels:
                if label in label_to_chapter:
                    chapter_distribution[label_to_chapter[label]] += 1
            
            if chapter_distribution:
                max_in_one_chapter = max(chapter_distribution.values())
                cohesion = max_in_one_chapter / len(natural_labels)
                natural_cohesion.append(cohesion)
                print(f"  자연어 클러스터 {natural_cluster_id}: 응집도 {cohesion:.2f} ({max_in_one_chapter}/{len(natural_labels)})")
        
        avg_cohesion = np.mean(natural_cohesion) if natural_cohesion else 0
        print(f"\n자연어 클러스터 평균 응집도: {avg_cohesion:.2f}")
        
        return len(violations) == 0
    
    def generate_output(self):
        """최종 결과 생성"""
        result = {
            "chapters": [],
            "summary": {
                "total_chapters": len(self.chapters),
                "total_labels": sum(len(ch["labels"]) for ch in self.chapters.values()) - len(self.chapters),  # 각 챕터의 None 제외
                "chapter_sizes": {name: len(ch["labels"]) for name, ch in self.chapters.items()}
            }
        }
        
        # 각 챕터를 요구사항 형식으로 변환
        for chapter_name, chapter_data in self.chapters.items():
            chapter_result = {
                "chapter_name": chapter_name,
                "label_dict": chapter_data["label_dict"].copy()
            }
            result["chapters"].append(chapter_result)
        
        return result

def main():
    # 교차 카테고라이저 초기화 (챕터당 최대 4개 라벨)
    categorizer = CrossCategorizer(max_labels_per_chapter=4)
    
    # 데이터 로드
    categorizer.load_data(
        'two-clusters/label_clusters.csv',
        'two-clusters/video_clusters.csv'
    )
    
    # 제약 조건 분석
    categorizer.analyze_constraints()
    
    # 챕터 생성
    chapters = categorizer.create_chapters()
    
    # 제약 조건 검증
    is_valid = categorizer.validate_constraints()
    
    # 결과 생성
    result = categorizer.generate_output()
    
    # 결과 출력
    print("\n" + "="*50)
    print("챕터 생성 결과")
    print("="*50)
    
    for chapter_result in result["chapters"]:
        chapter_name = chapter_result["chapter_name"]
        label_dict = chapter_result["label_dict"]
        print(f"\n{chapter_name}:")
        print("  label_dict: {")
        for label, idx in label_dict.items():
            print(f'    "{label}": {idx},')
        print("  }")
    
    print(f"\n요약:")
    print(f"  - 총 챕터 수: {result['summary']['total_chapters']}")
    print(f"  - 총 라벨 수: {result['summary']['total_labels']}")
    print(f"  - 제약 조건 만족: {'✅' if is_valid else '❌'}")
    
    # JSON 파일로 저장
    with open('chapter_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 'chapter_result.json'에 저장되었습니다.")
    
    return result

if __name__ == "__main__":
    result = main() 