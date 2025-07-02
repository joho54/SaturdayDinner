import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class CrossCategorizer:
    def __init__(self, max_chapters=4):
        self.max_chapters = max_chapters
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
        
        # 최대 비디오 클러스터 크기가 챕터 수를 초과하는지 확인
        max_video_cluster_size = max(video_cluster_sizes)
        if max_video_cluster_size > self.max_chapters:
            print(f"⚠️ 경고: 최대 비디오 클러스터 크기({max_video_cluster_size})가 챕터 수({self.max_chapters})를 초과합니다.")
            print("일부 비디오 클러스터의 라벨들이 다른 챕터에 배치될 수 없습니다.")
        
        return natural_clusters, video_clusters
    
    def create_chapters(self):
        """챕터 생성 알고리즘"""
        natural_clusters, video_clusters = self.analyze_constraints()
        
        # 모든 라벨 수집
        all_labels = list(self.label_info.keys())
        
        # 초기 챕터 할당
        label_to_chapter = {}
        
        # 1단계: 비디오 클러스터 제약 조건을 만족하는 초기 할당
        print("\n1단계: 비디오 클러스터 제약 조건 처리...")
        
        unassigned_labels = set(all_labels)
        
        # 큰 비디오 클러스터부터 처리
        video_clusters_sorted = sorted(video_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        for video_cluster_id, video_labels in video_clusters_sorted:
            available_chapters = list(range(self.max_chapters))
            
            # 이미 할당된 라벨들의 챕터를 제외
            used_chapters = set()
            for label in video_labels:
                if label in label_to_chapter:
                    used_chapters.add(label_to_chapter[label])
            
            available_chapters = [ch for ch in available_chapters if ch not in used_chapters]
            
            # 남은 라벨들을 사용 가능한 챕터에 할당
            unassigned_in_cluster = [label for label in video_labels if label not in label_to_chapter]
            
            if len(unassigned_in_cluster) > len(available_chapters):
                print(f"⚠️ 비디오 클러스터 {video_cluster_id}: {len(unassigned_in_cluster)}개 라벨, {len(available_chapters)}개 사용 가능한 챕터")
                # 가능한 만큼만 할당
                unassigned_in_cluster = unassigned_in_cluster[:len(available_chapters)]
            
            for i, label in enumerate(unassigned_in_cluster):
                if i < len(available_chapters):
                    label_to_chapter[label] = available_chapters[i]
                    unassigned_labels.discard(label)
        
        # 2단계: 자연어 클러스터 선호도를 고려하여 나머지 라벨 할당
        print("\n2단계: 자연어 클러스터 선호도 고려...")
        
        for natural_cluster_id, natural_labels in natural_clusters.items():
            # 이미 할당된 라벨들의 챕터 확인
            assigned_chapters = [label_to_chapter[label] for label in natural_labels if label in label_to_chapter]
            
            if assigned_chapters:
                # 가장 많이 사용된 챕터 찾기
                chapter_counts = Counter(assigned_chapters)
                preferred_chapter = chapter_counts.most_common(1)[0][0]
                
                # 아직 할당되지 않은 라벨들을 선호 챕터에 할당 (비디오 클러스터 제약 조건 확인)
                unassigned_in_natural = [label for label in natural_labels if label not in label_to_chapter]
                
                for label in unassigned_in_natural:
                    # 해당 라벨의 비디오 클러스터에서 이미 preferred_chapter를 사용하는지 확인
                    video_cluster_id = self.label_info[label]['video_cluster']
                    conflicting_labels = [
                        other_label for other_label in video_clusters[video_cluster_id]
                        if other_label in label_to_chapter and label_to_chapter[other_label] == preferred_chapter
                    ]
                    
                    if not conflicting_labels:  # 충돌이 없으면 할당
                        label_to_chapter[label] = preferred_chapter
                        unassigned_labels.discard(label)
        
        # 3단계: 남은 라벨들을 임의로 할당
        print("\n3단계: 남은 라벨들 할당...")
        for label in list(unassigned_labels):
            video_cluster_id = self.label_info[label]['video_cluster']
            
            # 비디오 클러스터에서 사용된 챕터들 확인
            used_chapters = set()
            for other_label in video_clusters[video_cluster_id]:
                if other_label in label_to_chapter:
                    used_chapters.add(label_to_chapter[other_label])
            
            # 사용 가능한 챕터 찾기
            available_chapters = [ch for ch in range(self.max_chapters) if ch not in used_chapters]
            
            if available_chapters:
                label_to_chapter[label] = available_chapters[0]
                unassigned_labels.discard(label)
            else:
                print(f"⚠️ 라벨 '{label}'을 할당할 수 없습니다 (비디오 클러스터 제약 조건)")
        
        # 챕터 딕셔너리 생성
        chapters = defaultdict(list)
        for label, chapter_id in label_to_chapter.items():
            chapters[chapter_id].append(label)
        
        # None 챕터 추가
        self.chapters = {f"챕터_{i}": labels for i, labels in chapters.items()}
        if len(self.chapters) < self.max_chapters:
            self.chapters["None"] = []
        
        return self.chapters
    
    def validate_constraints(self):
        """제약 조건 검증"""
        print("\n제약 조건 검증...")
        
        # 챕터별 라벨 딕셔너리 생성
        label_to_chapter = {}
        for chapter_name, labels in self.chapters.items():
            for label in labels:
                label_to_chapter[label] = chapter_name
        
        # 비디오 클러스터 제약 조건 검증
        natural_clusters, video_clusters = self.analyze_constraints()
        
        violations = []
        
        # 비디오 클러스터 제약 조건 확인
        for video_cluster_id, video_labels in video_clusters.items():
            assigned_chapters = set()
            for label in video_labels:
                if label in label_to_chapter:
                    assigned_chapters.add(label_to_chapter[label])
            
            if len(assigned_chapters) < len(video_labels):
                violations.append(f"비디오 클러스터 {video_cluster_id}: {len(video_labels)}개 라벨이 {len(assigned_chapters)}개 챕터에만 할당됨")
        
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
        
        avg_cohesion = np.mean(natural_cohesion) if natural_cohesion else 0
        print(f"자연어 클러스터 평균 응집도: {avg_cohesion:.2f}")
        
        return len(violations) == 0
    
    def generate_output(self):
        """최종 결과 생성"""
        # 라벨 딕셔너리 생성
        label_dict = {}
        label_id = 0
        
        for chapter_name, labels in self.chapters.items():
            for label in labels:
                label_dict[label] = label_id
                label_id += 1
        
        # None 추가
        label_dict["None"] = label_id
        
        result = {
            "label_dict": label_dict,
            "chapters": self.chapters,
            "summary": {
                "total_labels": len(label_dict) - 1,  # None 제외
                "total_chapters": len(self.chapters),
                "chapter_sizes": {name: len(labels) for name, labels in self.chapters.items()}
            }
        }
        
        return result

def main():
    # 교차 카테고라이저 초기화
    categorizer = CrossCategorizer(max_chapters=4)
    
    # 데이터 로드
    categorizer.load_data(
        'clustered_labels_with_filenames.csv',
        'clustering_results_20250702_195813.csv'
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
    
    for chapter_name, labels in result["chapters"].items():
        print(f"\n{chapter_name} ({len(labels)}개 라벨):")
        if labels:
            for i, label in enumerate(labels[:10]):  # 처음 10개만 표시
                print(f"  {i+1}. {label}")
            if len(labels) > 10:
                print(f"  ... (총 {len(labels)}개)")
        else:
            print("  (비어있음)")
    
    print(f"\n요약:")
    print(f"  - 총 라벨 수: {result['summary']['total_labels']}")
    print(f"  - 총 챕터 수: {result['summary']['total_chapters']}")
    print(f"  - 제약 조건 만족: {'✅' if is_valid else '❌'}")
    
    # JSON 파일로 저장
    with open('chapter_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 'chapter_result.json'에 저장되었습니다.")
    
    return result

if __name__ == "__main__":
    result = main() 