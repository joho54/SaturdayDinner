import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
import warnings
import random
warnings.filterwarnings('ignore')

class ImprovedCrossCategorizer:
    def __init__(self, max_chapters=4):
        self.max_chapters = max_chapters
        self.label_cluster_data = None
        self.video_cluster_data = None
        self.merged_data = None
        self.chapters = {}
        self.adjacency = None
        
    def load_data(self, label_cluster_file, video_cluster_file):
        """데이터 로드 및 전처리"""
        self.label_cluster_data = pd.read_csv(label_cluster_file)
        print(f"라벨 클러스터 데이터 로드: {len(self.label_cluster_data)} 행")
        
        self.video_cluster_data = pd.read_csv(video_cluster_file)
        print(f"비디오 클러스터 데이터 로드: {len(self.video_cluster_data)} 행")
        
        self._merge_data()
        
    def _merge_data(self):
        """두 데이터셋을 병합"""
        label_data = self.label_cluster_data.copy()
        video_data = self.video_cluster_data.copy()
        
        label_data['filename_normalized'] = label_data['파일명'].str.replace(r'\.(MOV|AVI|MP4|MTS)$', '', regex=True)
        video_data['filename_normalized'] = video_data['filename'].str.replace(r'\.(MOV|AVI|MP4|MTS)$', '', regex=True)
        
        self.merged_data = pd.merge(
            label_data[['filename_normalized', '한국어', '클러스터_ID', '클러스터_크기']],
            video_data[['filename_normalized', 'cluster_id']],
            on='filename_normalized',
            how='inner'
        )
        
        print(f"병합된 데이터: {len(self.merged_data)} 행")
        
        self.label_info = {}
        for _, row in self.merged_data.iterrows():
            label = row['한국어']
            if label not in self.label_info:
                self.label_info[label] = {
                    'natural_cluster': row['클러스터_ID'],
                    'video_cluster': row['cluster_id']
                }
        
        print(f"고유 라벨 수: {len(self.label_info)}")
        
    def build_constraint_graph(self):
        """제약 조건을 기반으로 그래프 구축"""
        print("\n제약 조건 그래프 구축...")
        
        natural_clusters = defaultdict(list)
        video_clusters = defaultdict(list)
        
        for label, info in self.label_info.items():
            natural_clusters[info['natural_cluster']].append(label)
            video_clusters[info['video_cluster']].append(label)
        
        # 인접 리스트 생성
        all_labels = list(self.label_info.keys())
        self.adjacency = {label: set() for label in all_labels}
        
        # 비디오 클러스터 제약 조건: 같은 비디오 클러스터의 라벨들은 서로 인접
        conflict_edges = 0
        for video_cluster_id, video_labels in video_clusters.items():
            for i in range(len(video_labels)):
                for j in range(i + 1, len(video_labels)):
                    self.adjacency[video_labels[i]].add(video_labels[j])
                    self.adjacency[video_labels[j]].add(video_labels[i])
                    conflict_edges += 1
        
        print(f"충돌 엣지 수: {conflict_edges}")
        print(f"자연어 클러스터 수: {len(natural_clusters)}")
        print(f"비디오 클러스터 수: {len(video_clusters)}")
        
        return natural_clusters, video_clusters
    
    def greedy_coloring_with_priorities(self, natural_clusters):
        """우선순위를 고려한 탐욕적 색칠 알고리즘"""
        print("\n우선순위 기반 탐욕적 색칠 시작...")
        
        # 차수 계산
        degree_dict = {label: len(neighbors) for label, neighbors in self.adjacency.items()}
        
        # 자연어 클러스터 크기 계산
        natural_cluster_sizes = {}
        for cluster_id, labels in natural_clusters.items():
            for label in labels:
                natural_cluster_sizes[label] = len(labels)
        
        # 우선순위: 차수가 높고, 자연어 클러스터가 작은 라벨부터 처리
        def priority_score(label):
            degree = degree_dict.get(label, 0)
            cluster_size = natural_cluster_sizes.get(label, 1)
            return (degree, -cluster_size)
        
        sorted_labels = sorted(self.label_info.keys(), key=priority_score, reverse=True)
        
        # 색칠 수행
        coloring = {}
        
        for label in sorted_labels:
            # 인접한 노드들의 색깔 확인
            used_colors = set()
            for neighbor in self.adjacency[label]:
                if neighbor in coloring:
                    used_colors.add(coloring[neighbor])
            
            # 사용 가능한 색깔 찾기
            for color in range(self.max_chapters):
                if color not in used_colors:
                    coloring[label] = color
                    break
            else:
                # 사용 가능한 색깔이 없는 경우 - 가장 적게 사용된 색깔 할당
                color_counts = Counter(coloring.values())
                min_color = min(range(self.max_chapters), key=lambda c: color_counts.get(c, 0))
                coloring[label] = min_color
                print(f"⚠️ 라벨 '{label}'에 강제로 색깔 {min_color} 할당")
        
        return coloring
    
    def improve_with_natural_clusters(self, coloring, natural_clusters):
        """자연어 클러스터 응집도를 개선"""
        print("\n자연어 클러스터 응집도 개선...")
        
        improved_coloring = coloring.copy()
        improvements = 0
        
        for cluster_id, cluster_labels in natural_clusters.items():
            if len(cluster_labels) <= 1:
                continue
                
            # 현재 클러스터의 색깔 분포 확인
            color_counts = Counter()
            for label in cluster_labels:
                if label in improved_coloring:
                    color_counts[improved_coloring[label]] += 1
            
            if not color_counts:
                continue
                
            # 가장 많이 사용된 색깔
            target_color = color_counts.most_common(1)[0][0]
            
            # 다른 색깔의 라벨들을 target_color로 변경 시도
            for label in cluster_labels:
                if label not in improved_coloring:
                    continue
                    
                current_color = improved_coloring[label]
                if current_color == target_color:
                    continue
                
                # target_color로 변경 가능한지 확인
                can_change = True
                for neighbor in self.adjacency[label]:
                    if neighbor in improved_coloring and improved_coloring[neighbor] == target_color:
                        can_change = False
                        break
                
                if can_change:
                    improved_coloring[label] = target_color
                    improvements += 1
        
        print(f"개선된 할당 수: {improvements}")
        return improved_coloring
    
    def evaluate_coloring(self, coloring, natural_clusters):
        """색칠 결과 평가"""
        # 제약 조건 위반 계산
        violations = 0
        for label, neighbors in self.adjacency.items():
            if label in coloring:
                for neighbor in neighbors:
                    if neighbor in coloring and coloring[label] == coloring[neighbor]:
                        violations += 1
        violations //= 2  # 각 위반이 두 번 계산되므로
        
        # 자연어 클러스터 응집도 계산
        cohesion_score = 0
        total_clusters = 0
        
        for cluster_id, cluster_labels in natural_clusters.items():
            if len(cluster_labels) <= 1:
                continue
                
            color_counts = Counter()
            for label in cluster_labels:
                if label in coloring:
                    color_counts[coloring[label]] += 1
            
            if color_counts:
                max_count = max(color_counts.values())
                total_count = sum(color_counts.values())
                cohesion = max_count / total_count
                cohesion_score += cohesion
                total_clusters += 1
        
        avg_cohesion = cohesion_score / total_clusters if total_clusters > 0 else 0
        
        # 전체 점수
        score = avg_cohesion - violations * 0.01
        return score, violations, avg_cohesion
    
    def create_chapters(self):
        """개선된 챕터 생성 알고리즘"""
        natural_clusters, video_clusters = self.build_constraint_graph()
        
        # 1단계: 탐욕적 색칠
        coloring = self.greedy_coloring_with_priorities(natural_clusters)
        
        # 2단계: 자연어 클러스터 응집도 개선
        coloring = self.improve_with_natural_clusters(coloring, natural_clusters)
        
        # 성능 평가
        score, violations, cohesion = self.evaluate_coloring(coloring, natural_clusters)
        print(f"최종 점수: {score:.3f} (위반: {violations}, 응집도: {cohesion:.3f})")
        
        # 챕터 딕셔너리 생성
        chapters = defaultdict(list)
        for label, chapter_id in coloring.items():
            chapters[chapter_id].append(label)
        
        self.chapters = {f"챕터_{i}": labels for i, labels in chapters.items()}
        
        # 빈 챕터가 있으면 추가
        while len(self.chapters) < self.max_chapters:
            empty_chapter_id = len(self.chapters)
            self.chapters[f"챕터_{empty_chapter_id}"] = []
        
        return self.chapters
    
    def validate_constraints(self):
        """제약 조건 검증"""
        print("\n제약 조건 검증...")
        
        label_to_chapter = {}
        for chapter_name, labels in self.chapters.items():
            for label in labels:
                label_to_chapter[label] = chapter_name
        
        violations = 0
        
        # 인접한 라벨들이 같은 챕터에 있는지 확인
        for label, neighbors in self.adjacency.items():
            if label in label_to_chapter:
                for neighbor in neighbors:
                    if neighbor in label_to_chapter and label_to_chapter[label] == label_to_chapter[neighbor]:
                        violations += 1
        violations //= 2  # 각 위반이 두 번 계산되므로
        
        # 자연어 클러스터 응집도 분석
        natural_clusters = defaultdict(list)
        for label, info in self.label_info.items():
            natural_clusters[info['natural_cluster']].append(label)
        
        natural_cohesion = []
        for natural_cluster_id, natural_labels in natural_clusters.items():
            if len(natural_labels) <= 1:
                continue
                
            chapter_distribution = defaultdict(int)
            for label in natural_labels:
                if label in label_to_chapter:
                    chapter_distribution[label_to_chapter[label]] += 1
            
            if chapter_distribution:
                max_in_one_chapter = max(chapter_distribution.values())
                cohesion = max_in_one_chapter / len(natural_labels)
                natural_cohesion.append(cohesion)
        
        avg_cohesion = np.mean(natural_cohesion) if natural_cohesion else 0
        
        print(f"제약 조건 위반 수: {violations}")
        print(f"자연어 클러스터 평균 응집도: {avg_cohesion:.3f}")
        print(f"할당된 라벨 수: {len(label_to_chapter)}/{len(self.label_info)}")
        
        return violations == 0, violations, avg_cohesion
    
    def generate_output(self):
        """최종 결과 생성"""
        label_dict = {}
        label_id = 0
        
        for chapter_name, labels in self.chapters.items():
            for label in labels:
                label_dict[label] = label_id
                label_id += 1
        
        # None 추가
        label_dict["None"] = label_id
        
        # 제약 조건 검증
        is_valid, violations, avg_cohesion = self.validate_constraints()
        
        result = {
            "label_dict": label_dict,
            "chapters": self.chapters,
            "summary": {
                "total_labels": len(self.label_info),
                "assigned_labels": len(label_dict) - 1,
                "assignment_rate": (len(label_dict) - 1) / len(self.label_info) * 100,
                "total_chapters": len([ch for ch in self.chapters.keys() if len(self.chapters[ch]) > 0]),
                "chapter_sizes": {name: len(labels) for name, labels in self.chapters.items()},
                "constraint_violations": violations,
                "natural_cluster_cohesion": avg_cohesion,
                "constraints_satisfied": is_valid
            }
        }
        
        return result

def main():
    categorizer = ImprovedCrossCategorizer(max_chapters=4)
    
    categorizer.load_data(
        'clustered_labels_with_filenames.csv',
        'clustering_results_20250702_195813.csv'
    )
    
    chapters = categorizer.create_chapters()
    result = categorizer.generate_output()
    
    print("\n" + "="*60)
    print("개선된 챕터 생성 결과")
    print("="*60)
    
    for chapter_name, labels in result["chapters"].items():
        if not labels:
            continue
            
        print(f"\n{chapter_name} ({len(labels)}개 라벨):")
        
        # 자연어 클러스터별로 그룹화하여 표시
        label_groups = defaultdict(list)
        for label in labels:
            if label in categorizer.label_info:
                cluster_id = categorizer.label_info[label]['natural_cluster']
                label_groups[cluster_id].append(label)
        
        for cluster_id, cluster_labels in list(label_groups.items())[:3]:  # 처음 3개 클러스터만
            print(f"  [자연어 클러스터 {cluster_id}] ({len(cluster_labels)}개):")
            for label in cluster_labels[:3]:  # 처음 3개 라벨만
                print(f"    - {label}")
            if len(cluster_labels) > 3:
                print(f"    ...")
        
        if len(label_groups) > 3:
            print(f"  ... (총 {len(label_groups)}개 자연어 클러스터)")
    
    summary = result["summary"]
    print(f"\n📊 요약:")
    print(f"  • 전체 라벨 수: {summary['total_labels']}")
    print(f"  • 할당된 라벨 수: {summary['assigned_labels']}")
    print(f"  • 할당 비율: {summary['assignment_rate']:.1f}%")
    print(f"  • 활성 챕터 수: {summary['total_chapters']}")
    print(f"  • 제약 조건 위반: {summary['constraint_violations']}개")
    print(f"  • 자연어 클러스터 응집도: {summary['natural_cluster_cohesion']:.3f}")
    print(f"  • 제약 조건 만족: {'✅' if summary['constraints_satisfied'] else '❌'}")
    
    with open('improved_chapter_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과가 'improved_chapter_result.json'에 저장되었습니다.")
    
    return result

if __name__ == "__main__":
    result = main() 