import pandas as pd
import json
from typing import Dict, List, Set
from collections import defaultdict
import random

class VideoClusterCategorizer:
    def __init__(self, csv_file_path: str):
        """
        VideoClusterCategorizer 초기화
        
        Args:
            csv_file_path (str): video_clusters.csv 파일 경로
        """
        self.csv_file_path = csv_file_path
        self.cluster_to_labels = defaultdict(list)
        self.label_to_cluster = {}
        self._load_data()
    
    def _load_data(self):
        """CSV 파일에서 데이터를 로드하고 클러스터별로 라벨을 그룹핑"""
        try:
            df = pd.read_csv(self.csv_file_path)
            
            for _, row in df.iterrows():
                label_name = row['label_name']
                cluster_id = row['cluster_id']
                
                self.cluster_to_labels[cluster_id].append(label_name)
                self.label_to_cluster[label_name] = cluster_id
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_cluster_info(self):
        """클러스터 정보 출력"""
        print(f"Total clusters: {len(self.cluster_to_labels)}")
        print(f"Total labels: {len(self.label_to_cluster)}")
        
        # 상위 10개 클러스터만 출력
        sorted_clusters = sorted(self.cluster_to_labels.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nTop 10 clusters by label count:")
        for cluster_id, labels in sorted_clusters[:10]:
            print(f"Cluster {cluster_id}: {len(labels)} labels")
    
    def _generate_model_id(self, labels: List[str]) -> str:
        """모델 ID 생성"""
        # 라벨들을 기반으로 간단한 모델 ID 생성
        label_hash = hash(tuple(sorted(labels))) % 10000
        return f"MODEL_{label_hash:04d}"
    
    def _get_single_label_combination(self, used_labels: Set[str]) -> List[str]:
        """
        사용되지 않은 라벨들로 서로 다른 클러스터에서 4개 라벨 하나의 조합 생성
        
        Args:
            used_labels (Set[str]): 이미 사용된 라벨들
            
        Returns:
            List[str]: 4개 라벨 조합 또는 빈 리스트
        """
        # 각 클러스터에서 사용되지 않은 라벨들 추출
        available_clusters = []
        for cluster_id, labels in self.cluster_to_labels.items():
            unused_labels = [label for label in labels if label not in used_labels]
            if unused_labels:
                available_clusters.append((cluster_id, unused_labels))
        
        # 최소 4개의 다른 클러스터가 있는지 확인
        if len(available_clusters) < 4:
            return []
        
        # 랜덤하게 4개 클러스터 선택
        selected_clusters = random.sample(available_clusters, 4)
        
        # 각 클러스터에서 하나의 라벨 선택
        selected_labels = []
        for cluster_id, unused_labels in selected_clusters:
            selected_labels.append(random.choice(unused_labels))
        
        return selected_labels
    
    def _get_fallback_combination(self) -> List[str]:
        """
        모든 라벨이 사용된 후 중복 사용을 위한 조합 생성
        
        Returns:
            List[str]: 4개 라벨 조합 또는 빈 리스트
        """
        # 라벨이 있는 클러스터들만 선택
        available_clusters = []
        for cluster_id, labels in self.cluster_to_labels.items():
            if labels:
                available_clusters.append((cluster_id, labels))
        
        # 최소 4개의 다른 클러스터가 있는지 확인
        if len(available_clusters) < 4:
            return []
        
        # 랜덤하게 4개 클러스터 선택
        selected_clusters = random.sample(available_clusters, 4)
        
        # 각 클러스터에서 하나의 라벨 선택
        selected_labels = []
        for cluster_id, labels in selected_clusters:
            selected_labels.append(random.choice(labels))
        
        return selected_labels

    def find_missing_labels(self, model_specs: Dict) -> List[str]:
        """
        모델 명세에서 누락된 라벨들을 찾음
        
        Args:
            model_specs (Dict): 모델 명세 딕셔너리
            
        Returns:
            List[str]: 누락된 라벨들
        """
        used_labels = set()
        
        for model in model_specs['models']:
            for label in model['label_dict'].keys():
                if label != "None":
                    used_labels.add(label)
        
        all_labels = set(self.label_to_cluster.keys())
        missing_labels = list(all_labels - used_labels)
        
        print(f"Total labels in dataset: {len(all_labels)}")
        print(f"Used labels in models: {len(used_labels)}")
        print(f"Missing labels: {len(missing_labels)}")
        
        if missing_labels:
            print(f"Missing labels: {missing_labels[:10]}...")  # 처음 10개만 출력
        
        return missing_labels

    def add_missing_labels_models(self, model_specs: Dict, ignore_cluster_rule: bool = True) -> Dict:
        """
        누락된 라벨들을 포함하는 추가 모델들을 생성
        
        Args:
            model_specs (Dict): 기존 모델 명세 딕셔너리
            ignore_cluster_rule (bool): 클러스터 룰을 무시할지 여부
            
        Returns:
            Dict: 업데이트된 모델 명세 딕셔너리
        """
        missing_labels = self.find_missing_labels(model_specs)
        
        if not missing_labels:
            print("No missing labels found!")
            return model_specs
        
        print(f"Adding models for {len(missing_labels)} missing labels...")
        
        models = model_specs['models'].copy()
        
        # 누락된 라벨들을 4개씩 그룹으로 나누어 모델 생성
        for i in range(0, len(missing_labels), 4):
            batch = missing_labels[i:i+4]
            
            # 4개 미만이면 기존 라벨로 채움
            while len(batch) < 4:
                # 아무 라벨이나 추가 (룰 무시)
                all_labels = list(self.label_to_cluster.keys())
                batch.append(random.choice(all_labels))
            
            # 모델 명세 생성
            model_id = self._generate_model_id(batch + [f"missing_{i}"])
            label_dict = {}
            
            for j, label in enumerate(batch):
                label_dict[label] = j
            
            # "None" 라벨 추가
            label_dict["None"] = len(batch)
            
            models.append({
                "model_id": model_id,
                "label_dict": label_dict
            })
        
        updated_specs = {
            "models": models
        }
        
        print(f"Added {len(models) - len(model_specs['models'])} models for missing labels")
        print(f"Total models: {len(models)}")
        
        return updated_specs

    def complete_model_specifications(self, json_file_path: str) -> Dict:
        """
        기존 모델 명세 파일을 읽어서 누락된 라벨들을 추가한 완전한 모델 명세를 생성
        
        Args:
            json_file_path (str): 기존 모델 명세 JSON 파일 경로
            
        Returns:
            Dict: 완성된 모델 명세 딕셔너리
        """
        # 기존 모델 명세 로드
        with open(json_file_path, 'r', encoding='utf-8') as f:
            existing_specs = json.load(f)
        
        # 누락된 라벨들을 포함하는 모델 추가
        complete_specs = self.add_missing_labels_models(existing_specs, ignore_cluster_rule=True)
        
        # 완성된 명세 저장
        output_path = json_file_path.replace('.json', '_complete.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_specs, f, ensure_ascii=False, indent=2)
        
        print(f"Complete model specifications saved to {output_path}")
        
        # 최종 검증
        final_missing = self.find_missing_labels(complete_specs)
        if not final_missing:
            print("✅ All labels are now included in the model specifications!")
        else:
            print(f"⚠️ Still missing {len(final_missing)} labels")
        
        return complete_specs
    
    def generate_model_specifications(self, max_models: int = 100) -> Dict:
        """
        모델 명세 목록 생성
        
        Args:
            max_models (int): 생성할 최대 모델 수
            
        Returns:
            Dict: 모델 명세 딕셔너리
        """
        models = []
        used_labels = set()
        all_labels = list(self.label_to_cluster.keys())
        
        model_count = 0
        max_attempts = 1000  # 무한 루프 방지
        attempts = 0
        
        print(f"Generating model specifications... (max {max_models} models)")
        
        # 1단계: 모든 라벨이 최소 한 번 이상 사용되도록 조합 생성
        while len(used_labels) < len(all_labels) and model_count < max_models and attempts < max_attempts:
            attempts += 1
            selected_labels = self._get_single_label_combination(used_labels)
            
            if not selected_labels:
                print(f"No more unique combinations available. Used {len(used_labels)}/{len(all_labels)} labels.")
                break
            
            # 모델 명세 생성
            model_id = self._generate_model_id(selected_labels)
            label_dict = {}
            
            for i, label in enumerate(selected_labels):
                label_dict[label] = i
            
            # "None" 라벨 추가
            label_dict["None"] = len(selected_labels)
            
            models.append({
                "model_id": model_id,
                "label_dict": label_dict
            })
            
            # 사용된 라벨 추가
            used_labels.update(selected_labels)
            model_count += 1
            
            if model_count % 10 == 0:
                print(f"Generated {model_count} models, used {len(used_labels)}/{len(all_labels)} labels")
        
        # 2단계: 추가 모델 생성 (중복 라벨 허용)
        additional_attempts = 0
        max_additional_attempts = 100
        
        while model_count < max_models and additional_attempts < max_additional_attempts:
            additional_attempts += 1
            selected_labels = self._get_fallback_combination()
            
            if not selected_labels:
                break
            
            # 모델 명세 생성
            model_id = self._generate_model_id(selected_labels + [str(additional_attempts)])
            label_dict = {}
            
            for i, label in enumerate(selected_labels):
                label_dict[label] = i
            
            # "None" 라벨 추가
            label_dict["None"] = len(selected_labels)
            
            models.append({
                "model_id": model_id,
                "label_dict": label_dict
            })
            
            model_count += 1
        
        print(f"Total models generated: {len(models)}")
        print(f"Total unique labels used: {len(used_labels)}/{len(all_labels)}")
        
        return {
            "models": models
        }
    
    def save_model_specifications(self, output_path: str, max_models: int = 100):
        """
        모델 명세를 JSON 파일로 저장
        
        Args:
            output_path (str): 출력 파일 경로
            max_models (int): 생성할 최대 모델 수
        """
        model_specs = self.generate_model_specifications(max_models)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_specs, f, ensure_ascii=False, indent=2)
        
        print(f"Model specifications saved to {output_path}")
        print(f"Generated {len(model_specs['models'])} models")
    
    def print_model_specifications(self, max_models: int = 10):
        """
        모델 명세 출력 (미리보기)
        
        Args:
            max_models (int): 출력할 최대 모델 수
        """
        model_specs = self.generate_model_specifications(max_models)
        
        print("\nModel Specifications:")
        print("=" * 50)
        
        for i, model in enumerate(model_specs['models'][:max_models]):
            print(f"\nModel {i+1}:")
            print(f"  ID: {model['model_id']}")
            print(f"  Labels:")
            for label, idx in model['label_dict'].items():
                if label != "None":
                    cluster_id = self.label_to_cluster.get(label, "Unknown")
                    print(f"    {label}: {idx} (Cluster {cluster_id})")
                else:
                    print(f"    {label}: {idx}")
    
    def validate_model_specifications(self, model_specs: Dict) -> bool:
        """
        모델 명세의 유효성 검증
        
        Args:
            model_specs (Dict): 모델 명세 딕셔너리
            
        Returns:
            bool: 유효성 검증 결과
        """
        print("\nValidating model specifications...")
        
        for i, model in enumerate(model_specs['models']):
            labels = [label for label in model['label_dict'].keys() if label != "None"]
            
            # 4개 라벨인지 확인
            if len(labels) != 4:
                print(f"Model {i+1}: Invalid label count ({len(labels)} instead of 4)")
                return False
            
            # 서로 다른 클러스터인지 확인
            clusters = set()
            for label in labels:
                cluster_id = self.label_to_cluster.get(label)
                if cluster_id in clusters:
                    print(f"Model {i+1}: Duplicate cluster {cluster_id} for labels {labels}")
                    return False
                clusters.add(cluster_id)
        
        print(f"All {len(model_specs['models'])} models are valid!")
        return True

# 사용 예시
if __name__ == "__main__":
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(42)
    
    # VideoClusterCategorizer 인스턴스 생성
    categorizer = VideoClusterCategorizer("two-clusters/video_clusters.csv")
    
    # 클러스터 정보 출력
    categorizer.get_cluster_info()
    
    # 기존 모델 명세를 완성하여 누락된 라벨들 추가
    complete_specs = categorizer.complete_model_specifications("model_specifications.json")
    
    # 최종 유효성 검증 (클러스터 룰 무시하고 라벨 포함 여부만 확인)
    categorizer.find_missing_labels(complete_specs)
