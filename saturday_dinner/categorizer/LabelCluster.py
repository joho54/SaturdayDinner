import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os

def load_korean_model():
    """한국어 sentence transformer 모델 로드"""
    print("한국어 임베딩 모델을 로드하는 중...")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

def create_embeddings(texts, model):
    """텍스트들의 임베딩을 생성"""
    print(f"{len(texts)}개 텍스트의 임베딩을 생성하는 중...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def recursive_clustering(texts, embeddings, max_cluster_size=20, min_cluster_size=2):
    """재귀적으로 클러스터링을 수행하여 최대 클러스터 크기를 제한"""
    
    def cluster_group(group_texts, group_embeddings, cluster_id_prefix=""):
        if len(group_texts) <= max_cluster_size:
            # 클러스터가 충분히 작으면 그대로 반환
            return [{
                'cluster_id': f"{cluster_id_prefix}C{len(results)}",
                'texts': group_texts,
                'size': len(group_texts)
            }]
        
        # 클러스터가 너무 크면 세분화
        n_clusters = max(2, min(len(group_texts) // max_cluster_size + 1, len(group_texts)))
        
        if len(group_texts) <= 3:
            # 너무 작은 그룹은 그대로 유지
            return [{
                'cluster_id': f"{cluster_id_prefix}C{len(results)}",
                'texts': group_texts,
                'size': len(group_texts)
            }]
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(group_embeddings)
        
        subclusters = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.sum(mask) >= min_cluster_size:
                sub_texts = [group_texts[j] for j in range(len(group_texts)) if mask[j]]
                sub_embeddings = group_embeddings[mask]
                
                # 재귀적으로 세분화
                subclusters.extend(
                    cluster_group(sub_texts, sub_embeddings, f"{cluster_id_prefix}{i}_")
                )
        
        return subclusters
    
    results = []
    
    # 첫 번째 클러스터링 - 대략적인 그룹 형성
    initial_clusters = max(2, len(texts) // (max_cluster_size * 2))
    initial_clusters = min(initial_clusters, len(texts) // 2)
    
    if initial_clusters < 2:
        initial_clusters = 2
    
    print(f"초기 클러스터링: {initial_clusters}개 클러스터로 시작")
    
    kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
    initial_labels = kmeans.fit_predict(embeddings)
    
    # 각 초기 클러스터를 재귀적으로 세분화
    for i in range(initial_clusters):
        mask = initial_labels == i
        if np.sum(mask) >= min_cluster_size:
            cluster_texts = [texts[j] for j in range(len(texts)) if mask[j]]
            cluster_embeddings = embeddings[mask]
            
            subclusters = cluster_group(cluster_texts, cluster_embeddings, f"G{i}_")
            results.extend(subclusters)
    
    return results

def analyze_clusters(clusters):
    """클러스터 분석 및 통계"""
    print("\n=== 클러스터링 결과 분석 ===")
    print(f"총 클러스터 수: {len(clusters)}")
    
    sizes = [cluster['size'] for cluster in clusters]
    print(f"클러스터 크기 통계:")
    print(f"  - 평균: {np.mean(sizes):.1f}")
    print(f"  - 최소: {np.min(sizes)}")
    print(f"  - 최대: {np.max(sizes)}")
    print(f"  - 중앙값: {np.median(sizes):.1f}")
    
    # 크기별 클러스터 개수
    size_counts = defaultdict(int)
    for size in sizes:
        size_counts[size] += 1
    
    print(f"\n클러스터 크기별 분포:")
    for size in sorted(size_counts.keys()):
        print(f"  - 크기 {size}: {size_counts[size]}개")
    
    return sizes

def save_clusters_to_csv(clusters, filename='two-clusters/clustered_labels.csv'):
    """클러스터 결과를 CSV로 저장"""
    data = []
    for cluster in clusters:
        for text in cluster['texts']:
            data.append({
                '한국어': text,
                '클러스터_ID': cluster['cluster_id'],
                '클러스터_크기': cluster['size']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n클러스터링 결과가 '{filename}'에 저장되었습니다.")
    return df

def create_sample_display(clusters, max_display=20):
    """클러스터 샘플을 보기 좋게 출력"""
    print(f"\n=== 클러스터 샘플 (상위 {max_display}개) ===")
    
    # 클러스터를 크기순으로 정렬
    sorted_clusters = sorted(clusters, key=lambda x: x['size'], reverse=True)
    
    for i, cluster in enumerate(sorted_clusters[:max_display]):
        print(f"\n[{cluster['cluster_id']}] (크기: {cluster['size']})")
        print("  내용:", end=" ")
        if len(cluster['texts']) <= 5:
            print(", ".join(cluster['texts']))
        else:
            print(", ".join(cluster['texts'][:3]) + f" ... (외 {len(cluster['texts'])-3}개)")

def visualize_clusters_2d(embeddings, clusters, texts, save_path='clusters_visualization.png'):
    """2D PCA를 사용한 클러스터 시각화"""
    print("\n클러스터 시각화를 생성하는 중...")
    
    # PCA로 2D 축소
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # 클러스터 라벨 매핑
    text_to_cluster = {}
    for cluster in clusters:
        for text in cluster['texts']:
            text_to_cluster[text] = cluster['cluster_id']
    
    cluster_labels = [text_to_cluster.get(text, 'Unknown') for text in texts]
    unique_labels = list(set(cluster_labels))
    
    # 시각화
    plt.figure(figsize=(15, 12))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(cluster_labels) == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=50)
    
    plt.title('한국어 레이블 클러스터링 시각화 (PCA 2D)', fontsize=16)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    
    # 범례는 너무 많으면 생략
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"시각화 결과가 '{save_path}'에 저장되었습니다.")
    plt.close()

def merge_cluster_with_filenames():
    """클러스터링된 데이터에 파일명 정보를 추가"""
    
    print("\n=== 파일명 매핑 시작 ===")
    
    # 1. 원본 labels.csv 읽기
    print("원본 labels.csv 파일을 읽는 중...")
    labels_df = pd.read_csv('labels.csv')
    print(f"원본 데이터: {len(labels_df)}행")
    
    # 2. 클러스터링된 파일 읽기
    print("clustered_labels.csv 파일을 읽는 중...")
    clustered_df = pd.read_csv('two-clusters/clustered_labels.csv')
    print(f"클러스터링된 데이터: {len(clustered_df)}행")
    
    # 3. 각 한국어 라벨에 대한 첫 번째 파일명 매핑 생성
    print("각 한국어 라벨에 대한 첫 번째 파일명 매핑을 생성하는 중...")
    
    # 한국어 라벨별로 그룹화하고 첫 번째 파일명 가져오기
    first_filename_mapping = labels_df.groupby('한국어')['파일명'].first().to_dict()
    
    print(f"총 {len(first_filename_mapping)}개의 유니크한 한국어 라벨에 대한 매핑 생성")
    
    # 4. 클러스터링된 데이터에 파일명 칼럼 추가
    print("클러스터링된 데이터에 파일명 정보를 추가하는 중...")
    
    clustered_df['파일명'] = clustered_df['한국어'].map(first_filename_mapping)
    
    # 5. 매핑되지 않은 데이터 확인
    missing_mappings = clustered_df['파일명'].isnull().sum()
    if missing_mappings > 0:
        print(f"경고: {missing_mappings}개 라벨에 대한 파일명을 찾을 수 없습니다.")
        print("매핑되지 않은 라벨들:")
        missing_labels = clustered_df[clustered_df['파일명'].isnull()]['한국어'].unique()
        for label in missing_labels[:10]:  # 처음 10개만 출력
            print(f"  - {label}")
        if len(missing_labels) > 10:
            print(f"  ... (외 {len(missing_labels)-10}개)")
    
    # 6. 칼럼 순서 재정렬 (파일명을 첫 번째로)
    column_order = ['파일명', '한국어', '클러스터_ID', '클러스터_크기']
    clustered_df = clustered_df[column_order]
    
    # 7. 결과 저장
    output_filename = 'two-clusters/label_clusters.csv'
    clustered_df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\n병합된 결과가 '{output_filename}'에 저장되었습니다.")
    
    # 8. 결과 요약
    print(f"\n=== 파일명 매핑 결과 요약 ===")
    print(f"총 데이터 행 수: {len(clustered_df)}")
    print(f"유니크한 클러스터 수: {clustered_df['클러스터_ID'].nunique()}")
    print(f"유니크한 한국어 라벨 수: {clustered_df['한국어'].nunique()}")
    print(f"파일명이 매핑된 행 수: {clustered_df['파일명'].notna().sum()}")
    
    # 9. 샘플 데이터 출력
    print(f"\n=== 최종 결과 샘플 (상위 10개) ===")
    print(clustered_df.head(10).to_string(index=False))
    
    return clustered_df

def verify_mapping_accuracy():
    """매핑의 정확성 검증"""
    print(f"\n=== 매핑 정확성 검증 ===")
    
    # 원본 데이터와 클러스터링 데이터 비교
    labels_df = pd.read_csv('labels.csv')
    clustered_df = pd.read_csv('two-clusters/label_clusters.csv')
    
    # 각 파일의 유니크한 한국어 라벨 수 비교
    original_unique_labels = set(labels_df['한국어'].dropna().unique())
    clustered_unique_labels = set(clustered_df['한국어'].dropna().unique())
    
    print(f"원본 파일의 유니크한 한국어 라벨 수: {len(original_unique_labels)}")
    print(f"클러스터링 파일의 유니크한 한국어 라벨 수: {len(clustered_unique_labels)}")
    
    # 차이점 확인
    only_in_original = original_unique_labels - clustered_unique_labels
    only_in_clustered = clustered_unique_labels - original_unique_labels
    
    if only_in_original:
        print(f"원본에만 있는 라벨 수: {len(only_in_original)}")
        if len(only_in_original) <= 10:
            print("원본에만 있는 라벨들:")
            for label in only_in_original:
                print(f"  - {label}")
    
    if only_in_clustered:
        print(f"클러스터링에만 있는 라벨 수: {len(only_in_clustered)}")
        if len(only_in_clustered) <= 10:
            print("클러스터링에만 있는 라벨들:")
            for label in only_in_clustered:
                print(f"  - {label}")
    
    # 파일명 매핑 검증 샘플
    print(f"\n=== 파일명 매핑 검증 샘플 ===")
    sample_labels = clustered_df['한국어'].drop_duplicates().head(5)
    
    for label in sample_labels:
        if pd.notna(label):
            # 클러스터링 파일에서의 파일명
            clustered_filename = clustered_df[clustered_df['한국어'] == label]['파일명'].iloc[0]
            
            # 원본 파일에서의 첫 번째 파일명
            original_first_filename = labels_df[labels_df['한국어'] == label]['파일명'].iloc[0]
            
            # 원본 파일에서 해당 라벨의 총 개수
            original_count = len(labels_df[labels_df['한국어'] == label])
            
            print(f"라벨: '{label}'")
            print(f"  클러스터링 파일의 파일명: {clustered_filename}")
            print(f"  원본 파일의 첫 번째 파일명: {original_first_filename}")
            print(f"  원본 파일에서 이 라벨의 총 개수: {original_count}")
            print(f"  매핑 일치: {'✓' if clustered_filename == original_first_filename else '✗'}")
            print()

def create_unique_labels_clean():
    """
    labels.csv에서 유니크한 한국어 라벨을 추출하고 정리하여 
    unique_labels_clean.csv 파일을 생성
    """
    
    print("=== 유니크 한국어 라벨 추출 시작 ===")
    
    # 1. CSV 파일 읽기
    print("labels.csv 파일을 읽는 중...")
    if not os.path.exists('labels.csv'):
        raise FileNotFoundError("labels.csv 파일을 찾을 수 없습니다. 먼저 labels.csv 파일이 있는지 확인해주세요.")
    
    df = pd.read_csv('labels.csv')
    print(f"원본 데이터: {len(df)}행")
    
    # 2. 한국어 칼럼의 유니크한 값들 추출
    print("한국어 칼럼의 유니크한 값들을 추출하는 중...")
    unique_labels = df['한국어'].unique()
    print(f"원본 유니크 라벨 수: {len(unique_labels)}")
    
    # 3. nan 값과 공백 문자 제거
    print("nan 값과 공백 문자를 제거하는 중...")
    filtered_labels = []
    
    for label in unique_labels:
        if pd.notna(label) and str(label).strip() != '':
            filtered_labels.append(str(label).strip())
    
    # 4. 중복 제거 (strip 후에도 중복될 수 있음)
    print("중복 제거 및 정렬하는 중...")
    filtered_labels = list(set(filtered_labels))
    filtered_labels.sort()  # 알파벳순 정렬
    
    print(f"정리된 유니크 라벨 수: {len(filtered_labels)}")
    
    # 5. 유니크한 값들을 데이터프레임으로 만들기
    print("데이터프레임 생성 중...")
    unique_df = pd.DataFrame({'한국어': filtered_labels})
    
    # 6. 새로운 CSV 파일로 저장
    output_filename = 'unique_labels_clean.csv'
    unique_df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"결과가 '{output_filename}'에 저장되었습니다.")
    
    # 7. 결과 요약
    print(f"\n=== 유니크 라벨 추출 완료 ===")
    print(f"원본 데이터 행 수: {len(df)}")
    print(f"원본 유니크 라벨 수: {len(unique_labels)}")
    print(f"정리된 유니크 라벨 수: {len(filtered_labels)}")
    print(f"제거된 라벨 수: {len(unique_labels) - len(filtered_labels)}")
    
    # 8. 샘플 출력
    print(f"\n=== 처음 10개 라벨 ===")
    for i, label in enumerate(filtered_labels[:10]):
        print(f"{i+1:2d}. {label}")
    
    if len(filtered_labels) > 10:
        print(f"\n=== 마지막 10개 라벨 ===")
        for i, label in enumerate(filtered_labels[-10:]):
            print(f"{len(filtered_labels)-9+i:2d}. {label}")
    
    return unique_df

def main():
    # 0. 전처리: unique_labels_clean.csv 파일 확인 및 생성
    print("=== 한국어 레이블 클러스터링 & 파일명 매핑 시작 ===")
    
    if not os.path.exists('unique_labels_clean.csv'):
        print("unique_labels_clean.csv 파일이 없습니다. 자동으로 생성합니다...")
        create_unique_labels_clean()
    else:
        print("✓ unique_labels_clean.csv 파일이 이미 존재합니다.")
    
    # 1. 데이터 로드
    print(f"\n=== 데이터 로드 ===")
    df = pd.read_csv('unique_labels_clean.csv')
    texts = df['한국어'].tolist()
    print(f"로드된 레이블 수: {len(texts)}")
    
    # 2. 모델 로드
    model = load_korean_model()
    
    # 3. 임베딩 생성
    embeddings = create_embeddings(texts, model)
    print(f"임베딩 차원: {embeddings.shape}")
    
    # 4. 클러스터링 수행
    print("\n재귀적 클러스터링을 수행하는 중...")
    clusters = recursive_clustering(texts, embeddings, max_cluster_size=12)
    
    # 5. 결과 분석
    sizes = analyze_clusters(clusters)
    
    # 6. 샘플 출력
    create_sample_display(clusters)
    
    # 7. CSV로 저장
    df_result = save_clusters_to_csv(clusters)
    
    # 8. 시각화
    visualize_clusters_2d(embeddings, clusters, texts)
    
    # 9. 요약 통계를 JSON으로 저장
    summary = {
        'total_labels': len(texts),
        'total_clusters': len(clusters),
        'max_cluster_size': max(sizes),
        'min_cluster_size': min(sizes),
        'avg_cluster_size': float(np.mean(sizes)),
        'cluster_size_distribution': dict(defaultdict(int))
    }
    
    for size in sizes:
        summary['cluster_size_distribution'][str(size)] = summary['cluster_size_distribution'].get(str(size), 0) + 1
    
    with open('two-clusters/clustering_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 클러스터링 완료 ===")
    print(f"총 {len(texts)}개 레이블이 {len(clusters)}개 클러스터로 분류되었습니다.")
    print(f"최대 클러스터 크기: {max(sizes)}개")
    
    # 10. 파일명 매핑 수행
    result_df = merge_cluster_with_filenames()
    
    # 11. 매핑 정확성 검증
    verify_mapping_accuracy()
    
    print("\n=== 전체 작업 완료 ===")
    print("클러스터링과 파일명 매핑이 모두 완료되었습니다!")
    print("최종 결과 파일: label_clusters.csv")

if __name__ == "__main__":
    main() 