import sys
import os
import json
import pandas as pd
from collections import defaultdict
import numpy as np

# config.py에서 상수들 가져오기
from config import (
    LABEL_MAX_SAMPLES_PER_CLASS,
    MIN_SAMPLES_PER_CLASS,
    AUGMENTATIONS_PER_VIDEO,
    get_action_index
)

# main.py에서 함수들 가져오기
from main import (
    validate_video_roots,
    get_video_root_and_path,
    extract_and_cache_label_data_optimized,
    generate_balanced_none_class_data
)


def main():
    """메인 실행 함수"""
    params = sys.argv[1]
    with open(params, "r") as f:
        params = json.load(f)
    label_dict = params["label_dict"]

    ACTIONS = list(label_dict.keys())

    print(f"🔧 라벨 목록: {ACTIONS}")
    # 1. 비디오 루트 디렉토리 검증
    valid_roots = validate_video_roots()
    if not valid_roots:
        print("❌ 유효한 비디오 루트 디렉토리가 없습니다.")
        sys.exit(1)

    # 2. labels.csv 파일 읽기 및 검증
    if not os.path.exists("labels.csv"):
        print("❌ labels.csv 파일이 없습니다.")
        sys.exit(1)

    labels_df = pd.read_csv("labels.csv")
    print(f"📊 labels.csv 로드 완료: {len(labels_df)}개 항목")
    print(labels_df.head())

    # 3. 파일명에서 비디오 루트 경로 추출 (개선된 방식)
    print("\n🔍 파일명 분석 및 경로 매핑 중...")
    file_mapping = {}
    found_files = 0
    missing_files = 0
    filtered_files = 0

    # 라벨별로 파일을 모아서 최대 개수만큼만 샘플링
    label_to_files = defaultdict(list)
    for idx, row in labels_df.iterrows():
        filename = row["파일명"]
        label = row["한국어"]
        if label not in ACTIONS:
            continue
        file_path = get_video_root_and_path(filename)
        if file_path:
            label_to_files[label].append((filename, file_path))
            found_files += 1
            filtered_files += 1
        else:
            missing_files += 1

    # 최대 개수만큼만 샘플링
    for label in ACTIONS:
        files = label_to_files[label]
        if LABEL_MAX_SAMPLES_PER_CLASS is not None:
            files = files[:LABEL_MAX_SAMPLES_PER_CLASS]
        for filename, file_path in files:
            file_mapping[filename] = {"path": file_path, "label": label}

    # [수정] 라벨별 원본 영상 개수 체크 및 최소 개수 미달 시 학습 중단 (None은 예외)
    insufficient_labels = []
    for label in ACTIONS:
        num_samples = len(label_to_files[label])
        if num_samples < MIN_SAMPLES_PER_CLASS:
            insufficient_labels.append((label, num_samples))
    if insufficient_labels:
        print("\n❌ 최소 샘플 개수 미달 라벨 발견! 학습을 중단합니다.")
        for label, count in insufficient_labels:
            print(f"   - {label}: {count}개 (최소 필요: {MIN_SAMPLES_PER_CLASS}개)")
        sys.exit(1)

    print(f"\n📊 파일 매핑 결과:")
    print(f"   ✅ 찾은 파일: {found_files}개")
    print(f"   ❌ 누락된 파일: {missing_files}개")
    print(f"   🎯 ACTIONS 라벨에 해당하는 파일: {filtered_files}개")
    print(f"   ⚡ 라벨별 최대 {LABEL_MAX_SAMPLES_PER_CLASS}개 파일만 사용")
    print(f"   ⚡ 라벨별 최소 {MIN_SAMPLES_PER_CLASS}개 파일 필요")

    if len(file_mapping) == 0:
        print("❌ 찾을 수 있는 파일이 없습니다.")
        sys.exit(1)

    # 4. 라벨별 데이터 추출 및 캐싱 (개별 처리)
    print("\n🚀 라벨별 데이터 추출 및 캐싱 시작...")

    # None 클래스 제외한 다른 클래스들의 평균 개수 계산
    other_class_counts = {}
    for filename, info in file_mapping.items():
        label = info["label"]
        other_class_counts[label] = other_class_counts.get(label, 0) + 1

    if other_class_counts:
        avg_other_class_count = sum(other_class_counts.values()) / len(
            other_class_counts
        )
        target_none_count = int(avg_other_class_count * (1 + AUGMENTATIONS_PER_VIDEO))
        print(
            f"📊 다른 클래스 평균: {avg_other_class_count:.1f}개 → None 클래스 목표: {target_none_count}개"
        )
    else:
        target_none_count = None
        print(f"📊 다른 클래스가 없음 → None 클래스 기본값 사용")

    X = []
    y = []

    for label in ACTIONS:
        print(f"\n{'='*50}")
        print(f"📋 {label} 라벨 처리 중...")
        print(f"{'='*50}")

        label_data = extract_and_cache_label_data_optimized(file_mapping, label)

        if label_data:
            label_index = get_action_index(label, ACTIONS)
            X.extend(label_data)
            y.extend([label_index] * len(label_data))
            print(f"✅ {label}: {len(label_data)}개 샘플 추가됨")
        else:
            print(f"⚠️ {label}: 데이터가 없습니다.")

    print(f"\n{'='*50}")
    print(f"📊 최종 데이터 통계:")
    print(f"{'='*50}")
    print(f"총 샘플 수: {len(X)}")

    # 클래스별 샘플 수 확인
    unique, counts = np.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        if 0 <= class_idx < len(ACTIONS):
            print(f"클래스 {class_idx} ({ACTIONS[class_idx]}): {count}개")
        else:
            print(f"클래스 {class_idx} (Unknown): {count}개")


if __name__ == "__main__":
    main()