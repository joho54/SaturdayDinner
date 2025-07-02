#!/usr/bin/env python3
"""
Motion Extraction System - Test Script
소수의 영상으로 시스템 테스트
"""

from extractor import MotionExtractor, MotionEmbedder, MotionClusterer
import pandas as pd

def test_label_extraction():
    """라벨 추출 테스트"""
    print("🧪 라벨 추출 테스트...")
    
    extractor = MotionExtractor()
    labels_dict = extractor.extract_unique_labels_with_first_files("labels.csv")
    
    print(f"총 {len(labels_dict)}개의 유니크한 라벨 발견")
    
    # 첫 5개만 테스트용으로 사용
    test_labels = dict(list(labels_dict.items())[:5])
    
    print("테스트용 라벨:")
    for label, filename in test_labels.items():
        print(f"  {filename} -> {label}")
        
    return test_labels

def test_video_path_resolution(test_labels):
    """비디오 경로 해석 테스트"""
    print("\n🧪 비디오 경로 해석 테스트...")
    
    from video_path_utils import get_video_root_and_path
    
    found_videos = {}
    
    for label, filename in test_labels.items():
        video_path = get_video_root_and_path(filename, verbose=True)
        if video_path:
            found_videos[label] = (filename, video_path)
            print(f"✅ {filename} -> {video_path}")
        else:
            print(f"❌ {filename} -> 경로를 찾을 수 없습니다")
            
    return found_videos

def test_keypoint_extraction(found_videos):
    """키포인트 추출 테스트 (1개 영상만)"""
    print("\n🧪 키포인트 추출 테스트...")
    
    if not found_videos:
        print("❌ 테스트할 영상이 없습니다.")
        return None
        
    # 첫 번째 영상만 테스트
    label, (filename, video_path) = list(found_videos.items())[0]
    
    print(f"테스트 영상: {filename} ({label})")
    
    extractor = MotionExtractor()
    sequence = extractor.extract_keypoints_from_video(video_path)
    
    if sequence:
        print(f"✅ 키포인트 추출 성공:")
        print(f"   프레임 수: {sequence.frame_count}")
        print(f"   FPS: {sequence.fps}")
        print(f"   시퀸스 모양: {sequence.sequence.shape}")
        print(f"   포즈 랜드마크: {sequence.pose_landmarks.shape}")
        print(f"   왼손 랜드마크: {sequence.left_hand_landmarks.shape}")
        print(f"   오른손 랜드마크: {sequence.right_hand_landmarks.shape}")
        print(f"   얼굴 랜드마크: {sequence.face_landmarks.shape}")
        
        # 시퀸스 저장 테스트
        try:
            saved_path = extractor.save_sequence(sequence, label)
            print(f"✅ 시퀸스 저장 성공: {saved_path}")
            return saved_path
        except Exception as e:
            print(f"❌ 시퀸스 저장 실패: {str(e)}")
            return None
    else:
        print("❌ 키포인트 추출 실패")
        return None

def test_embedding_and_clustering(sequence_path):
    """임베딩 및 클러스터링 테스트"""
    print("\n🧪 임베딩 및 클러스터링 테스트...")
    
    if not sequence_path:
        print("❌ 테스트할 시퀸스가 없습니다.")
        return
        
    embedder = MotionEmbedder()
    clusterer = MotionClusterer(embedder)
    
    # 시퀸스 로드
    sequences = clusterer.load_sequences([sequence_path])
    
    if sequences:
        sequence = sequences[0]
        print(f"✅ 시퀸스 로드 성공: {sequence.filename}")
        
        # 동적 특성 추출 테스트
        try:
            features = embedder.extract_dynamic_features(sequence)
            print(f"✅ 특성 추출 성공: {features.shape}")
            print(f"   특성 벡터 크기: {len(features)}")
            print(f"   특성 범위: [{features.min():.3f}, {features.max():.3f}]")
        except Exception as e:
            print(f"❌ 특성 추출 실패: {str(e)}")
            
    else:
        print("❌ 시퀸스 로드 실패")

def main():
    """테스트 메인 함수"""
    print("🚀 Motion Extraction System - 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 라벨 추출 테스트
        test_labels = test_label_extraction()
        
        # 2. 비디오 경로 해석 테스트
        found_videos = test_video_path_resolution(test_labels)
        
        # 3. 키포인트 추출 테스트
        sequence_path = test_keypoint_extraction(found_videos)
        
        # 4. 임베딩 테스트
        test_embedding_and_clustering(sequence_path)
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("=" * 50)

if __name__ == "__main__":
    main() 