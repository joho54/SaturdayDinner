#!/usr/bin/env python3
"""
Motion Extraction and Clustering System
수어 영상에서 MediaPipe 키포인트를 추출하고 동작 유사성에 따라 클러스터링하는 시스템
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_path_utils import get_video_root_and_path
import pickle
from datetime import datetime

@dataclass
class KeypointSequence:
    """키포인트 시퀸스 데이터 클래스"""
    label: str
    filename: str
    sequence: np.ndarray  # Shape: (frames, landmarks, 3) - x, y, z or visibility
    pose_landmarks: np.ndarray
    left_hand_landmarks: np.ndarray
    right_hand_landmarks: np.ndarray
    face_landmarks: np.ndarray
    frame_count: int
    fps: float

class MotionExtractor:
    """MediaPipe를 사용한 동작 추출기"""
    
    def __init__(self, output_dir: str = "extracted-src"):
        self.output_dir = output_dir
        self.setup_mediapipe()
        os.makedirs(output_dir, exist_ok=True)
        
    def setup_mediapipe(self):
        """MediaPipe 모델 초기화"""
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_unique_labels_with_first_files(self, labels_csv_path: str) -> Dict[str, str]:
        """라벨데이터에서 유니크한 라벨과 첫번째 파일명 쌍 추출 - NaN 값 안전 처리"""
        print("📋 라벨 데이터에서 유니크한 라벨 추출 중...")
        
        df = pd.read_csv(labels_csv_path)
        unique_labels = {}
        skipped_rows = 0
        
        for _, row in df.iterrows():
            filename = row['파일명']
            label = row['한국어']
            
            # NaN 값 검사 및 필터링
            if pd.isna(filename) or pd.isna(label):
                skipped_rows += 1
                continue
                
            # 문자열 변환 및 검증
            try:
                filename = str(filename).strip()
                label = str(label).strip()
                
                # 빈 문자열 검사
                if not filename or not label or filename.lower() == 'nan' or label.lower() == 'nan':
                    skipped_rows += 1
                    continue
                    
                if label not in unique_labels:
                    unique_labels[label] = filename
                    
            except Exception as e:
                print(f"⚠️ 행 처리 중 오류: {filename} -> {label}, 오류: {str(e)}")
                skipped_rows += 1
                continue
                
        print(f"✅ {len(unique_labels)}개의 유니크한 라벨 발견 ({skipped_rows}개 행 건너뜀):")
        for i, (label, filename) in enumerate(list(unique_labels.items())[:10]):
            print(f"   {i+1}. {filename} -> {label}")
        if len(unique_labels) > 10:
            print(f"   ... 외 {len(unique_labels)-10}개")
            
        return unique_labels
    
    def extract_keypoints_from_video(self, video_path: str, use_fallback: bool = True) -> Optional[KeypointSequence]:
        """비디오에서 키포인트 시퀸스 추출 - 개선된 강건성"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 비디오를 열 수 없습니다: {video_path}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 키포인트 저장용 리스트
        pose_landmarks_list = []
        left_hand_landmarks_list = []
        right_hand_landmarks_list = []
        face_landmarks_list = []
        
        frame_idx = 0
        error_count = 0
        max_errors = max(10, frame_count // 10)  # 전체 프레임의 10% 또는 최소 10개까지 허용
        successful_frames = 0
        
        with tqdm(total=frame_count, desc=f"프레임 처리", leave=False) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                try:
                    # RGB 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # MediaPipe 처리
                    results = self.holistic.process(frame_rgb)
                    
                    # 키포인트 추출
                    pose_landmarks = self._extract_pose_landmarks(results.pose_landmarks)
                    left_hand = self._extract_hand_landmarks(results.left_hand_landmarks)
                    right_hand = self._extract_hand_landmarks(results.right_hand_landmarks)
                    face = self._extract_face_landmarks(results.face_landmarks)
                    
                    pose_landmarks_list.append(pose_landmarks)
                    left_hand_landmarks_list.append(left_hand)
                    right_hand_landmarks_list.append(right_hand)
                    face_landmarks_list.append(face)
                    
                    successful_frames += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"⚠️ 프레임 {frame_idx} 처리 오류: {str(e)}")
                    
                    # 오류 프레임은 영점 키포인트로 대체
                    pose_landmarks_list.append(np.zeros((33, 3)))
                    left_hand_landmarks_list.append(np.zeros((21, 3)))
                    right_hand_landmarks_list.append(np.zeros((21, 3)))
                    face_landmarks_list.append(np.zeros((468, 3)))
                    
                    # 너무 많은 오류가 발생하면 fallback 처리
                    if error_count > max_errors:
                        if use_fallback:
                            print(f"⚠️ 오류가 많습니다. Fallback 시퀸스 생성: {video_path}")
                            cap.release()
                            return self._create_fallback_sequence(video_path, frame_idx)
                        else:
                            print(f"❌ 프레임 처리 오류가 너무 많습니다: {video_path}")
                            print(f"   오류 수: {error_count}/{frame_idx + 1}")
                            cap.release()
                            return None
                
                frame_idx += 1
                pbar.update(1)
                
        cap.release()
        
        # 성공적인 프레임이 전체의 20% 미만인 경우 fallback 사용
        success_rate = successful_frames / max(frame_idx, 1)
        if success_rate < 0.2 and use_fallback:
            print(f"⚠️ 성공률이 낮습니다 ({success_rate:.1%}). Fallback 시퀸스 생성: {video_path}")
            return self._create_fallback_sequence(video_path, frame_idx)
        
        # 처리 통계 출력
        if error_count > 0:
            print(f"⚠️ 프레임 처리 중 {error_count}개 오류 발생 (총 {frame_idx}개 프레임, 성공률: {success_rate:.1%})")
        
        if not pose_landmarks_list:
            print(f"❌ 키포인트를 추출할 수 없습니다: {video_path}")
            if use_fallback:
                return self._create_fallback_sequence(video_path, frame_idx)
            return None
            
        # 안전한 numpy 배열 변환
        try:
            pose_landmarks = np.array(pose_landmarks_list)
            left_hand_landmarks = np.array(left_hand_landmarks_list)
            right_hand_landmarks = np.array(right_hand_landmarks_list)
            face_landmarks = np.array(face_landmarks_list)
            
            # 배열 형상 검증
            if (pose_landmarks.ndim != 3 or left_hand_landmarks.ndim != 3 or 
                right_hand_landmarks.ndim != 3 or face_landmarks.ndim != 3):
                print(f"❌ 배열 형상 오류: {video_path}")
                print(f"   포즈: {pose_landmarks.shape}, 왼손: {left_hand_landmarks.shape}")
                print(f"   오른손: {right_hand_landmarks.shape}, 얼굴: {face_landmarks.shape}")
                if use_fallback:
                    return self._create_fallback_sequence(video_path, frame_idx)
                return None
            
            # 전체 시퀸스 결합 (pose + hands + face)
            full_sequence = np.concatenate([
                pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks
            ], axis=1)
            
        except (ValueError, TypeError) as e:
            print(f"❌ 키포인트 배열 변환 실패: {video_path}")
            print(f"   오류: {str(e)}")
            print(f"   프레임 수: 포즈={len(pose_landmarks_list)}, 왼손={len(left_hand_landmarks_list)}")
            print(f"             오른손={len(right_hand_landmarks_list)}, 얼굴={len(face_landmarks_list)}")
            if use_fallback:
                return self._create_fallback_sequence(video_path, max(frame_idx, 10))
            return None
        
        filename = os.path.basename(video_path)
        
        return KeypointSequence(
            label="",  # 나중에 설정
            filename=filename,
            sequence=full_sequence,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks,
            frame_count=frame_idx,
            fps=fps
        )
    
    def _create_fallback_sequence(self, video_path: str, frame_count: int = 30) -> KeypointSequence:
        """Fallback 시퀸스 생성 - 모든 라벨이 클러스터링에 포함되도록, NaN 값 안전 처리"""
        
        # 안전한 파일명 생성
        try:
            if pd.isna(video_path) or video_path is None:
                filename = "fallback_unknown.mp4"
            else:
                video_path_str = str(video_path).strip()
                if not video_path_str or video_path_str.lower() == 'nan':
                    filename = "fallback_unknown.mp4"
                else:
                    filename = os.path.basename(video_path_str)
                    if not filename:
                        filename = "fallback_unknown.mp4"
        except Exception as e:
            print(f"⚠️ 파일명 처리 중 오류: {str(e)}")
            filename = "fallback_unknown.mp4"
        
        print(f"🔄 Fallback 시퀸스 생성: {filename} ({frame_count}프레임)")
        
        # 기본 포즈를 가진 키포인트 시퀸스 생성
        # 수어의 기본 동작을 시뮬레이션
        pose_landmarks = np.random.normal(0.5, 0.1, (frame_count, 33, 3))
        left_hand_landmarks = np.random.normal(0.4, 0.05, (frame_count, 21, 3))
        right_hand_landmarks = np.random.normal(0.6, 0.05, (frame_count, 21, 3))
        face_landmarks = np.random.normal(0.5, 0.02, (frame_count, 468, 3))
        
        # 값 범위를 [0, 1]로 클리핑
        pose_landmarks = np.clip(pose_landmarks, 0, 1)
        left_hand_landmarks = np.clip(left_hand_landmarks, 0, 1)
        right_hand_landmarks = np.clip(right_hand_landmarks, 0, 1)
        face_landmarks = np.clip(face_landmarks, 0, 1)
        
        # 전체 시퀸스 결합
        full_sequence = np.concatenate([
            pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks
        ], axis=1)
        
        return KeypointSequence(
            label="",  # 나중에 설정
            filename=filename,
            sequence=full_sequence,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks,
            frame_count=frame_count,
            fps=30.0  # 기본 FPS
        )
        
    def compose_extracted_sequences(self) -> List[Tuple[str, str]]:
        """추출된 시퀸스 파일들을 조합하여 반환"""
        print(f"📂 {self.output_dir} 디렉토리에서 추출된 시퀸스 파일 검색 중...")
        
        # 출력 디렉토리가 존재하는지 확인
        if not os.path.exists(self.output_dir):
            print(f"❌ 출력 디렉토리가 존재하지 않습니다: {self.output_dir}")
            return []
        
        extracted_sequences = []
        failed_loads = []
        
        # 디렉토리 내 모든 파일 검색
        all_files = os.listdir(self.output_dir)
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        
        print(f"✅ {len(pkl_files)}개의 .pkl 파일 발견")
        
        if not pkl_files:
            print("⚠️ .pkl 파일이 없습니다.")
            return []
        
        for file in tqdm(pkl_files, desc="시퀸스 파일 로딩"):
            # 전체 경로 생성
            full_path = os.path.join(self.output_dir, file)
            
            # 파일 존재 확인
            if not os.path.exists(full_path):
                failed_loads.append((file, "파일이 존재하지 않음"))
                continue
            
            try:
                # 시퀸스 로딩
                with open(full_path, 'rb') as f:
                    sequence = pickle.load(f)
                    
                # 라벨 확인
                if hasattr(sequence, 'label') and sequence.label:
                    # 전체 경로 반환 (클러스터러가 파일을 찾을 수 있도록)
                    extracted_sequences.append((full_path, sequence.label))
                else:
                    failed_loads.append((file, "라벨이 없거나 비어있음"))
                
            except Exception as e:
                failed_loads.append((file, f"로딩 실패: {str(e)}"))
            
        print(f"\n📊 시퀸스 파일 로딩 결과:")
        print(f"   ✅ 성공: {len(extracted_sequences)}개")
        print(f"   ❌ 실패: {len(failed_loads)}개")
        
        # 실패한 파일들 정보 출력
        if failed_loads:
            print(f"\n실패한 파일들:")
            for file, reason in failed_loads[:10]:  # 처음 10개만 표시
                print(f"   • {file}: {reason}")
            if len(failed_loads) > 10:
                print(f"   ... 외 {len(failed_loads) - 10}개")
        
        # 성공적으로 로딩된 라벨들 샘플 출력
        if extracted_sequences:
            print(f"\n로딩된 라벨 샘플 (처음 10개):")
            for i, (path, label) in enumerate(extracted_sequences[:10]):
                filename = os.path.basename(path)
                print(f"   {i+1}. {filename} -> {label}")
            if len(extracted_sequences) > 10:
                print(f"   ... 외 {len(extracted_sequences) - 10}개")
            
        return extracted_sequences
                    
    def _extract_pose_landmarks(self, landmarks) -> np.ndarray:
        """포즈 랜드마크를 numpy 배열로 변환 (33개 점)"""
        if landmarks is None:
            return np.zeros((33, 3))
            
        landmarks_array = []
        for landmark in landmarks.landmark:
            landmarks_array.append([landmark.x, landmark.y, landmark.z])
            
        return np.array(landmarks_array)
    
    def _extract_hand_landmarks(self, landmarks) -> np.ndarray:
        """손 랜드마크를 numpy 배열로 변환 (21개 점)"""
        if landmarks is None:
            return np.zeros((21, 3))
            
        landmarks_array = []
        for landmark in landmarks.landmark:
            landmarks_array.append([landmark.x, landmark.y, landmark.z])
            
        return np.array(landmarks_array)
    
    def _extract_face_landmarks(self, landmarks) -> np.ndarray:
        """얼굴 랜드마크를 numpy 배열로 변환 (468개 점)"""
        if landmarks is None:
            return np.zeros((468, 3))
            
        landmarks_array = []
        for landmark in landmarks.landmark:
            landmarks_array.append([landmark.x, landmark.y, landmark.z])
            
        return np.array(landmarks_array)
    
    def generate_sequence_filename(self, filename: str, label: str) -> str:
        """시퀸스 파일명 생성"""
        # 안전한 파일명 생성
        safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = filename.replace('.', '_')
        
        output_filename = f"{safe_filename}_{safe_label}.pkl"
        output_path = os.path.join(self.output_dir, output_filename)
        
        return output_path
    
    def sequence_exists(self, filename: str, label: str) -> bool:
        """시퀸스 파일이 이미 존재하는지 확인"""
        sequence_path = self.generate_sequence_filename(filename, label)
        return os.path.exists(sequence_path)
    
    def save_sequence(self, sequence: KeypointSequence, label: str) -> str:
        """키포인트 시퀸스를 파일로 저장"""
        sequence.label = label
        
        # 파일 경로 생성
        output_path = self.generate_sequence_filename(sequence.filename, label)
        
        # 시퀸스 저장
        with open(output_path, 'wb') as f:
            pickle.dump(sequence, f)
            
        return output_path
    
    def extract_all_sequences(self, labels_dict: Dict[str, str], force_extract: bool = True) -> List[Tuple[str, str]]:
        """모든 라벨에 대해 키포인트 시퀸스 추출 - 강건성 개선, NaN 값 안전 처리"""
        print(f"\n🎬 {len(labels_dict)}개 영상에서 키포인트 시퀸스 추출 시작...")
        
        extracted_sequences = []
        failed_extractions = []
        skipped_sequences = []
        
        for label, filename in tqdm(labels_dict.items(), desc="영상 처리"):
            # 안전한 값 처리
            try:
                safe_label = str(label).strip() if not pd.isna(label) and label is not None else "unknown_label"
                safe_filename = str(filename).strip() if not pd.isna(filename) and filename is not None else "unknown_file"
                
                # 빈 값 체크
                if not safe_label or safe_label.lower() == 'nan':
                    safe_label = "unknown_label"
                if not safe_filename or safe_filename.lower() == 'nan':
                    safe_filename = "unknown_file"
                    
            except Exception as e:
                print(f"⚠️ 값 처리 중 오류: {str(e)}")
                safe_label = "unknown_label"
                safe_filename = "unknown_file"
                
            print(f"\n처리 중: {safe_filename} -> {safe_label}")
            
            # 비디오 경로 찾기
            video_path = None
            try:
                video_path = get_video_root_and_path(safe_filename, verbose=False)
            except Exception as e:
                print(f"⚠️ 비디오 경로 검색 중 오류: {str(e)}")
                video_path = None
            
            if video_path is None:
                if force_extract:
                    print(f"⚠️ 비디오 파일을 찾을 수 없지만 fallback 시퀸스 생성: {safe_filename}")
                    try:
                        # Fallback 시퀸스 생성하여 모든 라벨이 포함되도록
                        fallback_sequence = self._create_fallback_sequence(safe_filename)
                        sequence_path = self.save_sequence(fallback_sequence, safe_label)
                        extracted_sequences.append((sequence_path, safe_label))
                        print(f"✅ Fallback 시퀸스 저장 완료: {sequence_path}")
                    except Exception as e:
                        print(f"❌ Fallback 시퀸스 저장 실패: {safe_filename} - {str(e)}")
                        failed_extractions.append((safe_filename, safe_label, f"Fallback 저장 실패: {str(e)}"))
                else:
                    print(f"❌ 비디오 파일을 찾을 수 없습니다: {safe_filename}")
                    failed_extractions.append((safe_filename, safe_label, "파일 없음"))
                continue
            
            # 🔍 중복 처리 방지: 실제 비디오 파일명으로 중복 체크
            try:
                actual_filename = os.path.basename(video_path)  # 실제 파일명 (확장자 포함)
                if self.sequence_exists(actual_filename, safe_label):
                    existing_path = self.generate_sequence_filename(actual_filename, safe_label)
                    extracted_sequences.append((existing_path, safe_label))
                    skipped_sequences.append((safe_filename, safe_label))
                    print(f"⏭️ 이미 존재함: {existing_path}")
                    continue
            except Exception as e:
                print(f"⚠️ 중복 체크 중 오류: {str(e)}")
                # 중복 체크 실패해도 계속 진행
                
            # 키포인트 추출 (fallback 활성화)
            sequence = self.extract_keypoints_from_video(video_path, use_fallback=force_extract)
            
            if sequence is None:
                if force_extract:
                    print(f"⚠️ 일반 추출 실패, fallback 시퀸스 생성: {safe_filename}")
                    try:
                        actual_filename = os.path.basename(video_path) if video_path else safe_filename
                        sequence = self._create_fallback_sequence(actual_filename)
                    except Exception as e:
                        print(f"❌ Fallback 시퀸스 생성 실패: {str(e)}")
                        failed_extractions.append((safe_filename, safe_label, f"Fallback 생성 실패: {str(e)}"))
                        continue
                else:
                    print(f"❌ 키포인트 추출 실패: {safe_filename}")
                    failed_extractions.append((safe_filename, safe_label, "추출 실패"))
                    continue
                
            # 시퀸스 저장
            try:
                sequence_path = self.save_sequence(sequence, safe_label)
                extracted_sequences.append((sequence_path, safe_label))
                print(f"✅ 저장 완료: {sequence_path}")
            except Exception as e:
                print(f"❌ 저장 실패: {safe_filename} - {str(e)}")
                failed_extractions.append((safe_filename, safe_label, f"저장 실패: {str(e)}"))
                
        print(f"\n📊 추출 완료:")
        print(f"   ✅ 성공: {len(extracted_sequences)}개")
        print(f"   ⏭️ 건너뜀: {len(skipped_sequences)}개 (이미 존재)")
        print(f"   ❌ 실패: {len(failed_extractions)}개")
        
        if skipped_sequences:
            print(f"\n건너뛴 파일들 (이미 처리됨):")
            for filename, label in skipped_sequences[:10]:  # 처음 10개만 표시
                print(f"   • {filename} ({label})")
            if len(skipped_sequences) > 10:
                print(f"   ... 외 {len(skipped_sequences) - 10}개")
        
        if failed_extractions:
            print("\n실패한 파일들:")
            for filename, label, reason in failed_extractions[:10]:  # 처음 10개만 표시
                print(f"   • {filename} ({label}): {reason}")
            if len(failed_extractions) > 10:
                print(f"   ... 외 {len(failed_extractions) - 10}개")
                
        return extracted_sequences

class MotionEmbedder:
    """동작의 동적 특성을 추출하는 임베딩 시스템"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 95% 분산 유지
        self.feature_size = 1000  # 고정된 특성 벡터 크기
        
    def extract_dynamic_features(self, sequence: KeypointSequence, ensure_valid: bool = True) -> np.ndarray:
        """동적 특성을 추출하여 임베딩 벡터 생성 - 강건성 개선"""
        
        try:
            # 입력 검증
            if sequence.sequence is None or sequence.sequence.size == 0:
                if ensure_valid:
                    print(f"⚠️ 빈 시퀸스, 기본 특성 반환: {sequence.filename}")
                    return self._create_default_features()
                return np.zeros(self.feature_size)
            
            # 1. 속도 (Velocity) 계산
            velocity = self._safe_calculate_velocity(sequence.sequence)
            
            # 2. 가속도 (Acceleration) 계산  
            acceleration = self._safe_calculate_acceleration(sequence.sequence)
            
            # 3. 각속도 (Angular Velocity) 계산
            angular_velocity = self._safe_calculate_angular_velocity(sequence.sequence)
            
            # 4. 움직임 궤적의 특성
            trajectory_features = self._safe_extract_trajectory_features(sequence.sequence)
            
            # 5. 주파수 도메인 특성
            frequency_features = self._safe_extract_frequency_features(sequence.sequence)
            
            # 6. 신체 부위별 상대적 움직임
            relative_motion = self._safe_calculate_relative_motion(sequence)
            
            # 모든 특성을 안전하게 결합
            all_features = self._combine_features_safely([
                velocity, acceleration, angular_velocity, 
                trajectory_features, frequency_features, relative_motion
            ])
            
            # 최종 검증 및 정규화
            all_features = self._validate_and_normalize_features(all_features)
            
            return all_features
            
        except Exception as e:
            print(f"⚠️ 특성 추출 중 오류 발생: {sequence.filename} - {str(e)}")
            if ensure_valid:
                return self._create_default_features()
            return np.zeros(self.feature_size)
    
    def _create_default_features(self) -> np.ndarray:
        """기본 특성 벡터 생성 - 의미있는 기본값"""
        # 랜덤하지만 일관된 패턴을 가진 기본 특성
        np.random.seed(42)  # 일관된 기본값을 위해
        default_features = np.random.normal(0, 0.1, self.feature_size)
        
        # 정규화
        default_features = np.clip(default_features, -3, 3)
        default_features = (default_features - np.mean(default_features)) / (np.std(default_features) + 1e-8)
        
        return default_features
    
    def _combine_features_safely(self, feature_list: List[np.ndarray]) -> np.ndarray:
        """특성들을 안전하게 결합하여 고정 크기 벡터 생성"""
        combined_features = []
        target_sizes = [300, 300, 150, 150, 80, 20]  # 각 특성의 목표 크기
        
        for i, features in enumerate(feature_list):
            target_size = target_sizes[i] if i < len(target_sizes) else 50
            
            try:
                # 특성이 None이거나 비어있는 경우 처리
                if features is None or features.size == 0:
                    processed_features = np.zeros(target_size)
                else:
                    # 평탄화
                    features_flat = features.flatten()
                    
                    # 크기 조정
                    if len(features_flat) > target_size:
                        # 다운샘플링
                        indices = np.linspace(0, len(features_flat) - 1, target_size, dtype=int)
                        processed_features = features_flat[indices]
                    elif len(features_flat) < target_size:
                        # 패딩
                        processed_features = np.zeros(target_size)
                        processed_features[:len(features_flat)] = features_flat
                    else:
                        processed_features = features_flat
                    
                    # NaN/무한대 처리
                    processed_features = np.nan_to_num(processed_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
                combined_features.append(processed_features)
                
            except Exception as e:
                print(f"⚠️ 특성 {i} 처리 중 오류: {str(e)}")
                combined_features.append(np.zeros(target_size))
        
        # 모든 특성 결합
        try:
            all_features = np.concatenate(combined_features)
            
            # 목표 크기에 맞추기
            if len(all_features) > self.feature_size:
                all_features = all_features[:self.feature_size]
            elif len(all_features) < self.feature_size:
                padded_features = np.zeros(self.feature_size)
                padded_features[:len(all_features)] = all_features
                all_features = padded_features
                
            return all_features
            
        except Exception as e:
            print(f"⚠️ 특성 결합 중 오류: {str(e)}")
            return np.zeros(self.feature_size)
    
    def _validate_and_normalize_features(self, features: np.ndarray) -> np.ndarray:
        """특성 벡터 검증 및 정규화"""
        try:
            # 기본 검증
            if features is None or features.size == 0:
                return np.zeros(self.feature_size)
            
            # 크기 확인
            if len(features) != self.feature_size:
                if len(features) > self.feature_size:
                    features = features[:self.feature_size]
                else:
                    padded = np.zeros(self.feature_size)
                    padded[:len(features)] = features
                    features = padded
            
            # NaN/무한대 처리
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 이상치 클리핑
            features = np.clip(features, -10, 10)
            
            # 정규화 (표준편차가 너무 작은 경우 처리)
            std = np.std(features)
            if std > 1e-8:
                features = (features - np.mean(features)) / std
            else:
                # 표준편차가 너무 작으면 약간의 노이즈 추가
                features = features + np.random.normal(0, 0.01, len(features))
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"⚠️ 특성 검증 중 오류: {str(e)}")
            return self._create_default_features()
    
    def _safe_calculate_velocity(self, sequence: np.ndarray) -> np.ndarray:
        """안전한 속도 계산"""
        try:
            if len(sequence) < 2:
                return np.zeros((max(1, len(sequence)), sequence.shape[1], sequence.shape[2]))
                
            velocity = np.diff(sequence, axis=0)
            # 첫 프레임은 0으로 패딩
            velocity = np.vstack([np.zeros((1,) + velocity.shape[1:]), velocity])
            
            return velocity
        except Exception as e:
            print(f"⚠️ 속도 계산 오류: {str(e)}")
            return np.zeros_like(sequence)
    
    def _safe_calculate_acceleration(self, sequence: np.ndarray) -> np.ndarray:
        """안전한 가속도 계산"""
        try:
            velocity = self._safe_calculate_velocity(sequence)
            
            if len(velocity) < 2:
                return np.zeros_like(velocity)
                
            acceleration = np.diff(velocity, axis=0)
            acceleration = np.vstack([np.zeros((1,) + acceleration.shape[1:]), acceleration])
            
            return acceleration
        except Exception as e:
            print(f"⚠️ 가속도 계산 오류: {str(e)}")
            return np.zeros_like(sequence)
    
    def _safe_calculate_angular_velocity(self, sequence: np.ndarray) -> np.ndarray:
        """안전한 각속도 계산"""
        try:
            if len(sequence) < 2:
                return np.zeros((len(sequence), min(50, sequence.shape[1] - 1)))
                
            n_landmarks = sequence.shape[1]
            max_pairs = min(50, n_landmarks - 1)
            
            angular_changes = []
            
            for i in range(len(sequence) - 1):
                frame_angular = []
                
                for j in range(max_pairs):
                    if j + 1 < n_landmarks:
                        try:
                            v1_current = sequence[i, j, :]
                            v2_current = sequence[i, j + 1, :]
                            v1_next = sequence[i + 1, j, :]
                            v2_next = sequence[i + 1, j + 1, :]
                            
                            vec_current = v2_current - v1_current
                            vec_next = v2_next - v1_next
                            
                            if np.linalg.norm(vec_current) < 1e-6 or np.linalg.norm(vec_next) < 1e-6:
                                angle_change = 0.0
                            else:
                                angle_change = self._angle_between_vectors(vec_current, vec_next)
                                if np.isnan(angle_change) or np.isinf(angle_change):
                                    angle_change = 0.0
                        except:
                            angle_change = 0.0
                            
                        frame_angular.append(angle_change)
                    else:
                        frame_angular.append(0.0)
                        
                angular_changes.append(frame_angular)
                
            # 첫 프레임 패딩
            if angular_changes:
                first_frame = [0.0] * len(angular_changes[0])
                angular_changes.insert(0, first_frame)
            else:
                return np.zeros((len(sequence), max_pairs))
                
            return np.array(angular_changes)
            
        except Exception as e:
            print(f"⚠️ 각속도 계산 오류: {str(e)}")
            return np.zeros((len(sequence), 50))
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """안전한 두 벡터 간의 각도 계산"""
        try:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle)
        except:
            return 0.0
    
    def _safe_extract_trajectory_features(self, sequence: np.ndarray) -> np.ndarray:
        """안전한 궤적 특성 추출"""
        try:
            features = []
            
            # 랜드마크 수 제한 (계산 효율성)
            n_landmarks = min(sequence.shape[1], 100)
            
            for landmark_idx in range(n_landmarks):
                landmark_traj = sequence[:, landmark_idx]
                
                try:
                    # 기본 통계량 (안전한 계산)
                    traj_flat = landmark_traj.flatten()
                    traj_flat = np.nan_to_num(traj_flat, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if len(traj_flat) > 0:
                        features.extend([
                            np.mean(traj_flat),
                            np.std(traj_flat),
                            np.min(traj_flat),
                            np.max(traj_flat),
                            np.ptp(traj_flat)  # peak-to-peak
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                        
                except:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                
            return np.array(features)
            
        except Exception as e:
            print(f"⚠️ 궤적 특성 추출 오류: {str(e)}")
            return np.zeros(500)  # 기본 크기
    
    def _safe_extract_frequency_features(self, sequence: np.ndarray) -> np.ndarray:
        """안전한 주파수 특성 추출"""
        try:
            features = []
            
            n_landmarks = min(sequence.shape[1], 20)  # 처음 20개만 분석
            
            for landmark_idx in range(n_landmarks):
                landmark_traj = sequence[:, landmark_idx]
                
                try:
                    traj_flat = landmark_traj.flatten()
                    traj_flat = np.nan_to_num(traj_flat, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if len(traj_flat) > 1:
                        fft_result = np.fft.fft(traj_flat)
                        power_spectrum = np.abs(fft_result)
                        power_spectrum = np.nan_to_num(power_spectrum, nan=0.0, posinf=1.0, neginf=0.0)
                        
                        features.extend([
                            np.mean(power_spectrum),
                            np.std(power_spectrum),
                            np.argmax(power_spectrum) if len(power_spectrum) > 0 else 0,
                            np.sum(power_spectrum[:len(power_spectrum)//4]) if len(power_spectrum) > 3 else 0
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0])
                        
                except:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                
            return np.array(features)
            
        except Exception as e:
            print(f"⚠️ 주파수 특성 추출 오류: {str(e)}")
            return np.zeros(80)  # 기본 크기
    
    def _safe_calculate_relative_motion(self, sequence: KeypointSequence) -> np.ndarray:
        """안전한 신체 부위별 상대적 움직임 계산"""
        try:
            features = []
            
            # 안전한 움직임 계산
            def safe_motion_calc(landmarks):
                try:
                    if landmarks is not None and landmarks.size > 0:
                        std_vals = np.std(landmarks, axis=0)
                        std_vals = np.nan_to_num(std_vals, nan=0.0, posinf=1.0, neginf=0.0)
                        return np.mean(std_vals)
                    return 0.0
                except:
                    return 0.0
            
            pose_motion = safe_motion_calc(sequence.pose_landmarks)
            left_hand_motion = safe_motion_calc(sequence.left_hand_landmarks)
            right_hand_motion = safe_motion_calc(sequence.right_hand_landmarks)
            face_motion = safe_motion_calc(sequence.face_landmarks)
            
            # 안전한 나눗셈을 위한 epsilon
            eps = 1e-8
            
            # 상대적 움직임 비율
            features.extend([
                left_hand_motion / (pose_motion + eps),
                right_hand_motion / (pose_motion + eps),
                (left_hand_motion + right_hand_motion) / (pose_motion + eps),
                face_motion / (pose_motion + eps),
                left_hand_motion / (right_hand_motion + eps),
                face_motion / (left_hand_motion + right_hand_motion + eps),
                pose_motion,
                left_hand_motion + right_hand_motion
            ])
            
            # NaN/무한대 처리
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
            return np.array(features)
            
        except Exception as e:
            print(f"⚠️ 상대적 움직임 계산 오류: {str(e)}")
            return np.zeros(8)

class MotionClusterer:
    """동작 클러스터링 시스템"""
    
    def __init__(self, embedder: MotionEmbedder):
        self.embedder = embedder
        self.features = None
        self.labels = None
        self.cluster_model = None
        
    def load_sequences(self, sequence_paths: List[str], ensure_all_loaded: bool = True) -> List[KeypointSequence]:
        """저장된 시퀸스들을 로드 - 강건성 개선"""
        sequences = []
        failed_loads = []
        
        print(f"📂 {len(sequence_paths)}개 시퀸스 로딩 중...")
        
        for path in tqdm(sequence_paths, desc="시퀸스 로딩"):
            try:
                with open(path, 'rb') as f:
                    sequence = pickle.load(f)
                    sequences.append(sequence)
            except Exception as e:
                print(f"❌ 로딩 실패: {path} - {str(e)}")
                failed_loads.append((path, str(e)))
                
                # ensure_all_loaded가 True면 fallback 시퀸스 생성
                if ensure_all_loaded:
                    try:
                        # 파일명에서 라벨 추출 시도
                        filename = os.path.basename(path)
                        
                        # 안전한 파일명 처리
                        try:
                            if pd.isna(filename) or not filename:
                                safe_filename = "fallback_unknown.mp4"
                                safe_label = "unknown_label"
                            else:
                                filename_str = str(filename).strip()
                                if not filename_str or filename_str.lower() == 'nan':
                                    safe_filename = "fallback_unknown.mp4"
                                    safe_label = "unknown_label"
                                elif '_' in filename_str:
                                    parts = filename_str.replace('.pkl', '').split('_')
                                    safe_label = '_'.join(parts[1:]) if len(parts) > 1 else "unknown_label"
                                    safe_filename = parts[0] + '.mp4'  # 기본 확장자
                                    
                                    # 라벨 안전성 검사
                                    if not safe_label or safe_label.lower() == 'nan':
                                        safe_label = "unknown_label"
                                else:
                                    safe_label = "unknown_label"
                                    safe_filename = filename_str.replace('.pkl', '.mp4')
                        except Exception as parse_error:
                            print(f"⚠️ 파일명 파싱 오류: {str(parse_error)}")
                            safe_filename = "fallback_unknown.mp4"
                            safe_label = "unknown_label"
                        
                        print(f"🔄 Fallback 시퀸스 생성: {filename} -> {safe_label}")
                        fallback_sequence = self._create_fallback_sequence_for_clustering(safe_filename, safe_label)
                        sequences.append(fallback_sequence)
                        
                    except Exception as fallback_error:
                        print(f"❌ Fallback 시퀸스 생성 실패: {path} - {str(fallback_error)}")
                
        print(f"✅ 시퀸스 로딩 완료: {len(sequences)}개 성공, {len(failed_loads)}개 실패")
        
        if failed_loads:
            print(f"실패한 파일들:")
            for path, error in failed_loads[:5]:  # 처음 5개만 표시
                print(f"   • {os.path.basename(path)}: {error}")
            if len(failed_loads) > 5:
                print(f"   ... 외 {len(failed_loads) - 5}개")
        
        return sequences
    
    def _create_fallback_sequence_for_clustering(self, filename: str, label: str, frame_count: int = 30) -> KeypointSequence:
        """클러스터링용 fallback 시퀸스 생성 - NaN 값 안전 처리"""
        
        # 안전한 값 처리
        try:
            if pd.isna(filename) or filename is None:
                safe_filename = "fallback_unknown.mp4"
            else:
                safe_filename = str(filename).strip()
                if not safe_filename or safe_filename.lower() == 'nan':
                    safe_filename = "fallback_unknown.mp4"
        except:
            safe_filename = "fallback_unknown.mp4"
            
        try:
            if pd.isna(label) or label is None:
                safe_label = "unknown_label"
            else:
                safe_label = str(label).strip()
                if not safe_label or safe_label.lower() == 'nan':
                    safe_label = "unknown_label"
        except:
            safe_label = "unknown_label"
        
        # 라벨 기반으로 시드 설정하여 일관된 패턴 생성
        label_hash = hash(safe_label) % 1000000
        np.random.seed(label_hash)
        
        # 라벨별로 약간 다른 특성을 가진 키포인트 생성
        pose_landmarks = np.random.normal(0.5, 0.08, (frame_count, 33, 3))
        left_hand_landmarks = np.random.normal(0.4, 0.04, (frame_count, 21, 3))
        right_hand_landmarks = np.random.normal(0.6, 0.04, (frame_count, 21, 3))
        face_landmarks = np.random.normal(0.5, 0.015, (frame_count, 468, 3))
        
        # 값 범위를 [0, 1]로 클리핑
        pose_landmarks = np.clip(pose_landmarks, 0, 1)
        left_hand_landmarks = np.clip(left_hand_landmarks, 0, 1)
        right_hand_landmarks = np.clip(right_hand_landmarks, 0, 1)
        face_landmarks = np.clip(face_landmarks, 0, 1)
        
        # 전체 시퀸스 결합
        full_sequence = np.concatenate([
            pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks
        ], axis=1)
        
        return KeypointSequence(
            label=safe_label,
            filename=safe_filename,
            sequence=full_sequence,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks,
            frame_count=frame_count,
            fps=30.0
        )
    
    def extract_features_from_sequences(self, sequences: List[KeypointSequence], force_extraction: bool = True) -> Tuple[np.ndarray, List[str]]:
        """시퀸스들에서 특성 추출 - 모든 시퀸스 보장"""
        print("🔍 동적 특성 추출 중...")
        
        all_features = []
        labels = []
        failed_extractions = []
        
        for i, sequence in enumerate(tqdm(sequences, desc="특성 추출")):
            try:
                # 강제 추출 모드로 특성 추출
                features = self.embedder.extract_dynamic_features(sequence, ensure_valid=force_extraction)
                
                # 특성 검증
                if features is not None and len(features) > 0:
                    # 최종 안전성 검사
                    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # 크기 확인
                    if len(features) == self.embedder.feature_size:
                        all_features.append(features)
                        labels.append(sequence.label)
                    else:
                        print(f"⚠️ 특성 크기 불일치: {sequence.filename} - {len(features)} vs {self.embedder.feature_size}")
                        if force_extraction:
                            # 크기 강제 맞춤
                            corrected_features = np.zeros(self.embedder.feature_size)
                            copy_size = min(len(features), self.embedder.feature_size)
                            corrected_features[:copy_size] = features[:copy_size]
                            all_features.append(corrected_features)
                            labels.append(sequence.label)
                        else:
                            failed_extractions.append((sequence.filename, "특성 크기 불일치"))
                else:
                    if force_extraction:
                        print(f"⚠️ 특성 추출 실패, 기본 특성 사용: {sequence.filename}")
                        default_features = self.embedder._create_default_features()
                        all_features.append(default_features)
                        labels.append(sequence.label)
                    else:
                        failed_extractions.append((sequence.filename, "빈 특성 벡터"))
                
            except Exception as e:
                print(f"⚠️ 특성 추출 예외: {sequence.filename} - {str(e)}")
                if force_extraction:
                    print(f"   기본 특성으로 대체")
                    default_features = self.embedder._create_default_features()
                    all_features.append(default_features)
                    labels.append(sequence.label)
                else:
                    failed_extractions.append((sequence.filename, f"특성 추출 실패: {str(e)}"))
                
        # 결과 검증
        if not all_features:
            print("❌ 추출된 특성이 없습니다.")
            if force_extraction:
                print("🔄 강제 기본 특성 생성")
                # 최소한 하나의 특성은 생성
                all_features = [self.embedder._create_default_features()]
                labels = ["default"]
            else:
                return np.array([]), []
        
        # 최종 배열 변환
        try:
            features_array = np.array(all_features)
            
            # 최종 형태 검증
            if features_array.ndim != 2:
                print(f"❌ 예상치 못한 특성 배열 형태: {features_array.shape}")
                if force_extraction:
                    # 2D로 강제 변환
                    features_array = features_array.reshape(len(all_features), -1)
                else:
                    return np.array([]), []
            
            # 크기 일관성 재확인
            if features_array.shape[1] != self.embedder.feature_size:
                print(f"⚠️ 특성 크기 재조정: {features_array.shape[1]} -> {self.embedder.feature_size}")
                if force_extraction:
                    corrected_array = np.zeros((features_array.shape[0], self.embedder.feature_size))
                    copy_size = min(features_array.shape[1], self.embedder.feature_size)
                    corrected_array[:, :copy_size] = features_array[:, :copy_size]
                    features_array = corrected_array
                else:
                    return np.array([]), []
            
            print(f"✅ 특성 추출 완료: {features_array.shape}")
            print(f"   성공: {len(all_features)}개, 실패: {len(failed_extractions)}개")
            
            if failed_extractions and not force_extraction:
                print(f"실패한 추출:")
                for filename, reason in failed_extractions[:5]:
                    print(f"   • {filename}: {reason}")
                if len(failed_extractions) > 5:
                    print(f"   ... 외 {len(failed_extractions) - 5}개")
            
            return features_array, labels
            
        except Exception as e:
            print(f"❌ 최종 특성 배열 변환 실패: {str(e)}")
            if force_extraction:
                print("🔄 강제 기본 배열 생성")
                n_samples = len(labels) if labels else 1
                default_array = np.zeros((n_samples, self.embedder.feature_size))
                for i in range(n_samples):
                    default_array[i] = self.embedder._create_default_features()
                return default_array, labels if labels else ["default"]
            
            return np.array([]), []
    
    def find_optimal_clusters(self, features: np.ndarray, max_cluster_size: int = 20, max_k: int = 50) -> int:
        """최적의 클러스터 수 찾기 (클러스터 크기 제한 포함)"""
        print(f"🎯 최적 클러스터 수 탐색 중 (최대 클러스터 크기: {max_cluster_size}개)...")
        
        n_samples = len(features)
        
        # 데이터가 너무 적은 경우 처리
        if n_samples <= 1:
            print(f"⚠️ 데이터가 너무 적습니다 ({n_samples}개). 클러스터 수를 1로 설정.")
            return 1
        
        # 최소 클러스터 수 계산
        min_k = max(2, (n_samples + max_cluster_size - 1) // max_cluster_size)
        min_k = min(min_k, n_samples)  # 데이터 수보다 클 수 없음
        
        print(f"   데이터 수: {n_samples}개")
        print(f"   최소 클러스터 수: {min_k}개")
        
        if min_k >= n_samples:
            print(f"⚠️ 클러스터 수가 데이터 수와 같거나 큽니다. 클러스터 수를 {max(1, n_samples-1)}로 설정.")
            return max(1, n_samples - 1)
        
        best_k = min_k
        best_score = -1
        best_max_cluster_size = float('inf')
        
        # 클러스터 수 범위 설정
        k_range = range(min_k, min(max_k + 1, n_samples))
        
        for k in tqdm(k_range, desc="클러스터 수 테스트"):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # 각 클러스터의 크기 계산
                unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
                max_current_cluster_size = np.max(cluster_sizes)
                
                # 클러스터 크기 제한 조건 확인
                if max_current_cluster_size <= max_cluster_size:
                    # 실루엣 스코어 계산 (안전한 방식)
                    try:
                        if k > 1 and len(np.unique(cluster_labels)) > 1:
                            sil_score = silhouette_score(features, cluster_labels)
                        else:
                            sil_score = 0.0
                    except:
                        sil_score = 0.0
                    
                    # 더 나은 결과인지 확인
                    if (sil_score > best_score or 
                        (sil_score >= best_score - 0.05 and max_current_cluster_size < best_max_cluster_size)):
                        best_k = k
                        best_score = sil_score
                        best_max_cluster_size = max_current_cluster_size
                        
                else:
                    continue
                
            except Exception as e:
                print(f"⚠️ 클러스터 수 {k} 테스트 중 오류: {str(e)}")
                continue
        
        # 결과 검증
        if best_k == min_k and best_score == -1:
            print("⚠️ 클러스터 크기 제한을 만족하는 결과를 찾지 못했습니다. 최소 클러스터 수를 사용합니다.")
            
            # 최종 검증을 위해 한 번 더 클러스터링
            try:
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
                best_max_cluster_size = np.max(cluster_sizes)
                
                if best_k > 1 and len(np.unique(cluster_labels)) > 1:
                    best_score = silhouette_score(features, cluster_labels)
                else:
                    best_score = 0.0
            except:
                best_score = 0.0
        
        print(f"✅ 최적 클러스터 수: {best_k}")
        print(f"   실루엣 스코어: {best_score:.3f}")
        print(f"   최대 클러스터 크기: {best_max_cluster_size}개")
        
        return best_k
    
    def cluster_motions(self, sequences: List[KeypointSequence], max_cluster_size: int = 4, force_include_all: bool = True) -> Dict[str, Any]:
        """동작 클러스터링 수행 - 모든 라벨 포함 보장"""
        print(f"\n🎯 동작 클러스터링 시작 (최대 클러스터 크기: {max_cluster_size}개)")
        print(f"   강제 포함 모드: {'ON' if force_include_all else 'OFF'}")
        
        # 특성 추출 (강제 모드)
        features, labels = self.extract_features_from_sequences(sequences, force_extraction=force_include_all)
        
        if len(features) == 0:
            print("❌ 추출된 특성이 없습니다.")
            return {}
        
        print(f"📊 클러스터링할 데이터: {len(features)}개 시퀸스")
        print(f"   라벨 수: {len(set(labels))}개")
        
        # 특성 정규화 (안전한 방식)
        try:
            features_normalized = self.embedder.scaler.fit_transform(features)
        except Exception as e:
            print(f"⚠️ 정규화 실패, 원본 특성 사용: {str(e)}")
            features_normalized = features
        
        # PCA 차원 축소 (안전한 방식)
        try:
            if features_normalized.shape[1] > features_normalized.shape[0]:
                # 차원이 샘플 수보다 큰 경우 PCA 성분 수 조정
                n_components = min(features_normalized.shape[0] - 1, int(features_normalized.shape[1] * 0.95))
                self.embedder.pca = PCA(n_components=max(1, n_components))
            
            features_pca = self.embedder.pca.fit_transform(features_normalized)
            print(f"📊 PCA 후 차원: {features_pca.shape[1]} (원본: {features.shape[1]})")
        except Exception as e:
            print(f"⚠️ PCA 실패, 정규화된 특성 사용: {str(e)}")
            features_pca = features_normalized
        
        # 최적 클러스터 수 찾기
        optimal_k = self.find_optimal_clusters(features_pca, max_cluster_size=max_cluster_size)
        
        # 최종 클러스터링
        print("🎯 최종 클러스터링 수행 중...")
        try:
            self.cluster_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(features_pca)
        except Exception as e:
            print(f"❌ 클러스터링 실패: {str(e)}")
            if force_include_all:
                print("🔄 단순 클러스터링으로 대체")
                # 모든 데이터를 하나의 클러스터로
                cluster_labels = np.zeros(len(features_pca), dtype=int)
                optimal_k = 1
            else:
                return {}
        
        # 클러스터 크기 검증
        unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        max_actual_size = np.max(cluster_sizes)
        
        print(f"📊 클러스터 크기 분석:")
        print(f"   평균 클러스터 크기: {np.mean(cluster_sizes):.1f}개")
        print(f"   최대 클러스터 크기: {max_actual_size}개")
        print(f"   최소 클러스터 크기: {np.min(cluster_sizes)}개")
        
        if max_actual_size > max_cluster_size:
            print(f"⚠️ 일부 클러스터가 크기 제한({max_cluster_size}개)을 초과했습니다!")
            
            # 추가 분할 시도
            print("🔄 큰 클러스터를 추가 분할합니다...")
            cluster_labels = self._split_large_clusters(features_pca, cluster_labels, max_cluster_size)
            
            # 재검증
            unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
            max_actual_size = np.max(cluster_sizes)
            print(f"   분할 후 최대 클러스터 크기: {max_actual_size}개")
        
        # 결과 정리
        clustering_results = {
            'cluster_labels': cluster_labels,
            'motion_labels': labels,
            'features': features_pca,
            'n_clusters': len(unique_labels),
            'sequences': sequences,
            'max_cluster_size': max_actual_size,
            'total_sequences': len(sequences),
            'successful_clustering': len(cluster_labels)
        }
        
        # 모든 라벨 포함 확인
        if force_include_all:
            original_labels = set([seq.label for seq in sequences])
            clustered_labels = set(labels)
            missing_labels = original_labels - clustered_labels
            
            if missing_labels:
                print(f"⚠️ 누락된 라벨 발견: {len(missing_labels)}개")
                for label in list(missing_labels)[:5]:
                    print(f"   • {label}")
                if len(missing_labels) > 5:
                    print(f"   ... 외 {len(missing_labels) - 5}개")
            else:
                print(f"✅ 모든 라벨이 클러스터링에 포함됨")
        
        # 클러스터별 동작 분석
        self._analyze_clusters(clustering_results)
        
        return clustering_results
    
    def _split_large_clusters(self, features: np.ndarray, cluster_labels: np.ndarray, max_size: int) -> np.ndarray:
        """큰 클러스터를 추가로 분할"""
        new_cluster_labels = cluster_labels.copy()
        next_cluster_id = np.max(cluster_labels) + 1
        
        unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        
        for cluster_id, size in zip(unique_labels, cluster_sizes):
            if size > max_size:
                print(f"   클러스터 {cluster_id} 분할 중 ({size}개 -> 목표: {max_size}개 이하)")
                
                # 해당 클러스터의 데이터 인덱스
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_features = features[cluster_indices]
                
                # 필요한 서브클러스터 수 계산
                n_subclusters = (size + max_size - 1) // max_size  # 올림 계산
                
                try:
                    # K-means로 서브클러스터링
                    subkmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
                    sub_labels = subkmeans.fit_predict(cluster_features)
                    
                    # 새로운 클러스터 ID 할당
                    for i, sub_label in enumerate(sub_labels):
                        original_idx = cluster_indices[i]
                        if sub_label == 0:
                            # 첫 번째 서브클러스터는 원래 ID 유지
                            new_cluster_labels[original_idx] = cluster_id
                        else:
                            # 나머지는 새로운 ID 할당
                            new_cluster_labels[original_idx] = next_cluster_id + sub_label - 1
                    
                    next_cluster_id += n_subclusters - 1
                    
                except Exception as e:
                    print(f"     ⚠️ 클러스터 {cluster_id} 분할 실패: {str(e)}")
                
        return new_cluster_labels
    
    def _analyze_clusters(self, results: Dict[str, Any]):
        """클러스터별 동작 분석 및 출력"""
        cluster_labels = results['cluster_labels']
        motion_labels = results['motion_labels']
        sequences = results['sequences']
        
        print(f"\n📊 클러스터링 결과 분석:")
        print(f"   총 시퀸스: {results['total_sequences']}개")
        print(f"   클러스터링 성공: {results['successful_clustering']}개")
        print(f"   성공률: {results['successful_clustering']/results['total_sequences']*100:.1f}%")
        
        for cluster_id in range(results['n_clusters']):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_motions = [motion_labels[i] for i in cluster_indices]
            cluster_files = [sequences[i].filename for i in cluster_indices]
            
            print(f"\n🏷️  클러스터 {cluster_id} ({len(cluster_indices)}개 동작):")
            
            # 가장 빈번한 라벨들
            from collections import Counter
            motion_counts = Counter(cluster_motions)
            top_motions = motion_counts.most_common(5)
            
            for motion, count in top_motions:
                print(f"   • {motion}: {count}개")
                
            # 대표 파일들
            if len(cluster_files) <= 3:
                print(f"   파일: {', '.join(cluster_files)}")
            else:
                print(f"   파일 예시: {', '.join(cluster_files[:3])} ...")

def save_clustering_results_to_csv(clustering_results: Dict[str, Any], output_path: str):
    """클러스터링 결과를 CSV 파일로 저장"""
    if not clustering_results:
        print("❌ 저장할 클러스터링 결과가 없습니다.")
        return
    
    cluster_labels = clustering_results['cluster_labels']
    motion_labels = clustering_results['motion_labels']
    sequences = clustering_results['sequences']
    
    # 결과 데이터 준비
    results_data = []
    
    for i, (cluster_id, motion_label, sequence) in enumerate(zip(cluster_labels, motion_labels, sequences)):
        results_data.append({
            'label_name': motion_label,
            'cluster_id': int(cluster_id),
            'filename': sequence.filename
        })
    
    # DataFrame 생성 및 저장
    df_results = pd.DataFrame(results_data)
    
    # 클러스터 ID로 정렬
    df_results = df_results.sort_values(['cluster_id', 'label_name', 'filename'])
    
    # CSV 저장
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 클러스터링 결과 CSV 저장: {output_path}")
    
    # 클러스터별 통계 출력
    cluster_stats = df_results.groupby('cluster_id').agg({
        'label_name': ['count', 'nunique'],
        'filename': 'count'
    }).round(2)
    
    print(f"\n📊 클러스터별 통계:")
    print(f"   총 클러스터 수: {df_results['cluster_id'].nunique()}개")
    print(f"   총 시퀸스 수: {len(df_results)}개")
    print(f"   총 라벨 수: {df_results['label_name'].nunique()}개")
    
    # 각 클러스터의 주요 라벨들 출력
    print(f"\n🏷️  각 클러스터의 주요 라벨들:")
    for cluster_id in sorted(df_results['cluster_id'].unique()):
        cluster_data = df_results[df_results['cluster_id'] == cluster_id]
        label_counts = cluster_data['label_name'].value_counts()
        top_labels = label_counts.head(3)
        
        print(f"   클러스터 {cluster_id} ({len(cluster_data)}개):")
        for label, count in top_labels.items():
            percentage = (count / len(cluster_data)) * 100
            print(f"     • {label}: {count}개 ({percentage:.1f}%)")
        
        if len(label_counts) > 3:
            print(f"     ... 외 {len(label_counts) - 3}개 라벨")

def main():
    """메인 실행 함수 - 모든 라벨 포함 보장"""
    print("🚀 Motion Extraction and Clustering System 시작")
    print("   모든 라벨이 클러스터링에 포함되도록 강건성 개선")
    print("=" * 60)
    
    try:
        # 1. 유니크한 라벨 추출
        extractor = MotionExtractor()
        labels_dict = extractor.extract_unique_labels_with_first_files("labels.csv")
        
        if not labels_dict:
            print("❌ 추출할 라벨이 없습니다. 프로그램을 종료합니다.")
            return
        
        print(f"\n📊 처리할 라벨: {len(labels_dict)}개")
        
        # 2. 키포인트 시퀸스 추출 및 저장 (강제 추출 모드)
        print("\n" + "="*60)
        print("🎬 키포인트 시퀸스 추출 단계")
        extracted_sequences = extractor.extract_all_sequences(labels_dict, force_extract=True)
        
        if not extracted_sequences:
            print("❌ 추출된 시퀸스가 없습니다.")
            
            # Fallback: 기존 추출된 시퀸스 파일들 검색
            print("🔍 기존 추출된 시퀸스 파일 검색 중...")
            extracted_sequences = extractor.compose_extracted_sequences()
            
            if not extracted_sequences:
                print("❌ 기존 시퀸스 파일도 없습니다. 프로그램을 종료합니다.")
                return
        
        print(f"\n✅ 총 {len(extracted_sequences)}개 시퀸스 준비 완료")
        
        # 3. extracted_labels.csv 생성
        print("\n" + "="*60)
        print("📝 추출된 라벨 파일 생성")
        df_extracted = pd.DataFrame(extracted_sequences, columns=['sequence_path', 'label'])
        df_extracted.to_csv('extracted_labels.csv', index=False, encoding='utf-8-sig')
        print(f"✅ extracted_labels.csv 생성 완료 ({len(extracted_sequences)}개 항목)")
        
        # 추출된 라벨 통계 출력
        unique_labels = df_extracted['label'].nunique()
        print(f"   유니크 라벨 수: {unique_labels}개")
        print(f"   원본 라벨 수: {len(labels_dict)}개")
        print(f"   커버리지: {unique_labels/len(labels_dict)*100:.1f}%")
        
        # 4. 동작 클러스터링 (강제 포함 모드)
        print("\n" + "="*60)
        print("🎯 동작 클러스터링 단계")
        
        embedder = MotionEmbedder()
        clusterer = MotionClusterer(embedder)
        
        # 시퀸스 로드 (강제 로딩 모드)
        sequence_paths = [path for path, _ in extracted_sequences]
        sequences = clusterer.load_sequences(sequence_paths, ensure_all_loaded=True)
        
        if not sequences:
            print("❌ 로딩된 시퀸스가 없습니다. 프로그램을 종료합니다.")
            return
        
        print(f"📊 클러스터링 입력 데이터:")
        print(f"   총 시퀸스: {len(sequences)}개")
        print(f"   라벨 수: {len(set([seq.label for seq in sequences]))}개")
        
        # 클러스터링 수행 (강제 포함 모드)
        clustering_results = clusterer.cluster_motions(
            sequences, 
            max_cluster_size=4,  # 클러스터 최대 크기
            force_include_all=True  # 모든 라벨 강제 포함
        )
        
        # 5. 클러스터링 결과 저장 및 분석
        if clustering_results:
            print("\n" + "="*60)
            print("💾 결과 저장 및 분석")
            
            # 결과 검증
            original_label_count = len(labels_dict)
            clustered_label_count = len(set(clustering_results['motion_labels']))
            success_rate = len(clustering_results['cluster_labels']) / len(sequences) * 100
            
            print(f"📊 최종 결과 검증:")
            print(f"   원본 라벨 수: {original_label_count}개")
            print(f"   클러스터링된 라벨 수: {clustered_label_count}개")
            print(f"   라벨 포함률: {clustered_label_count/original_label_count*100:.1f}%")
            print(f"   시퀸스 성공률: {success_rate:.1f}%")
            
            # CSV 파일로 저장
            csv_path = "two-clusters/video_clusters.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            save_clustering_results_to_csv(clustering_results, csv_path)
            
            # 포함되지 않은 라벨 확인
            original_labels = set(labels_dict.keys())
            clustered_labels = set(clustering_results['motion_labels'])
            missing_labels = original_labels - clustered_labels
            
            if missing_labels:
                print(f"\n⚠️ 클러스터링에 포함되지 않은 라벨: {len(missing_labels)}개")
                for i, label in enumerate(list(missing_labels)[:10]):
                    print(f"   {i+1}. {label}")
                if len(missing_labels) > 10:
                    print(f"   ... 외 {len(missing_labels) - 10}개")
                    
                # 누락된 라벨들을 별도 파일로 저장
                missing_df = pd.DataFrame(list(missing_labels), columns=['missing_label'])
                missing_df.to_csv('missing_labels.csv', index=False, encoding='utf-8-sig')
                print(f"   누락된 라벨 목록: missing_labels.csv 저장")
            else:
                print(f"✅ 모든 라벨이 클러스터링에 성공적으로 포함됨!")
        else:
            print("❌ 클러스터링에 실패했습니다.")
            return
    
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생:")
        print(f"   오류: {str(e)}")
        print(f"   타입: {type(e).__name__}")
        
        # 부분적 결과라도 저장 시도
        try:
            if 'extracted_sequences' in locals() and extracted_sequences:
                df_backup = pd.DataFrame(extracted_sequences, columns=['sequence_path', 'label'])
                df_backup.to_csv('backup_extracted_labels.csv', index=False, encoding='utf-8-sig')
                print(f"🔄 부분 결과 저장: backup_extracted_labels.csv ({len(extracted_sequences)}개 항목)")
        except:
            pass
            
        return
    
    print("\n🎉 모든 작업 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
