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
from saturday_dinner.utils.video_path_utils import get_video_root_and_path
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
        """라벨데이터에서 유니크한 라벨과 첫번째 파일명 쌍 추출"""
        print("📋 라벨 데이터에서 유니크한 라벨 추출 중...")
        
        df = pd.read_csv(labels_csv_path)
        unique_labels = {}
        
        for _, row in df.iterrows():
            filename = row['파일명']
            label = row['한국어']
            
            if label not in unique_labels:
                unique_labels[label] = filename
                
        print(f"✅ {len(unique_labels)}개의 유니크한 라벨 발견:")
        for i, (label, filename) in enumerate(list(unique_labels.items())[:10]):
            print(f"   {i+1}. {filename} -> {label}")
        if len(unique_labels) > 10:
            print(f"   ... 외 {len(unique_labels)-10}개")
            
        return unique_labels
    
    def extract_keypoints_from_video(self, video_path: str) -> Optional[KeypointSequence]:
        """비디오에서 키포인트 시퀸스 추출"""
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
        max_errors = 10  # 최대 허용 오류 수
        
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
                    
                except Exception as e:
                    error_count += 1
                    # 너무 많은 오류가 발생하면 비디오 처리를 중단
                    if error_count > max_errors:
                        print(f"❌ 프레임 처리 오류가 너무 많습니다: {video_path}")
                        print(f"   오류 수: {error_count}/{frame_idx + 1}")
                        cap.release()
                        return None
                    
                    # 오류 프레임은 영점 키포인트로 대체
                    pose_landmarks_list.append(np.zeros((33, 3)))
                    left_hand_landmarks_list.append(np.zeros((21, 3)))
                    right_hand_landmarks_list.append(np.zeros((21, 3)))
                    face_landmarks_list.append(np.zeros((468, 3)))
                
                frame_idx += 1
                pbar.update(1)
                
        cap.release()
        
        # 처리 통계 출력
        if error_count > 0:
            print(f"⚠️ 프레임 처리 중 {error_count}개 오류 발생 (총 {frame_idx}개 프레임)")
        
        if not pose_landmarks_list:
            print(f"❌ 키포인트를 추출할 수 없습니다: {video_path}")
            return None
            
        # 안전한 numpy 배열 변환 (형상 불일치 예외 처리)
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
    
    def extract_all_sequences(self, labels_dict: Dict[str, str]) -> List[Tuple[str, str]]:
        """모든 라벨에 대해 키포인트 시퀸스 추출"""
        print(f"\n🎬 {len(labels_dict)}개 영상에서 키포인트 시퀸스 추출 시작...")
        
        extracted_sequences = []
        failed_extractions = []
        skipped_sequences = []
        
        for label, filename in tqdm(labels_dict.items(), desc="영상 처리"):
            print(f"\n처리 중: {filename} -> {label}")
            
            # 비디오 경로 찾기 (중복 체크를 위해 먼저 실행)
            video_path = get_video_root_and_path(filename, verbose=False)
            
            if video_path is None:
                print(f"❌ 비디오 파일을 찾을 수 없습니다: {filename}")
                failed_extractions.append((filename, label, "파일 없음"))
                continue
            
            # 🔍 중복 처리 방지: 실제 비디오 파일명으로 중복 체크
            actual_filename = os.path.basename(video_path)  # 실제 파일명 (확장자 포함)
            if self.sequence_exists(actual_filename, label):
                existing_path = self.generate_sequence_filename(actual_filename, label)
                extracted_sequences.append((existing_path, label))
                skipped_sequences.append((filename, label))
                print(f"⏭️ 이미 존재함: {existing_path}")
                continue
                
            # 키포인트 추출
            sequence = self.extract_keypoints_from_video(video_path)
            
            if sequence is None:
                print(f"❌ 키포인트 추출 실패: {filename}")
                failed_extractions.append((filename, label, "추출 실패"))
                continue
                
            # 시퀸스 저장 (실제 파일명 사용)
            try:
                sequence_path = self.save_sequence(sequence, label)
                extracted_sequences.append((sequence_path, label))
                print(f"✅ 저장 완료: {sequence_path}")
            except Exception as e:
                print(f"❌ 저장 실패: {filename} - {str(e)}")
                failed_extractions.append((filename, label, f"저장 실패: {str(e)}"))
                
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
            for filename, label, reason in failed_extractions:
                print(f"   • {filename} ({label}): {reason}")
                
        return extracted_sequences

class MotionEmbedder:
    """동작의 동적 특성을 추출하는 임베딩 시스템"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 95% 분산 유지
        
    def extract_dynamic_features(self, sequence: KeypointSequence) -> np.ndarray:
        """동적 특성을 추출하여 임베딩 벡터 생성"""
        
        try:
            # 1. 속도 (Velocity) 계산
            velocity = self._calculate_velocity(sequence.sequence)
            
            # 2. 가속도 (Acceleration) 계산  
            acceleration = self._calculate_acceleration(sequence.sequence)
            
            # 3. 각속도 (Angular Velocity) 계산
            angular_velocity = self._calculate_angular_velocity(sequence.sequence)
            
            # 4. 움직임 궤적의 특성
            trajectory_features = self._extract_trajectory_features(sequence.sequence)
            
            # 5. 주파수 도메인 특성
            frequency_features = self._extract_frequency_features(sequence.sequence)
            
            # 6. 신체 부위별 상대적 움직임
            relative_motion = self._calculate_relative_motion(sequence)
            
            # 모든 특성을 안전하게 결합
            feature_list = []
            
            # 각 특성을 1차원으로 평탄화하여 추가 (고정된 크기로)
            try:
                velocity_flat = velocity.flatten()
                # 너무 큰 특성은 크기 제한
                if len(velocity_flat) > 10000:
                    velocity_flat = velocity_flat[:10000]
                feature_list.append(velocity_flat)
            except:
                feature_list.append(np.zeros(100))  # 기본 크기
            
            try:
                acceleration_flat = acceleration.flatten()
                if len(acceleration_flat) > 10000:
                    acceleration_flat = acceleration_flat[:10000]
                feature_list.append(acceleration_flat)
            except:
                feature_list.append(np.zeros(100))
            
            try:
                angular_flat = angular_velocity.flatten()
                if len(angular_flat) > 5000:
                    angular_flat = angular_flat[:5000]
                feature_list.append(angular_flat)
            except:
                feature_list.append(np.zeros(50))
            
            try:
                trajectory_flat = trajectory_features.flatten()
                if len(trajectory_flat) > 5000:
                    trajectory_flat = trajectory_flat[:5000]
                feature_list.append(trajectory_flat)
            except:
                feature_list.append(np.zeros(100))
            
            try:
                frequency_flat = frequency_features.flatten()
                if len(frequency_flat) > 1000:
                    frequency_flat = frequency_flat[:1000]
                feature_list.append(frequency_flat)
            except:
                feature_list.append(np.zeros(40))
            
            try:
                relative_flat = relative_motion.flatten()
                feature_list.append(relative_flat)
            except:
                feature_list.append(np.zeros(8))
            
            # 모든 특성 결합
            all_features = np.concatenate(feature_list)
            
            # NaN이나 무한대 값 처리
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return all_features
            
        except Exception as e:
            print(f"⚠️ 특성 추출 중 오류 발생: {sequence.filename} - {str(e)}")
            # 기본 특성 벡터 반환
            return np.zeros(1000)  # 기본 크기의 영벡터
    
    def _calculate_velocity(self, sequence: np.ndarray) -> np.ndarray:
        """프레임 간 속도 계산"""
        if len(sequence) < 2:
            return np.zeros_like(sequence)
            
        velocity = np.diff(sequence, axis=0)
        # 첫 프레임은 0으로 패딩
        velocity = np.vstack([np.zeros((1,) + velocity.shape[1:]), velocity])
        
        return velocity
    
    def _calculate_acceleration(self, sequence: np.ndarray) -> np.ndarray:
        """가속도 계산 (속도의 변화율)"""
        velocity = self._calculate_velocity(sequence)
        
        if len(velocity) < 2:
            return np.zeros_like(velocity)
            
        acceleration = np.diff(velocity, axis=0)
        acceleration = np.vstack([np.zeros((1,) + acceleration.shape[1:]), acceleration])
        
        return acceleration
    
    def _calculate_angular_velocity(self, sequence: np.ndarray) -> np.ndarray:
        """각속도 계산 (관절 각도의 변화율)"""
        if len(sequence) < 2:
            return np.zeros((len(sequence), min(50, sequence.shape[1] - 1)))
            
        # 인접한 랜드마크 쌍들의 각도 변화 계산 (고정된 수의 특성)
        n_landmarks = sequence.shape[1]
        max_pairs = min(50, n_landmarks - 1)  # 최대 50쌍의 각도 변화 계산
        
        angular_changes = []
        
        for i in range(len(sequence) - 1):
            frame_angular = []
            
            # 인접한 랜드마크 쌍들의 벡터 변화 계산
            for j in range(max_pairs):
                if j + 1 < n_landmarks:
                    # 3D 좌표에서 벡터 계산
                    v1_current = sequence[i, j, :]      # 현재 프레임의 j번째 랜드마크
                    v2_current = sequence[i, j + 1, :]  # 현재 프레임의 j+1번째 랜드마크
                    v1_next = sequence[i + 1, j, :]     # 다음 프레임의 j번째 랜드마크  
                    v2_next = sequence[i + 1, j + 1, :] # 다음 프레임의 j+1번째 랜드마크
                    
                    # 안전한 각도 변화 계산
                    try:
                        vec_current = v2_current - v1_current
                        vec_next = v2_next - v1_next
                        
                        # 벡터의 크기가 너무 작으면 0으로 처리
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
            
        # 첫 프레임은 0으로 패딩
        if angular_changes:
            first_frame = [0.0] * len(angular_changes[0])
            angular_changes.insert(0, first_frame)
        else:
            # 빈 경우 기본값 반환
            return np.zeros((len(sequence), max_pairs))
            
        return np.array(angular_changes)
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """두 벡터 간의 각도 계산"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def _extract_trajectory_features(self, sequence: np.ndarray) -> np.ndarray:
        """움직임 궤적의 통계적 특성 추출"""
        features = []
        
        # 각 랜드마크별로 궤적 특성 계산
        for landmark_idx in range(sequence.shape[1]):
            landmark_traj = sequence[:, landmark_idx]
            
            # 기본 통계량
            features.extend([
                np.mean(landmark_traj),
                np.std(landmark_traj),
                np.min(landmark_traj),
                np.max(landmark_traj),
                np.ptp(landmark_traj)  # peak-to-peak
            ])
            
        return np.array(features)
    
    def _extract_frequency_features(self, sequence: np.ndarray) -> np.ndarray:
        """주파수 도메인 특성 추출 (FFT 기반)"""
        features = []
        
        # 각 랜드마크별로 주파수 분석
        for landmark_idx in range(min(10, sequence.shape[1])):  # 처음 10개만 분석 (계산 효율성)
            landmark_traj = sequence[:, landmark_idx]
            
            # FFT 계산
            fft_result = np.fft.fft(landmark_traj)
            power_spectrum = np.abs(fft_result)
            
            # 주요 주파수 성분
            features.extend([
                np.mean(power_spectrum),
                np.std(power_spectrum),
                np.argmax(power_spectrum),  # 주요 주파수
                np.sum(power_spectrum[:len(power_spectrum)//4])  # 저주파 에너지
            ])
            
        return np.array(features)
    
    def _calculate_relative_motion(self, sequence: KeypointSequence) -> np.ndarray:
        """신체 부위별 상대적 움직임 계산"""
        features = []
        
        # 각 신체 부위의 움직임 정도 계산
        pose_motion = 0.0
        left_hand_motion = 0.0
        right_hand_motion = 0.0
        face_motion = 0.0
        
        if sequence.pose_landmarks.shape[1] > 0:
            pose_motion = np.mean(np.std(sequence.pose_landmarks, axis=0))
            
        if sequence.left_hand_landmarks.shape[1] > 0:
            left_hand_motion = np.mean(np.std(sequence.left_hand_landmarks, axis=0))
            
        if sequence.right_hand_landmarks.shape[1] > 0:
            right_hand_motion = np.mean(np.std(sequence.right_hand_landmarks, axis=0))
            
        if sequence.face_landmarks.shape[1] > 0:
            face_motion = np.mean(np.std(sequence.face_landmarks, axis=0))
        
        # 손 대비 몸통의 움직임 비율
        features.extend([
            left_hand_motion / (pose_motion + 1e-8),
            right_hand_motion / (pose_motion + 1e-8),
            (left_hand_motion + right_hand_motion) / (pose_motion + 1e-8)
        ])
            
        # 얼굴과 몸통의 상대적 움직임
        features.append(face_motion / (pose_motion + 1e-8))
        
        # 추가적인 비율 특성들
        features.extend([
            left_hand_motion / (right_hand_motion + 1e-8),  # 왼손 vs 오른손
            face_motion / (left_hand_motion + right_hand_motion + 1e-8),  # 얼굴 vs 손
            pose_motion,  # 절대적 몸통 움직임
            left_hand_motion + right_hand_motion,  # 총 손 움직임
        ])
            
        return np.array(features)

class MotionClusterer:
    """동작 클러스터링 시스템"""
    
    def __init__(self, embedder: MotionEmbedder):
        self.embedder = embedder
        self.features = None
        self.labels = None
        self.cluster_model = None
        
    def load_sequences(self, sequence_paths: List[str]) -> List[KeypointSequence]:
        """저장된 시퀸스들을 로드"""
        sequences = []
        
        print(f"📂 {len(sequence_paths)}개 시퀸스 로딩 중...")
        
        for path in tqdm(sequence_paths, desc="시퀸스 로딩"):
            try:
                with open(path, 'rb') as f:
                    sequence = pickle.load(f)
                    sequences.append(sequence)
            except Exception as e:
                print(f"❌ 로딩 실패: {path} - {str(e)}")
                
        print(f"✅ {len(sequences)}개 시퀸스 로딩 완료")
        return sequences
    
    def extract_features_from_sequences(self, sequences: List[KeypointSequence]) -> Tuple[np.ndarray, List[str]]:
        """시퀸스들에서 특성 추출"""
        print("🔍 동적 특성 추출 중...")
        
        all_features = []
        labels = []
        failed_extractions = []
        
        for i, sequence in enumerate(tqdm(sequences, desc="특성 추출")):
            try:
                features = self.embedder.extract_dynamic_features(sequence)
                
                # 특성이 유효한지 확인
                if features is not None and len(features) > 0:
                    # NaN이나 무한대 값 처리
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    all_features.append(features)
                    labels.append(sequence.label)
                else:
                    failed_extractions.append((sequence.filename, "빈 특성 벡터"))
                
            except Exception as e:
                failed_extractions.append((sequence.filename, f"특성 추출 실패: {str(e)}"))
                
        if not all_features:
            print("❌ 추출된 특성이 없습니다.")
            return np.array([]), []
        
        # 특성 벡터 크기 확인 및 정규화
        feature_sizes = [len(f) for f in all_features]
        min_size = min(feature_sizes)
        max_size = max(feature_sizes)
        
        print(f"📊 특성 벡터 크기 분석:")
        print(f"   최소 크기: {min_size}")
        print(f"   최대 크기: {max_size}")
        print(f"   평균 크기: {np.mean(feature_sizes):.1f}")
        
        # 크기가 다른 경우 처리
        if min_size != max_size:
            print("⚠️ 특성 벡터 크기가 불일치합니다. 크기를 맞춰줍니다...")
            
            # 가장 일반적인 크기를 기준으로 설정
            from collections import Counter
            size_counts = Counter(feature_sizes)
            target_size = size_counts.most_common(1)[0][0]
            
            print(f"   기준 크기: {target_size} (가장 빈번한 크기)")
            
            normalized_features = []
            for i, features in enumerate(all_features):
                if len(features) > target_size:
                    # 크기가 큰 경우 자르기
                    normalized_features.append(features[:target_size])
                elif len(features) < target_size:
                    # 크기가 작은 경우 0으로 패딩
                    padded = np.zeros(target_size)
                    padded[:len(features)] = features
                    normalized_features.append(padded)
                else:
                    normalized_features.append(features)
                
            all_features = normalized_features
        
        try:
            # numpy 배열로 변환
            features_array = np.array(all_features)
            
            # 최종 검증
            if features_array.ndim != 2:
                print(f"❌ 예상치 못한 특성 배열 형태: {features_array.shape}")
                return np.array([]), []
            
            print(f"✅ 특성 추출 완료: {features_array.shape}")
            
            if failed_extractions:
                print(f"⚠️ 실패한 추출: {len(failed_extractions)}개")
                for filename, reason in failed_extractions[:5]:  # 처음 5개만 표시
                    print(f"   • {filename}: {reason}")
                if len(failed_extractions) > 5:
                    print(f"   ... 외 {len(failed_extractions) - 5}개")
            
            return features_array, labels
            
        except Exception as e:
            print(f"❌ 특성 배열 변환 실패: {str(e)}")
            print(f"   특성 개수: {len(all_features)}")
            if all_features:
                print(f"   첫 번째 특성 크기: {len(all_features[0])}")
                print(f"   마지막 특성 크기: {len(all_features[-1])}")
            
            return np.array([]), []
    
    def find_optimal_clusters(self, features: np.ndarray, max_cluster_size: int = 20, max_k: int = 50) -> int:
        """최적의 클러스터 수 찾기 (클러스터 크기 제한 포함)"""
        print(f"🎯 최적 클러스터 수 탐색 중 (최대 클러스터 크기: {max_cluster_size}개)...")
        
        n_samples = len(features)
        
        # 최소 클러스터 수 계산 (모든 클러스터가 max_cluster_size 이하가 되도록)
        min_k = max(2, (n_samples + max_cluster_size - 1) // max_cluster_size)  # 올림 계산
        
        print(f"   데이터 수: {n_samples}개")
        print(f"   최소 클러스터 수: {min_k}개")
        
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
                    # 실루엣 스코어 계산
                    sil_score = silhouette_score(features, cluster_labels)
                    
                    # 더 나은 결과인지 확인 (실루엣 스코어 우선, 클러스터 균형 고려)
                    if (sil_score > best_score or 
                        (sil_score >= best_score - 0.05 and max_current_cluster_size < best_max_cluster_size)):
                        best_k = k
                        best_score = sil_score
                        best_max_cluster_size = max_current_cluster_size
                        
                else:
                    # 클러스터 크기가 제한을 넘는 경우 계속 탐색
                    continue
                
            except Exception as e:
                print(f"⚠️ 클러스터 수 {k} 테스트 중 오류: {str(e)}")
                continue
        
        # 결과 검증
        if best_k == min_k and best_score == -1:
            print("⚠️ 클러스터 크기 제한을 만족하는 결과를 찾지 못했습니다. 최소 클러스터 수를 사용합니다.")
            best_k = min_k
            
            # 최종 검증을 위해 한 번 더 클러스터링
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
            best_max_cluster_size = np.max(cluster_sizes)
            best_score = silhouette_score(features, cluster_labels)
        
        print(f"✅ 최적 클러스터 수: {best_k}")
        print(f"   실루엣 스코어: {best_score:.3f}")
        print(f"   최대 클러스터 크기: {best_max_cluster_size}개")
        
        return best_k
    
    def cluster_motions(self, sequences: List[KeypointSequence], max_cluster_size: int = 4) -> Dict[str, Any]:
        """동작 클러스터링 수행 (클러스터 크기 제한 포함)"""
        print(f"\n🎯 동작 클러스터링 시작 (최대 클러스터 크기: {max_cluster_size}개)...")
        
        # 특성 추출
        features, labels = self.extract_features_from_sequences(sequences)
        
        if len(features) == 0:
            print("❌ 추출된 특성이 없습니다.")
            return {}
        
        # 특성 정규화
        features_normalized = self.embedder.scaler.fit_transform(features)
        
        # PCA 차원 축소
        features_pca = self.embedder.pca.fit_transform(features_normalized)
        
        print(f"📊 PCA 후 차원: {features_pca.shape[1]} (원본: {features.shape[1]})")
        
        # 최적 클러스터 수 찾기 (크기 제한 포함)
        optimal_k = self.find_optimal_clusters(features_pca, max_cluster_size=max_cluster_size)
        
        # 최종 클러스터링
        print("🎯 최종 클러스터링 수행 중...")
        self.cluster_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(features_pca)
        
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
            'max_cluster_size': max_actual_size
        }
        
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
    """메인 실행 함수"""
    print("🚀 Motion Extraction and Clustering System 시작")
    print("=" * 60)
    
    # 1. 유니크한 라벨 추출
    extractor = MotionExtractor()
    labels_dict = extractor.extract_unique_labels_with_first_files("labels.csv")
    
    # 2. 키포인트 시퀸스 추출 및 저장
    extracted_sequences = extractor.extract_all_sequences(labels_dict)
    
    if not extracted_sequences:
        print("❌ 추출된 시퀸스가 없습니다. 프로그램을 종료합니다.")
        return
    
    # compose extracted sequences from current files in the output directory
    # extracted_sequences = extractor.compose_extracted_sequences()
    
    if not extracted_sequences:
        print("❌ 추출된 시퀸스가 없습니다. 프로그램을 종료합니다.")
        return
    
    # 3. extracted_labels.csv 생성
    df_extracted = pd.DataFrame(extracted_sequences, columns=['sequence_path', 'label'])
    df_extracted.to_csv('extracted_labels.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ extracted_labels.csv 생성 완료 ({len(extracted_sequences)}개 항목)")
    
    # 4. 동작 클러스터링
    embedder = MotionEmbedder()
    clusterer = MotionClusterer(embedder)
    
    # 시퀸스 로드
    sequence_paths = [path for path, _ in extracted_sequences]
    sequences = clusterer.load_sequences(sequence_paths)
    
    if sequences:
        # 클러스터링 수행
        clustering_results = clusterer.cluster_motions(sequences)
        
        # 클러스터링 결과 저장
        if clustering_results:
            
            # CSV 파일로 저장
            csv_path = f"two-clusters/video_clusters.csv"
            save_clustering_results_to_csv(clustering_results, csv_path)
    
    print("\n🎉 모든 작업 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
