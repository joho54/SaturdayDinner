import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    BatchNormalization,
    Lambda,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from scipy.interpolate import interp1d
import sys
import json
import pandas as pd
import pickle
from datetime import datetime
import logging
from collections import defaultdict
from config import LABEL_MAX_SAMPLES_PER_CLASS, MIN_SAMPLES_PER_CLASS

# .env 파일 로드 (s3_utils보다 먼저 로드)
try:
    from dotenv import load_dotenv
    # .env 파일이 있는 현재 디렉토리에서 로드
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드: {env_path}")
    else:
        # 현재 작업 디렉토리에서 .env 파일 찾기
        if os.path.exists('.env'):
            load_dotenv('.env')
            print("✅ .env 파일 로드: ./.env")
        else:
            print("⚠️ .env 파일을 찾을 수 없습니다.")
except ImportError:
    print("⚠️ python-dotenv를 설치하세요: pip install python-dotenv")

# S3 호환 캐시 시스템 import
from s3_utils import (
    cache_join,
    cache_exists,
    cache_makedirs,
    cache_save_pickle,
    cache_load_pickle,
    cache_remove,
    is_s3_path
)

# MediaPipe 및 TensorFlow 로깅 완전 억제
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ERROR만 출력
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU 비활성화 (CPU만 사용)
logging.getLogger("mediapipe").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# 설정 파일에서 파라미터 import
from config import *

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic


class MediaPipeManager:
    """MediaPipe 객체를 안전하게 관리하는 컨텍스트 매니저"""

    _instance = None
    _holistic = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MediaPipeManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._holistic is None:
            self._holistic = mp_holistic.Holistic(
                static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
                model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
                smooth_landmarks=MEDIAPIPE_SMOOTH_LANDMARKS,
                enable_segmentation=MEDIAPIPE_ENABLE_SEGMENTATION,
                smooth_segmentation=MEDIAPIPE_SMOOTH_SEGMENTATION,
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            )

    def __enter__(self):
        return self._holistic

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 전역 객체는 유지하고 정리만
        pass

    @classmethod
    def cleanup(cls):
        """전역 MediaPipe 객체 정리"""
        if cls._holistic:
            cls._holistic.close()
            cls._holistic = None


# 디렉토리 생성
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(INFO_DIR, exist_ok=True)

# 고유한 모델 이름 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"{MODELS_DIR}/sign_language_model_{timestamp}.keras"
MODEL_INFO_PATH = f"{INFO_DIR}/model-info-{timestamp}.json"

# 캐시 디렉토리 설정
cache_makedirs(CACHE_DIR, exist_ok=True)

DATA_CACHE_PATH = "fixed_preprocessed_data.npz"


# 라벨별 캐시 파일 경로 생성 함수
def get_label_cache_path(label):
    """라벨별 캐시 파일 경로를 반환합니다. 주요 파라미터를 파일명에 포함시켜 캐시 무효화가 자동으로 되도록 합니다."""
    safe_label = label.replace(" ", "_").replace("/", "_")

    # 데이터 개수 관련 파라미터들을 파일명에 포함
    max_samples_str = (
        f"max{LABEL_MAX_SAMPLES_PER_CLASS}"
        if LABEL_MAX_SAMPLES_PER_CLASS
        else "maxNone"
    )
    min_samples_str = f"min{MIN_SAMPLES_PER_CLASS}"

    return cache_join(
        CACHE_DIR,
        f"{safe_label}_seq{TARGET_SEQ_LENGTH}_aug{AUGMENTATIONS_PER_VIDEO}_{max_samples_str}_{min_samples_str}.pkl",
    )


def save_label_cache(label, data):
    """라벨별 데이터를 캐시에 저장합니다."""
    cache_path = get_label_cache_path(label)

    # 캐시에 저장할 데이터와 파라미터 정보
    cache_data = {
        "data": data,
        "parameters": {
            "TARGET_SEQ_LENGTH": TARGET_SEQ_LENGTH,
            "AUGMENTATIONS_PER_VIDEO": AUGMENTATIONS_PER_VIDEO,
            "AUGMENTATION_NOISE_LEVEL": AUGMENTATION_NOISE_LEVEL,
            "AUGMENTATION_SCALE_RANGE": AUGMENTATION_SCALE_RANGE,
            "AUGMENTATION_ROTATION_RANGE": AUGMENTATION_ROTATION_RANGE,
            "NONE_CLASS_NOISE_LEVEL": NONE_CLASS_NOISE_LEVEL,
            "NONE_CLASS_AUGMENTATIONS_PER_FRAME": NONE_CLASS_AUGMENTATIONS_PER_FRAME,
            # 데이터 개수 관련 파라미터 추가
            "LABEL_MAX_SAMPLES_PER_CLASS": LABEL_MAX_SAMPLES_PER_CLASS,
            "MIN_SAMPLES_PER_CLASS": MIN_SAMPLES_PER_CLASS,
        },
    }

    # S3 호환 캐시 저장
    try:
        if is_s3_path(cache_path):
            # S3에서는 put_object가 원자적이므로 직접 저장
            success = cache_save_pickle(cache_path, cache_data)
            if success:
                print(f"💾 {label} 라벨 데이터 캐시 저장 (S3): {cache_path} ({len(data)}개 샘플)")
            else:
                raise Exception("S3 캐시 저장 실패")
        else:
            # 로컬에서는 임시 파일 방식 사용 (원자적 쓰기)
            temp_path = cache_path + ".tmp"
            
            with open(temp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 성공적으로 저장되면 최종 위치로 이동
            os.replace(temp_path, cache_path)
            print(f"💾 {label} 라벨 데이터 캐시 저장 (로컬): {cache_path} ({len(data)}개 샘플)")

    except Exception as e:
        # 오류 발생 시 임시 파일 정리 (로컬 파일인 경우만)
        if not is_s3_path(cache_path):
            temp_path = cache_path + ".tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise e


def load_label_cache(label):
    """라벨별 데이터를 캐시에서 로드합니다."""
    cache_path = get_label_cache_path(label)
    if cache_exists(cache_path):
        try:
            # S3 호환 캐시 로드
            cache_data = cache_load_pickle(cache_path)
            if cache_data is None:
                return None

            # 캐시 형식 확인 (구버전 호환성)
            if (
                isinstance(cache_data, dict)
                and "data" in cache_data
                and "parameters" in cache_data
            ):
                # 새 형식: 파라미터 검증
                cached_params = cache_data["parameters"]
                current_params = {
                    "TARGET_SEQ_LENGTH": TARGET_SEQ_LENGTH,
                    "AUGMENTATIONS_PER_VIDEO": AUGMENTATIONS_PER_VIDEO,
                    "AUGMENTATION_NOISE_LEVEL": AUGMENTATION_NOISE_LEVEL,
                    "AUGMENTATION_SCALE_RANGE": AUGMENTATION_SCALE_RANGE,
                    "AUGMENTATION_ROTATION_RANGE": AUGMENTATION_ROTATION_RANGE,
                    "NONE_CLASS_NOISE_LEVEL": NONE_CLASS_NOISE_LEVEL,
                    "NONE_CLASS_AUGMENTATIONS_PER_FRAME": NONE_CLASS_AUGMENTATIONS_PER_FRAME,
                    # 데이터 개수 관련 파라미터 추가
                    "LABEL_MAX_SAMPLES_PER_CLASS": LABEL_MAX_SAMPLES_PER_CLASS,
                    "MIN_SAMPLES_PER_CLASS": MIN_SAMPLES_PER_CLASS,
                }

                # 파라미터 비교
                if cached_params != current_params:
                    print(f"⚠️ {label} 캐시 파라미터가 다릅니다. 캐시 무효화.")
                    print(f"   캐시된 파라미터: {cached_params}")
                    print(f"   현재 파라미터: {current_params}")
                    cache_remove(cache_path)
                    return None

                data = cache_data["data"]
            else:
                # 구버전: 리스트 형태 (파라미터 검증 없이 사용)
                print(f"⚠️ {label} 구버전 캐시 형식입니다. 파라미터 검증을 건너뜁니다.")
                data = cache_data

            # 데이터 검증
            if isinstance(data, list) and len(data) > 0:
                cache_type = "S3" if is_s3_path(cache_path) else "로컬"
                print(
                    f"📂 {label} 라벨 데이터 캐시 로드 ({cache_type}): {cache_path} ({len(data)}개 샘플)"
                )
                return data
            else:
                print(f"⚠️ {label} 캐시 데이터가 비어있거나 잘못된 형식입니다.")
                return None

        except Exception as e:
            print(f"⚠️ {label} 캐시 로드 실패: {e}")
            # 손상된 캐시 파일 삭제
            try:
                cache_remove(cache_path)
                print(f"🗑️ 손상된 캐시 파일 삭제: {cache_path}")
            except:
                pass
            return None
    return None


def process_data_in_batches(file_mapping, batch_size=100):
    """메모리 효율성을 위해 데이터를 배치 단위로 처리합니다."""
    all_files = list(file_mapping.items())
    total_files = len(all_files)

    print(f"📊 총 {total_files}개 파일을 {batch_size}개씩 배치 처리합니다.")

    # 진행률 표시 설정에 따라 tqdm 사용
    if ENABLE_PROGRESS_BAR:
        iterator = tqdm(range(0, total_files, batch_size), desc="배치 처리")
    else:
        iterator = range(0, total_files, batch_size)

    # MediaPipe 객체 재사용
    try:
        with MediaPipeManager() as holistic:
            print("✅ MediaPipe 객체 초기화 완료")

            for i in iterator:
                batch_files = all_files[i : i + batch_size]
                batch_data = []

                print(
                    f"🔄 배치 {i//batch_size + 1} 처리 중... ({len(batch_files)}개 파일)"
                )

                for filename, info in batch_files:
                    try:
                        print(f"  📹 {filename} 처리 중...")
                        landmarks = extract_landmarks_with_holistic(
                            info["path"], holistic
                        )
                        if not landmarks:
                            print(f"    ⚠️ 랜드마크 추출 실패: {filename}")
                            continue

                        processed_sequence = improved_preprocess_landmarks(landmarks)
                        if processed_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                            print(
                                f"    ⚠️ 시퀀스 형태 불일치: {filename} - {processed_sequence.shape}"
                            )
                            continue

                        batch_data.append(
                            {
                                "sequence": processed_sequence,
                                "label": info["label"],
                                "filename": filename,
                            }
                        )
                        print(f"    ✅ 성공: {filename}")

                    except Exception as e:
                        print(f"    ❌ 오류: {filename} - {e}")
                        continue

                print(f"✅ 배치 {i//batch_size + 1} 완료: {len(batch_data)}개 성공")
                yield batch_data

    except Exception as e:
        print(f"❌ MediaPipe 처리 중 오류: {e}")
        yield []


def extract_and_cache_label_data_optimized(file_mapping, label):
    """메모리 효율적인 라벨별 데이터 추출 및 캐싱"""
    print(f"\n🔄 {label} 라벨 데이터 추출 중...")

    # 캐시 확인
    cached_data = load_label_cache(label)
    if cached_data:
        print(f"✅ {label} 라벨 캐시 데이터 사용: {len(cached_data)}개 샘플")
        return cached_data

    # 해당 라벨의 파일들만 필터링
    label_files = {
        filename: info
        for filename, info in file_mapping.items()
        if info["label"] == label
    }

    if not label_files:
        print(f"⚠️ {label} 라벨에 해당하는 파일이 없습니다.")
        return []

    label_data = []

    # 배치 단위로 처리
    for batch in process_data_in_batches(
        label_files, batch_size=BATCH_SIZE_FOR_PROCESSING
    ):
        for item in batch:
            if item["label"] == label:
                # 원본 데이터 추가
                label_data.append(item["sequence"])

                # 증강 데이터 추가
                for _ in range(AUGMENTATIONS_PER_VIDEO):
                    try:
                        augmented = augment_sequence_improved(item["sequence"])
                        if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                            label_data.append(augmented)
                    except Exception as e:
                        print(f"⚠️ 증강 중 오류: {e}")
                        continue

    print(f"✅ {label} 라벨 데이터 추출 완료: {len(label_data)}개 샘플")

    # 캐시에 저장
    save_label_cache(label, label_data)

    return label_data


def generate_balanced_none_class_data(file_mapping, none_class, target_count=None):
    """다른 클래스와 균형있는 None 클래스 데이터를 생성하고 캐시에 저장합니다."""
    print(f"\n✨ '{none_class}' 클래스 데이터 생성 중...")

    # 기존 캐시 확인 (target_count 정보 포함)
    if target_count is not None:
        cached_none_data = load_none_class_cache(none_class, target_count)
    else:
        cached_none_data = load_label_cache(none_class)  # 기존 방식으로 폴백

    if cached_none_data:
        print(
            f"✅ {none_class} 클래스 캐시 데이터 사용: {len(cached_none_data)}개 샘플"
        )
        return cached_none_data

    # 목표 개수 계산 (다른 클래스의 평균 개수)
    if target_count is None:
        # 다른 클래스들의 원본 파일 개수 계산
        other_class_counts = []
        for filename, info in file_mapping.items():
            if info["label"] != none_class:
                other_class_counts.append(info["label"])

        # 라벨별 개수 집계
        from collections import Counter

        label_counts = Counter(other_class_counts)

        if label_counts:
            # 다른 클래스들의 평균 개수 계산 (증강 후 예상 개수)
            avg_original_count = sum(label_counts.values()) / len(label_counts)
            target_count = int(avg_original_count * (1 + AUGMENTATIONS_PER_VIDEO))
            print(
                f"📊 다른 클래스 평균: {avg_original_count:.1f}개 → 목표 None 클래스: {target_count}개"
            )
        else:
            target_count = 100  # 기본값
            print(f"📊 기본 목표 None 클래스: {target_count}개")

    none_samples = []
    source_videos = list(file_mapping.keys())

    # 목표 개수에 도달할 때까지 반복
    video_index = 0
    while len(none_samples) < target_count and video_index < len(source_videos):
        filename = source_videos[video_index % len(source_videos)]  # 순환 사용
        file_path = file_mapping[filename]["path"]

        try:
            # MediaPipe 객체 재사용 (한 번에 하나씩 처리)
            with MediaPipeManager() as holistic:
                landmarks = extract_landmarks_with_holistic(file_path, holistic)

                if landmarks and len(landmarks) > 10:
                    # 영상의 시작, 1/4, 1/2, 3/4, 끝 지점에서 프레임 추출
                    frame_indices = [
                        0,
                        len(landmarks) // 4,
                        len(landmarks) // 2,
                        3 * len(landmarks) // 4,
                        -1,
                    ]

                    for idx in frame_indices:
                        if len(none_samples) >= target_count:
                            break

                        static_landmarks = [landmarks[idx]] * TARGET_SEQ_LENGTH
                        static_sequence = improved_preprocess_landmarks(
                            static_landmarks
                        )

                        if static_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                            continue

                        # 정적 시퀀스 추가
                        none_samples.append(static_sequence)

                        # 미세한 움직임 추가 (노이즈) - 목표 개수 제한
                        for _ in range(
                            min(
                                NONE_CLASS_AUGMENTATIONS_PER_FRAME,
                                target_count - len(none_samples),
                            )
                        ):
                            if len(none_samples) >= target_count:
                                break
                            augmented = augment_sequence_improved(
                                static_sequence, noise_level=NONE_CLASS_NOISE_LEVEL
                            )
                            if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                                none_samples.append(augmented)

                    # 느린 전환 데이터 생성 (목표 개수 제한)
                    if len(none_samples) < target_count:
                        start_frame_lm = landmarks[0]
                        middle_frame_lm = landmarks[len(landmarks) // 2]

                        transition_landmarks = []
                        for i in range(TARGET_SEQ_LENGTH):
                            alpha = i / (TARGET_SEQ_LENGTH - 1)
                            interp_frame = {}
                            for key in ["pose", "left_hand", "right_hand"]:
                                if start_frame_lm.get(key) and middle_frame_lm.get(key):
                                    interp_lm = []
                                    start_lms = start_frame_lm[key].landmark
                                    mid_lms = middle_frame_lm[key].landmark
                                    for j in range(len(start_lms)):
                                        new_x = (
                                            start_lms[j].x * (1 - alpha)
                                            + mid_lms[j].x * alpha
                                        )
                                        new_y = (
                                            start_lms[j].y * (1 - alpha)
                                            + mid_lms[j].y * alpha
                                        )
                                        new_z = (
                                            start_lms[j].z * (1 - alpha)
                                            + mid_lms[j].z * alpha
                                        )
                                        interp_lm.append(
                                            type(
                                                "obj",
                                                (object,),
                                                {"x": new_x, "y": new_y, "z": new_z},
                                            )
                                        )
                                    interp_frame[key] = type(
                                        "obj", (object,), {"landmark": interp_lm}
                                    )
                                else:
                                    interp_frame[key] = None
                            transition_landmarks.append(interp_frame)

                        transition_sequence = improved_preprocess_landmarks(
                            transition_landmarks
                        )
                        if transition_sequence.shape == (TARGET_SEQ_LENGTH, 675):
                            none_samples.append(transition_sequence)

        except Exception as e:
            print(f"⚠️ None 클래스 데이터 생성 중 오류: {filename}, 오류: {e}")

        video_index += 1

    print(
        f"✅ {none_class} 클래스 데이터 생성 완료: {len(none_samples)}개 샘플 (목표: {target_count}개)"
    )

    # 캐시에 저장
    save_none_class_cache(none_class, none_samples, target_count)

    return none_samples


def validate_video_roots():
    """VIDEO_ROOTS의 모든 디렉토리가 존재하는지 확인합니다."""
    print("🔍 비디오 루트 디렉토리 검증 중...")
    valid_roots = []

    for (range_start, range_end), root_path in VIDEO_ROOTS:
        if os.path.exists(root_path):
            valid_roots.append(((range_start, range_end), root_path))
            print(f"✅ {range_start}~{range_end}: {root_path}")
        else:
            print(f"❌ {range_start}~{range_end}: {root_path} (존재하지 않음)")

    return valid_roots


def find_file_in_directory(directory, filename_pattern):
    """디렉토리에서 파일 패턴에 맞는 파일을 찾습니다."""
    if not os.path.exists(directory):
        return None

    # 파일명에서 확장자 제거
    base_name = filename_pattern.split(".")[0]

    # 가능한 확장자들 (config에서 가져옴)
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(directory, base_name + ext)
        if os.path.exists(candidate):
            return candidate

    return None


def get_video_root_and_path(filename):
    """파일명에서 번호를 추출해 올바른 VIDEO_ROOT 경로와 실제 파일 경로를 반환합니다."""
    try:
        # 파일 확장자 제거
        file_id = filename.split(".")[0]

        # KETI_SL_ 형식 확인
        if not file_id.startswith("KETI_SL_"):
            print(f"⚠️ KETI_SL_ 형식이 아닌 파일명: {filename}")
            return None

        # 숫자 부분 추출
        number_str = file_id.replace("KETI_SL_", "")
        if not number_str.isdigit():
            print(f"⚠️ 숫자가 아닌 파일명: {filename}")
            return None

        num = int(number_str)

        # 적절한 디렉토리 찾기
        target_root = None
        for (range_start, range_end), root_path in VIDEO_ROOTS:
            if range_start <= num <= range_end:
                target_root = root_path
                break

        if target_root is None:
            print(f"⚠️ 번호 {num}에 해당하는 디렉토리를 찾을 수 없음: {filename}")
            return None

        # 파일 찾기
        file_path = find_file_in_directory(target_root, filename)
        if file_path:
            return file_path

        print(f"⚠️ 파일을 찾을 수 없음: {filename} (디렉토리: {target_root})")
        return None

    except Exception as e:
        print(f"⚠️ 파일명 파싱 오류: {filename}, 오류: {e}")
        return None


def normalize_sequence_length(sequence, target_length=30):
    """시퀀스 길이를 정규화합니다."""
    current_length = len(sequence)

    if current_length == target_length:
        return sequence

    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)

    normalized_sequence = []
    for i in range(sequence.shape[1]):
        f = interp1d(
            x_old,
            sequence[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        normalized_sequence.append(f(x_new))

    return np.array(normalized_sequence).T


def extract_dynamic_features(sequence):
    """속도와 가속도 특징을 추출합니다."""
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    dynamic_features = np.concatenate([sequence, velocity, acceleration], axis=1)
    return dynamic_features


def convert_to_relative_coordinates(landmarks_list):
    """절대 좌표를 어깨 중심 상대 좌표계로 변환합니다."""
    relative_landmarks = []

    for frame in landmarks_list:
        if not frame["pose"]:
            relative_landmarks.append(frame)
            continue

        pose_landmarks = frame["pose"].landmark

        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2

        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        if shoulder_width == 0:
            shoulder_width = 1.0

        new_frame = {}

        if frame["pose"]:
            relative_pose = []
            for landmark in pose_landmarks:
                rel_x = (landmark.x - shoulder_center_x) / shoulder_width
                rel_y = (landmark.y - shoulder_center_y) / shoulder_width
                rel_z = (landmark.z - shoulder_center_z) / shoulder_width
                relative_pose.append([rel_x, rel_y, rel_z])
            new_frame["pose"] = relative_pose

        for hand_key in ["left_hand", "right_hand"]:
            if frame[hand_key]:
                relative_hand = []
                for landmark in frame[hand_key].landmark:
                    rel_x = (landmark.x - shoulder_center_x) / shoulder_width
                    rel_y = (landmark.y - shoulder_center_y) / shoulder_width
                    rel_z = (landmark.z - shoulder_center_z) / shoulder_width
                    relative_hand.append([rel_x, rel_y, rel_z])
                new_frame[hand_key] = relative_hand
            else:
                new_frame[hand_key] = None

        relative_landmarks.append(new_frame)

    return relative_landmarks


def improved_preprocess_landmarks(landmarks_list):
    """개선된 랜드마크 전처리 함수."""
    if not landmarks_list:
        return np.zeros((TARGET_SEQ_LENGTH, 675))

    relative_landmarks = convert_to_relative_coordinates(landmarks_list)

    processed_frames = []
    for frame in relative_landmarks:
        combined = []
        for key in ["pose", "left_hand", "right_hand"]:
            if frame[key]:
                if isinstance(frame[key], list):
                    combined.extend(frame[key])
                else:
                    combined.extend([[l.x, l.y, l.z] for l in frame[key].landmark])
            else:
                num_points = {"pose": 33, "left_hand": 21, "right_hand": 21}[key]
                combined.extend([[0, 0, 0]] * num_points)

        if combined:
            processed_frames.append(np.array(combined).flatten())
        else:
            processed_frames.append(np.zeros(75 * 3))

    if not processed_frames:
        return np.zeros((TARGET_SEQ_LENGTH, 675))

    sequence = np.array(processed_frames)

    if len(sequence) > 0:
        try:
            sequence = normalize_sequence_length(sequence, TARGET_SEQ_LENGTH)
            sequence = extract_dynamic_features(sequence)

            # 정규화 개선: 더 강한 정규화
            sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)

            return sequence
        except Exception as e:
            print(f"⚠️ 시퀀스 처리 중 오류 발생: {e}")
            return np.zeros((TARGET_SEQ_LENGTH, 675))

    return np.zeros((TARGET_SEQ_LENGTH, 675))


def create_simple_model(input_shape, num_classes):
    """간단하고 효과적인 모델을 생성합니다."""
    inputs = Input(shape=input_shape)

    # 1D CNN
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # LSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)

    # Dense layers
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_model_info():
    """기존 모델 정보를 로드합니다."""
    try:
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ 모델 정보 로드 중 오류: {e}")
    return None


def save_model_info(actions, model_path, info_path, training_stats):
    """모델 정보를 JSON 파일로 저장합니다."""
    model_info = {
        "model_path": model_path,
        "created_at": datetime.now().isoformat(),
        "labels": actions,
        "label_mapping": {label: idx for idx, label in enumerate(actions)},
        "num_classes": len(actions),
        "input_shape": [TARGET_SEQ_LENGTH, 675],  # 시퀀스 길이, 특징 수
        "training_stats": training_stats,
        "model_type": "LSTM",
        "description": "수어 인식 모델 - LSTM 기반",
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"📄 모델 정보 저장: {info_path}")


def augment_sequence_improved(
    sequence,
    noise_level=AUGMENTATION_NOISE_LEVEL,
    scale_range=AUGMENTATION_SCALE_RANGE,
    rotation_range=AUGMENTATION_ROTATION_RANGE,
):
    """개선된 시퀀스 증강."""
    augmented = sequence.copy()

    # 노이즈 추가
    noise = np.random.normal(0, noise_level, augmented.shape)
    augmented += noise

    # 스케일링
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    augmented *= scale_factor

    # 시간축에서의 회전 (시프트)
    shift = np.random.randint(-3, 4)
    if shift > 0:
        augmented = np.roll(augmented, shift, axis=0)
    elif shift < 0:
        augmented = np.roll(augmented, shift, axis=0)

    return augmented


def extract_landmarks_with_holistic(video_path, holistic):
    """전달받은 MediaPipe 객체를 사용하여 랜드마크를 추출합니다."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ 비디오 파일을 열 수 없음: {video_path}")
            return None

        # 비디오 정보 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"    📊 비디오 정보: {total_frames}프레임, {fps:.1f}fps")

        landmarks_list = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 처리
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            frame_data = {
                "pose": results.pose_landmarks,
                "left_hand": results.left_hand_landmarks,
                "right_hand": results.right_hand_landmarks,
            }
            landmarks_list.append(frame_data)
            frame_count += 1

            # 진행상황 표시 (10프레임마다)
            # if frame_count % 10 == 0:
            # print(f"      📹 프레임 {frame_count}/{total_frames} 처리 중...")

        cap.release()
        print(f"    ✅ 랜드마크 추출 완료: {len(landmarks_list)}프레임")
        return landmarks_list

    except (cv2.error, OSError) as e:
        print(f"⚠️ 비디오 파일 읽기 오류: {video_path}, 오류: {e}")
        return None
    except Exception as e:
        print(f"⚠️ 랜드마크 추출 중 예상치 못한 오류: {video_path}, 오류: {e}")
        return None


def get_action_index(label, actions):
    """라벨의 인덱스를 반환합니다."""
    return actions.index(label)


def get_all_video_paths():
    video_paths = []

    return video_paths


def cleanup_old_checkpoints(
    checkpoint_dir="checkpoints", keep_best=True, max_checkpoints=10
):
    """개선된 체크포인트 정리 함수 - 완전하고 안전한 정리"""
    if not os.path.exists(checkpoint_dir):
        return

    print(f"🧹 체크포인트 디렉토리 정리 중: {checkpoint_dir}")

    try:
        # 디스크 공간 확인
        import shutil

        total, used, free = shutil.disk_usage(checkpoint_dir)
        free_gb = free / (1024**3)
        print(f"   💾 사용 가능한 디스크 공간: {free_gb:.2f}GB")

        if free_gb < 0.5:  # 500MB 미만
            print("   ⚠️ 디스크 공간이 부족합니다. 더 적극적인 정리를 수행합니다.")
            max_checkpoints = 5

        # 에폭별 체크포인트 파일들 찾기 (에폭 기반 파일명)
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith("model-epoch-") and file.endswith(".keras"):
                checkpoint_files.append(file)

        if not checkpoint_files:
            print("   📁 정리할 에폭 체크포인트가 없습니다.")
            return

        print(f"   📊 발견된 체크포인트: {len(checkpoint_files)}개")

        # 에폭 기준으로 정렬 (가장 최근이 마지막)
        def extract_epoch(filename):
            try:
                epoch_part = filename.split("-")[2].split(".")[0]  # "05"
                return int(epoch_part)
            except:
                return 0

        checkpoint_files.sort(key=extract_epoch)

        # 보존할 체크포인트 수 결정
        files_to_keep = (
            checkpoint_files[-max_checkpoints:]
            if len(checkpoint_files) > max_checkpoints
            else []
        )
        files_to_delete = [f for f in checkpoint_files if f not in files_to_keep]

        print(f"   🎯 보존할 체크포인트: {len(files_to_keep)}개")
        print(f"   🗑️ 삭제할 체크포인트: {len(files_to_delete)}개")

        # 파일 삭제 (체크포인트와 info 파일 모두)
        deleted_count = 0
        for file in files_to_delete:
            try:
                # 체크포인트 파일 삭제
                checkpoint_path = os.path.join(checkpoint_dir, file)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    deleted_count += 1

                # 해당하는 info 파일도 삭제
                info_path = checkpoint_path.replace(".keras", "_info.json")
                if os.path.exists(info_path):
                    os.remove(info_path)
                    deleted_count += 1

            except Exception as e:
                print(f"   ⚠️ 파일 삭제 실패: {file} - {e}")

        print(f"   ✅ {deleted_count}개 파일 삭제됨")

        # 최고 성능 모델은 유지
        if keep_best and os.path.exists(
            os.path.join(checkpoint_dir, "best_model.keras")
        ):
            print("   💎 최고 성능 모델 유지됨")

            # best_model_info.json도 확인
            best_info_path = os.path.join(checkpoint_dir, "best_model_info.json")
            if not os.path.exists(best_info_path):
                print("   ⚠️ best_model_info.json이 없습니다.")

        # 정리 후 디스크 사용량 확인
        total_after, used_after, free_after = shutil.disk_usage(checkpoint_dir)
        freed_gb = (free_after - free) / (1024**3)
        if freed_gb > 0:
            print(f"   💾 정리로 {freed_gb:.2f}GB 공간 확보됨")

    except Exception as e:
        print(f"   ❌ 체크포인트 정리 중 오류: {e}")
        import traceback

        traceback.print_exc()


class ImprovedCheckpointInfoCallback(tf.keras.callbacks.Callback):
    """개선된 체크포인트 정보 저장 콜백 - 효율적이고 안전한 처리"""

    def __init__(self, actions, checkpoint_dir, training_stats):
        super().__init__()
        self.actions = actions
        self.checkpoint_dir = checkpoint_dir
        self.training_stats = training_stats
        self.saved_checkpoints = set()  # 이미 저장한 체크포인트 추적
        self.last_scan_time = 0  # 마지막 스캔 시간
        self.scan_interval = 5  # 스캔 간격 (초)

        # 디스크 공간 확인
        self._check_disk_space()

    def _check_disk_space(self):
        """디스크 공간 확인"""
        try:
            import shutil

            total, used, free = shutil.disk_usage(self.checkpoint_dir)
            free_gb = free / (1024**3)
            if free_gb < 1.0:  # 1GB 미만
                print(f"⚠️ 디스크 공간 부족: {free_gb:.2f}GB 남음")
        except Exception as e:
            print(f"⚠️ 디스크 공간 확인 실패: {e}")

    def _should_scan_directory(self):
        """디렉토리 스캔이 필요한지 확인"""
        import time

        current_time = time.time()
        if current_time - self.last_scan_time > self.scan_interval:
            self.last_scan_time = current_time
            return True
        return False

    def on_epoch_end(self, epoch, logs=None):
        # 스캔 간격 제어로 성능 최적화
        if not self._should_scan_directory():
            return

        try:
            # 에폭별 체크포인트 파일들 확인 (에폭 기반 파일명)
            checkpoint_files = [
                f
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith("model-epoch-") and f.endswith(".keras")
            ]

            for checkpoint_file in checkpoint_files:
                # 이미 처리한 체크포인트는 건너뛰기
                if checkpoint_file in self.saved_checkpoints:
                    continue

                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)

                # 파일명에서 에폭 정보 추출
                try:
                    # "model-epoch-05.keras" -> epoch=5
                    parts = checkpoint_file.split("-")
                    epoch_part = parts[2].split(".")[0]  # "05"
                    epoch_num = int(epoch_part)

                    # 성능 정보는 logs에서 가져오기
                    val_accuracy = logs.get("val_accuracy", 0) if logs else 0

                except (IndexError, ValueError) as e:
                    print(f"⚠️ 파일명 파싱 실패: {checkpoint_file} - {e}")
                    epoch_num = epoch + 1
                    val_accuracy = logs.get("val_accuracy", 0) if logs else 0

                # 체크포인트별 모델 정보 생성 (최종 결과 양식과 일치)
                checkpoint_info = {
                    "model_path": checkpoint_path,
                    "created_at": datetime.now().isoformat(),
                    "labels": self.actions,
                    "label_mapping": {
                        label: idx for idx, label in enumerate(self.actions)
                    },
                    "num_classes": len(self.actions),
                    "input_shape": [TARGET_SEQ_LENGTH, 675],
                    "training_stats": {
                        **self.training_stats,
                        "checkpoint_epoch": epoch_num,
                        "checkpoint_accuracy": logs.get("accuracy", 0) if logs else 0,
                        "checkpoint_val_accuracy": val_accuracy,
                        "checkpoint_loss": logs.get("loss", 0) if logs else 0,
                        "checkpoint_val_loss": logs.get("val_loss", 0) if logs else 0,
                    },
                    "model_type": "LSTM",
                    "description": f"수어 인식 모델 - LSTM 기반 (Epoch {epoch_num}, Val Acc: {val_accuracy:.4f})",
                }

                # 체크포인트별 info 파일 저장
                checkpoint_info_path = checkpoint_path.replace(".keras", "_info.json")
                try:
                    with open(checkpoint_info_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_info, f, ensure_ascii=False, indent=2)

                    # 출력을 줄여서 중첩 방지 (에폭 끝에만 출력)
                    if epoch_num % 5 == 0:  # 5 에폭마다만 출력
                        print(f"📄 체크포인트 정보 저장: Epoch {epoch_num}")
                    self.saved_checkpoints.add(checkpoint_file)

                    # 메모리 최적화: 세트 크기 제한
                    if len(self.saved_checkpoints) > 100:
                        # 가장 오래된 항목들 제거
                        oldest_items = list(self.saved_checkpoints)[:20]
                        for item in oldest_items:
                            self.saved_checkpoints.remove(item)

                except Exception as e:
                    print(f"⚠️ 체크포인트 정보 저장 실패: {checkpoint_file} - {e}")

        except Exception as e:
            print(f"⚠️ 체크포인트 처리 중 오류: {e}")

    def on_train_end(self, logs=None):
        """학습 종료 시 최고 성능 체크포인트 복사"""
        try:
            # 가장 높은 성능의 체크포인트 찾기
            checkpoint_files = [
                f
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith("model-epoch-") and f.endswith(".keras")
            ]

            if checkpoint_files:
                # 에폭 기준으로 정렬 (가장 최근이 마지막)
                def extract_epoch(filename):
                    try:
                        epoch_part = filename.split("-")[2].split(".")[0]  # "05"
                        return int(epoch_part)
                    except:
                        return 0

                # 가장 최근 체크포인트를 best로 선택
                best_checkpoint = max(checkpoint_files, key=extract_epoch)
                best_path = os.path.join(self.checkpoint_dir, best_checkpoint)
                best_final_path = os.path.join(self.checkpoint_dir, "best_model.keras")

                # 최고 성능 모델을 best_model.keras로 복사
                import shutil

                shutil.copy2(best_path, best_final_path)

                # best_model_info.json도 복사
                best_info_path = best_path.replace(".keras", "_info.json")
                best_final_info_path = best_final_path.replace(".keras", "_info.json")
                if os.path.exists(best_info_path):
                    shutil.copy2(best_info_path, best_final_info_path)

                print(f"🏆 최신 체크포인트 복사: {best_checkpoint} -> best_model.keras")

        except Exception as e:
            print(f"⚠️ 최고 성능 체크포인트 복사 실패: {e}")


def load_latest_checkpoint(checkpoint_dir="checkpoints"):
    """가장 최근 체크포인트를 로드합니다."""
    if not os.path.exists(checkpoint_dir):
        return None, None, 0

    try:
        # 체크포인트 파일들 찾기
        checkpoint_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("model-epoch-") and f.endswith(".keras")
        ]

        if not checkpoint_files:
            return None, None, 0

        # 에폭 기준으로 정렬 (가장 최근이 마지막)
        def extract_epoch(filename):
            try:
                epoch_part = filename.split("-")[2]  # "05"
                return int(epoch_part)
            except:
                return 0

        checkpoint_files.sort(key=extract_epoch)
        latest_checkpoint = checkpoint_files[-1]
        latest_path = os.path.join(checkpoint_dir, latest_checkpoint)

        # 에폭 번호 추출
        latest_epoch = extract_epoch(latest_checkpoint)

        # 해당하는 info 파일 로드
        info_path = latest_path.replace(".keras", "_info.json")
        checkpoint_info = None
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    checkpoint_info = json.load(f)
            except Exception as e:
                print(f"⚠️ 체크포인트 정보 로드 실패: {e}")

        print(f"📂 최신 체크포인트 로드: {latest_checkpoint} (Epoch {latest_epoch})")
        return latest_path, checkpoint_info, latest_epoch

    except Exception as e:
        print(f"⚠️ 체크포인트 로드 중 오류: {e}")
        return None, None, 0


def resume_training_from_checkpoint(
    model, checkpoint_path, checkpoint_info, latest_epoch
):
    """체크포인트에서 학습을 재개합니다."""
    try:
        print(f"🔄 체크포인트에서 학습 재개: {checkpoint_path}")

        # 모델 가중치 로드
        model.load_weights(checkpoint_path)
        print(f"✅ 모델 가중치 로드 완료 (Epoch {latest_epoch})")

        # 체크포인트 정보 출력
        if checkpoint_info:
            print(f"📊 체크포인트 성능:")
            print(f"   - 검증 정확도: {checkpoint_info.get('val_accuracy', 'N/A')}")
            print(
                f"   - 검증 손실: {checkpoint_info.get('training_stats', {}).get('checkpoint_val_loss', 'N/A')}"
            )

        return True

    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return False


def save_none_class_cache(none_class, data, target_count):
    """None 클래스 데이터를 캐시에 저장합니다. target_count 정보도 포함합니다."""
    cache_path = get_label_cache_path(none_class)

    # 캐시에 저장할 데이터와 파라미터 정보
    cache_data = {
        "data": data,
        "parameters": {
            "TARGET_SEQ_LENGTH": TARGET_SEQ_LENGTH,
            "AUGMENTATIONS_PER_VIDEO": AUGMENTATIONS_PER_VIDEO,
            "AUGMENTATION_NOISE_LEVEL": AUGMENTATION_NOISE_LEVEL,
            "AUGMENTATION_SCALE_RANGE": AUGMENTATION_SCALE_RANGE,
            "AUGMENTATION_ROTATION_RANGE": AUGMENTATION_ROTATION_RANGE,
            "NONE_CLASS_NOISE_LEVEL": NONE_CLASS_NOISE_LEVEL,
            "NONE_CLASS_AUGMENTATIONS_PER_FRAME": NONE_CLASS_AUGMENTATIONS_PER_FRAME,
            # 데이터 개수 관련 파라미터 추가
            "LABEL_MAX_SAMPLES_PER_CLASS": LABEL_MAX_SAMPLES_PER_CLASS,
            "MIN_SAMPLES_PER_CLASS": MIN_SAMPLES_PER_CLASS,
            # None 클래스 특별 파라미터
            "TARGET_NONE_COUNT": target_count,
        },
    }

    # S3 호환 캐시 저장
    try:
        if is_s3_path(cache_path):
            # S3에서는 put_object가 원자적이므로 직접 저장
            success = cache_save_pickle(cache_path, cache_data)
            if success:
                print(
                    f"💾 {none_class} 클래스 데이터 캐시 저장 (S3): {cache_path} ({len(data)}개 샘플, 목표: {target_count}개)"
                )
            else:
                raise Exception("S3 캐시 저장 실패")
        else:
            # 로컬에서는 임시 파일 방식 사용 (원자적 쓰기)
            temp_path = cache_path + ".tmp"

            with open(temp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 성공적으로 저장되면 최종 위치로 이동
            os.replace(temp_path, cache_path)
            print(
                f"💾 {none_class} 클래스 데이터 캐시 저장 (로컬): {cache_path} ({len(data)}개 샘플, 목표: {target_count}개)"
            )

    except Exception as e:
        # 오류 발생 시 임시 파일 정리 (로컬 파일인 경우만)
        if not is_s3_path(cache_path):
            temp_path = cache_path + ".tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise e


def load_none_class_cache(none_class, target_count):
    """None 클래스 데이터를 캐시에서 로드합니다. target_count 정보도 검증합니다."""
    cache_path = get_label_cache_path(none_class)
    if cache_exists(cache_path):
        try:
            # S3 호환 캐시 로드
            cache_data = cache_load_pickle(cache_path)
            if cache_data is None:
                return None

            # 캐시 형식 확인 (구버전 호환성)
            if (
                isinstance(cache_data, dict)
                and "data" in cache_data
                and "parameters" in cache_data
            ):
                # 새 형식: 파라미터 검증
                cached_params = cache_data["parameters"]
                current_params = {
                    "TARGET_SEQ_LENGTH": TARGET_SEQ_LENGTH,
                    "AUGMENTATIONS_PER_VIDEO": AUGMENTATIONS_PER_VIDEO,
                    "AUGMENTATION_NOISE_LEVEL": AUGMENTATION_NOISE_LEVEL,
                    "AUGMENTATION_SCALE_RANGE": AUGMENTATION_SCALE_RANGE,
                    "AUGMENTATION_ROTATION_RANGE": AUGMENTATION_ROTATION_RANGE,
                    "NONE_CLASS_NOISE_LEVEL": NONE_CLASS_NOISE_LEVEL,
                    "NONE_CLASS_AUGMENTATIONS_PER_FRAME": NONE_CLASS_AUGMENTATIONS_PER_FRAME,
                    # 데이터 개수 관련 파라미터 추가
                    "LABEL_MAX_SAMPLES_PER_CLASS": LABEL_MAX_SAMPLES_PER_CLASS,
                    "MIN_SAMPLES_PER_CLASS": MIN_SAMPLES_PER_CLASS,
                    # None 클래스 특별 파라미터
                    "TARGET_NONE_COUNT": target_count,
                }

                # 파라미터 비교
                if cached_params != current_params:
                    print(f"⚠️ {none_class} 캐시 파라미터가 다릅니다. 캐시 무효화.")
                    print(f"   캐시된 파라미터: {cached_params}")
                    print(f"   현재 파라미터: {current_params}")
                    cache_remove(cache_path)
                    return None

                data = cache_data["data"]
            else:
                # 구버전: 리스트 형태 (파라미터 검증 없이 사용)
                print(
                    f"⚠️ {none_class} 구버전 캐시 형식입니다. 파라미터 검증을 건너뜁니다."
                )
                data = cache_data

            # 데이터 검증
            if isinstance(data, list) and len(data) > 0:
                cache_type = "S3" if is_s3_path(cache_path) else "로컬"
                print(
                    f"📂 {none_class} 클래스 데이터 캐시 로드 ({cache_type}): {cache_path} ({len(data)}개 샘플, 목표: {target_count}개)"
                )
                return data
            else:
                print(f"⚠️ {none_class} 캐시 데이터가 비어있거나 잘못된 형식입니다.")
                return None

        except Exception as e:
            print(f"⚠️ {none_class} 캐시 로드 실패: {e}")
            # 손상된 캐시 파일 삭제
            try:
                cache_remove(cache_path)
                print(f"🗑️ 손상된 캐시 파일 삭제: {cache_path}")
            except:
                pass
            return None
    return None


def main():
    """메인 실행 함수"""
    params = sys.argv[1]
    with open(params, "r") as f:
        params = json.load(f)
    label_dict = params["label_dict"]

    ACTIONS = list(label_dict.keys())
    NONE_CLASS = ACTIONS[-1]

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
        if label == NONE_CLASS:
            continue  # None 클래스는 예외
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
        if info["label"] != NONE_CLASS:
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

        if label == NONE_CLASS:
            label_data = generate_balanced_none_class_data(
                file_mapping, NONE_CLASS, target_none_count
            )
        else:
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

    X = np.array(X)
    y = np.array(y)

    # 모델 학습
    print("\n🏋️‍♀️ 모델 학습 시작")
    print(
        f"   📊 Early Stopping: patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}"
    )
    print(f"   📊 Learning Rate: patience={REDUCE_LR_PATIENCE}, min_lr={MIN_LR}")
    print(
        f"   📊 정규화: L2={USE_L2_REGULARIZATION}, BatchNorm={USE_BATCH_NORMALIZATION}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = create_simple_model(
        input_shape=(X.shape[1], X.shape[2]), num_classes=len(ACTIONS)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=1.0  # 그래디언트 클리핑 추가
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n--- 모델 구조 ---")
    model.summary()

    # 체크포인트용 training_stats 미리 정의
    training_stats = {
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "augmentations_per_video": AUGMENTATIONS_PER_VIDEO,
        "target_sequence_length": TARGET_SEQ_LENGTH,
        "model_parameters": {
            "lstm_units_1": MODEL_LSTM_UNITS_1,
            "lstm_units_2": MODEL_LSTM_UNITS_2,
            "dense_units": MODEL_DENSE_UNITS,
            "dropout_rate": MODEL_DROPOUT_RATE,
            "l2_regularization": USE_L2_REGULARIZATION,
            "batch_normalization": USE_BATCH_NORMALIZATION,
        },
        "training_parameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
        },
    }

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 체크포인트 로딩 및 학습 재개
    best_checkpoint_path, best_checkpoint_info, best_epoch = load_latest_checkpoint(
        CHECKPOINT_DIR
    )

    # 체크포인트에서 재개할지 결정
    resume_from_checkpoint = False
    if best_checkpoint_path:
        print(f"📂 발견된 체크포인트: {best_checkpoint_path} (Epoch {best_epoch})")

        # 사용자 입력 또는 자동 결정 (여기서는 자동으로 재개)
        resume_from_checkpoint = True

        if resume_from_checkpoint:
            if resume_training_from_checkpoint(
                model, best_checkpoint_path, best_checkpoint_info, best_epoch
            ):
                print("✅ 체크포인트에서 학습 재개 준비 완료")
                initial_epoch = best_epoch
            else:
                print("❌ 체크포인트 로드 실패, 처음부터 시작")
                initial_epoch = 0
        else:
            print("🔄 처음부터 학습 시작")
            initial_epoch = 0
    else:
        print("🆕 새로운 학습 시작")
        initial_epoch = 0

    # 콜백 설정
    callbacks = [
        # 통합된 체크포인트 저장 (에폭 기반 파일명)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "model-epoch-{epoch:02d}.keras"),
            save_best_only=False,
            save_freq=5,  # 5 에폭마다
            verbose=0,  # 출력 중첩 방지를 위해 0으로 변경
        ),
        # 개선된 Early Stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning Rate 감소
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        ImprovedCheckpointInfoCallback(ACTIONS, CHECKPOINT_DIR, training_stats),
    ]

    # 모델 학습 (체크포인트에서 재개)
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2,  # 더 깔끔한 진행률 표시
        initial_epoch=initial_epoch,  # 체크포인트에서 재개
    )

    print(f"🧠 수정된 모델 저장: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    # 평가
    print("\n--- 모델 평가 ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"🚀 테스트 정확도: {accuracy * 100:.2f}%")

    # 예측 결과 확인
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = y_test

    print("\n--- 클래스별 정확도 ---")
    class_accuracies = {}
    for i in range(len(ACTIONS)):
        class_mask = y_true_classes == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(
                y_pred_classes[class_mask] == y_true_classes[class_mask]
            )
            class_accuracies[ACTIONS[i]] = class_accuracy
            print(f"{ACTIONS[i]}: {class_accuracy:.4f}")

    # 모델 정보 저장 (최종 결과 추가)
    training_stats.update(
        {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "class_accuracies": class_accuracies,
        }
    )
    save_model_info(ACTIONS, MODEL_SAVE_PATH, MODEL_INFO_PATH, training_stats)

    print("\n✅ 모든 작업 완료!")
    print(f"📁 모델 저장 위치: {MODEL_SAVE_PATH}")
    print(f"📄 모델 정보 위치: {MODEL_INFO_PATH}")

    # 오래된 체크포인트 정리 (개선된 버전)
    cleanup_old_checkpoints(
        checkpoint_dir=CHECKPOINT_DIR, keep_best=True, max_checkpoints=10
    )


if __name__ == "__main__":
    print("🔧 학습 데이터 문제 해결 및 모델 재학습 시작")

    try:
        # 기존 모델 정보 로드
        model_info = load_model_info()
        if model_info:
            print(f"📋 기존 모델 정보 로드됨: {model_info['model_name']}")
            print(f"   - 정확도: {model_info['test_accuracy']:.4f}")
            print(f"   - 손실: {model_info['test_loss']:.4f}")
            print(f"   - 훈련 시간: {model_info['training_time']:.2f}초")

        # 데이터 처리 및 모델 재학습
        main()

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # MediaPipe 객체 정리
        MediaPipeManager.cleanup()
        print("\n🧹 리소스 정리 완료")
