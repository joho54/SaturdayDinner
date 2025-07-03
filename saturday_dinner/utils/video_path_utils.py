"""
비디오 경로 처리를 위한 공통 유틸리티 모듈
"""

import os
from typing import Optional, Dict, Tuple

# 비디오 루트 경로 설정 (전체 범위)
VIDEO_ROOTS = {
    (1, 3000): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/0001~3000(영상)",
    (3001, 6000): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/3001~6000(영상)",
    (6001, 8280): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/6001~8280(영상)",
    (8381, 9000): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/8381~9000(영상)",
    (9001, 9600): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/9001~9600(영상)",
    (9601, 10480): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/9601~10480(영상)",
    (10481, 12994): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/10481~12994",
    (12995, 15508): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/12995~15508",
    (15509, 18022): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/15509~18022",
    (18023, 20536): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/18023~20536",
    (20537, 23050): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/20537~23050",
    (23051, 25564): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/23051~25564",
    (25565, 28078): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/25565~28078",
    (28079, 30592): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/28079~30592",
    (30593, 33106): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/30593~33106",
    (33107, 35620): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/33107~35620",
    (36878, 40027): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/36878~40027",
    (40028, 43177): "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/40028~43177"
}

# 지원하는 비디오 확장자
VIDEO_EXTENSIONS = [".mp4", ".MP4", ".MOV", ".MTS", ".AVI", ".avi"]

def get_video_number(filename: str) -> Optional[int]:
    """파일명에서 비디오 번호를 추출합니다."""
    try:
        # filename이 문자열이 아닌 경우 문자열로 변환
        if not isinstance(filename, str):
            filename = str(filename)
        
        num_str = filename.split("_")[-1].split(".")[0]
        return int(num_str)
    except (ValueError, IndexError, AttributeError):
        return None

def get_root_path_for_number(num: int) -> Optional[str]:
    """번호에 해당하는 루트 경로를 반환합니다."""
    for (start, end), path in VIDEO_ROOTS.items():
        if start <= num <= end:
            return path
    return None

def get_video_root_and_path(filename: str, verbose: bool = True) -> Optional[str]:
    """
    파일명에서 번호를 추출해 올바른 VIDEO_ROOT 경로와 실제 파일 경로를 반환합니다.
    
    Args:
        filename: 비디오 파일명 (예: "KETI_SL_0000000419.MOV")
        verbose: 디버그 메시지 출력 여부
        
    Returns:
        실제 파일 경로 또는 None (파일을 찾을 수 없는 경우)
    """
    # 파일명에서 번호 추출
    num = get_video_number(filename)
    if num is None:
        if verbose:
            print(f"⚠️ 파일명에서 번호를 추출할 수 없습니다: {filename}")
        return None
    
    # 번호 범위에 해당하는 루트 경로 찾기
    root_path = get_root_path_for_number(num)
    if root_path is None:
        if verbose:
            print(f"⚠️ 번호 {num}에 해당하는 루트 경로를 찾을 수 없습니다: {filename}")
        return None
    
    # 기본 파일명 생성 (확장자 제외)
    num_str = str(num)
    base_name = "_".join(filename.split("_")[:-1]) + f"_{num:010d}"
    
    # 다양한 확장자로 파일 존재 확인
    for ext in VIDEO_EXTENSIONS:
        # 대소문자 변형도 시도
        for case_ext in [ext, ext.lower()]:
            candidate_path = os.path.join(root_path, base_name + case_ext)
            if os.path.exists(candidate_path):
                return candidate_path
    
    if verbose:
        print(f"⚠️ 파일을 찾을 수 없습니다: {filename} (검색 경로: {root_path})")
    return None

def get_all_video_paths() -> Dict[int, str]:
    """모든 비디오 번호와 경로를 반환합니다."""
    video_paths = {}
    
    for (start, end), root_path in VIDEO_ROOTS.items():
        if os.path.exists(root_path):
            for num in range(start, end + 1):
                # 다양한 확장자로 파일 존재 확인
                for ext in VIDEO_EXTENSIONS:
                    for case_ext in [ext, ext.lower()]:
                        candidate_path = os.path.join(root_path, f"KETI_SL_{num:010d}{case_ext}")
                        if os.path.exists(candidate_path):
                            video_paths[num] = candidate_path
                            break
                    if num in video_paths:
                        break
    
    return video_paths

def validate_video_paths() -> Dict[str, int]:
    """비디오 경로들의 유효성을 검사하고 통계를 반환합니다."""
    stats = {
        "total_ranges": len(VIDEO_ROOTS),
        "existing_roots": 0,
        "total_videos": 0,
        "missing_roots": []
    }
    
    for (start, end), root_path in VIDEO_ROOTS.items():
        if os.path.exists(root_path):
            stats["existing_roots"] += 1
            # 해당 범위의 비디오 개수 계산
            for num in range(start, end + 1):
                for ext in VIDEO_EXTENSIONS:
                    for case_ext in [ext, ext.lower()]:
                        candidate_path = os.path.join(root_path, f"KETI_SL_{num:010d}{case_ext}")
                        if os.path.exists(candidate_path):
                            stats["total_videos"] += 1
                            break
                    if os.path.exists(candidate_path):
                        break
        else:
            stats["missing_roots"].append(f"{start}~{end}: {root_path}")
    
    return stats

def print_video_path_stats():
    """비디오 경로 통계를 출력합니다."""
    stats = validate_video_paths()
    
    print("📊 비디오 경로 통계:")
    print(f"   - 총 범위 수: {stats['total_ranges']}")
    print(f"   - 존재하는 루트: {stats['existing_roots']}")
    print(f"   - 총 비디오 수: {stats['total_videos']}")
    
    if stats["missing_roots"]:
        print("   - 누락된 루트:")
        for missing in stats["missing_roots"]:
            print(f"     • {missing}")
    else:
        print("   - 모든 루트가 존재합니다.")

if __name__ == "__main__":
    # 테스트 및 통계 출력
    print_video_path_stats()
    
    # 예시 파일 테스트
    test_files = [
        "KETI_SL_0000000001.avi",
        "KETI_SL_0000000002.avi",
        "KETI_SL_0000000003.avi"
    ]
    
    print("\n🧪 파일 경로 테스트:")
    for test_file in test_files:
        path = get_video_root_and_path(test_file)
        if path:
            print(f"   ✅ {test_file} -> {path}")
        else:
            print(f"   ❌ {test_file} -> 파일을 찾을 수 없습니다") 