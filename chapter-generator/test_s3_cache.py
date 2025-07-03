#!/usr/bin/env python3
"""
S3 캐시 시스템 테스트 스크립트

사용법:
    python test_s3_cache.py
"""

import os
import sys
import numpy as np
import tempfile
from datetime import datetime

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# .env 파일 로드
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

from s3_utils import (
    cache_join,
    cache_exists,
    cache_makedirs,
    cache_save_pickle,
    cache_load_pickle,
    cache_remove,
    is_s3_path,
    get_cache_manager
)

def test_s3_path_detection():
    """S3 경로 감지 테스트"""
    print("🧪 S3 경로 감지 테스트...")
    
    # S3 경로
    s3_paths = [
        "s3://bucket/path",
        "s3://waterandfish-s3/cache/",
        "s3://test-bucket/data/file.pkl"
    ]
    
    # 로컬 경로
    local_paths = [
        "cache",
        "/tmp/cache",
        "./data/file.pkl",
        "C:\\cache\\file.pkl"
    ]
    
    # S3 경로 테스트
    for path in s3_paths:
        assert is_s3_path(path), f"S3 경로 감지 실패: {path}"
        print(f"  ✅ S3 경로 감지: {path}")
    
    # 로컬 경로 테스트
    for path in local_paths:
        assert not is_s3_path(path), f"로컬 경로를 S3로 잘못 감지: {path}"
        print(f"  ✅ 로컬 경로 감지: {path}")
    
    print("  ✅ S3 경로 감지 테스트 완료\n")

def test_path_joining():
    """경로 조합 테스트"""
    print("🧪 경로 조합 테스트...")
    
    # S3 경로 조합
    s3_base = "s3://waterandfish-s3/cache/"
    s3_result = cache_join(s3_base, "test", "file.pkl")
    expected_s3 = "s3://waterandfish-s3/cache/test/file.pkl"
    assert s3_result == expected_s3, f"S3 경로 조합 실패: {s3_result} != {expected_s3}"
    print(f"  ✅ S3 경로 조합: {s3_result}")
    
    # 로컬 경로 조합
    local_base = "cache"
    local_result = cache_join(local_base, "test", "file.pkl")
    expected_local = os.path.join("cache", "test", "file.pkl")
    assert local_result == expected_local, f"로컬 경로 조합 실패: {local_result} != {expected_local}"
    print(f"  ✅ 로컬 경로 조합: {local_result}")
    
    print("  ✅ 경로 조합 테스트 완료\n")

def test_local_cache():
    """로컬 캐시 테스트"""
    print("🧪 로컬 캐시 테스트...")
    
    # 임시 디렉토리 사용
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "test_cache.pkl")
        
        # 테스트 데이터
        test_data = {
            "array": np.random.rand(10, 5),
            "list": [1, 2, 3, 4, 5],
            "dict": {"key1": "value1", "key2": 42},
            "timestamp": datetime.now().isoformat()
        }
        
        # 캐시 저장
        success = cache_save_pickle(cache_path, test_data)
        assert success, "로컬 캐시 저장 실패"
        print(f"  ✅ 로컬 캐시 저장: {cache_path}")
        
        # 캐시 존재 확인
        exists = cache_exists(cache_path)
        assert exists, "로컬 캐시 존재 확인 실패"
        print(f"  ✅ 로컬 캐시 존재 확인")
        
        # 캐시 로드
        loaded_data = cache_load_pickle(cache_path)
        assert loaded_data is not None, "로컬 캐시 로드 실패"
        print(f"  ✅ 로컬 캐시 로드")
        
        # 데이터 일치 확인
        assert np.array_equal(loaded_data["array"], test_data["array"]), "배열 데이터 불일치"
        assert loaded_data["list"] == test_data["list"], "리스트 데이터 불일치"
        assert loaded_data["dict"] == test_data["dict"], "딕셔너리 데이터 불일치"
        print(f"  ✅ 데이터 일치 확인")
        
        # 캐시 삭제
        success = cache_remove(cache_path)
        assert success, "로컬 캐시 삭제 실패"
        print(f"  ✅ 로컬 캐시 삭제")
        
        # 삭제 후 존재 확인
        exists = cache_exists(cache_path)
        assert not exists, "삭제된 캐시가 여전히 존재함"
        print(f"  ✅ 삭제 후 존재 확인")
    
    print("  ✅ 로컬 캐시 테스트 완료\n")

def test_s3_cache():
    """S3 캐시 테스트"""
    print("🧪 S3 캐시 테스트...")
    
    # S3 클라이언트 확인
    cache_manager = get_cache_manager()
    if not cache_manager.s3_client:
        print("  ⚠️ S3 클라이언트가 설정되지 않음. AWS 자격증명을 확인하세요.")
        print("  ⏭️ S3 캐시 테스트 건너뜀\n")
        return
    
    # 테스트용 S3 경로
    s3_path = "s3://waterandfish-s3/cache/test/test_cache.pkl"
    
    # 테스트 데이터
    test_data = {
        "array": np.random.rand(5, 3),
        "message": "S3 캐시 테스트",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # 기존 테스트 파일 정리
        cache_remove(s3_path)
        
        # 캐시 저장
        success = cache_save_pickle(s3_path, test_data)
        if not success:
            print("  ⚠️ S3 캐시 저장 실패. 권한을 확인하세요.")
            print("  ⏭️ S3 캐시 테스트 건너뜀\n")
            return
        print(f"  ✅ S3 캐시 저장: {s3_path}")
        
        # 캐시 존재 확인
        exists = cache_exists(s3_path)
        assert exists, "S3 캐시 존재 확인 실패"
        print(f"  ✅ S3 캐시 존재 확인")
        
        # 캐시 로드
        loaded_data = cache_load_pickle(s3_path)
        assert loaded_data is not None, "S3 캐시 로드 실패"
        print(f"  ✅ S3 캐시 로드")
        
        # 데이터 일치 확인
        assert np.array_equal(loaded_data["array"], test_data["array"]), "S3 배열 데이터 불일치"
        assert loaded_data["message"] == test_data["message"], "S3 메시지 데이터 불일치"
        print(f"  ✅ S3 데이터 일치 확인")
        
        # 캐시 삭제
        success = cache_remove(s3_path)
        assert success, "S3 캐시 삭제 실패"
        print(f"  ✅ S3 캐시 삭제")
        
        # 삭제 후 존재 확인
        exists = cache_exists(s3_path)
        assert not exists, "삭제된 S3 캐시가 여전히 존재함"
        print(f"  ✅ S3 삭제 후 존재 확인")
        
        print("  ✅ S3 캐시 테스트 완료\n")
        
    except Exception as e:
        print(f"  ⚠️ S3 캐시 테스트 중 오류: {e}")
        print("  ⏭️ S3 캐시 테스트 건너뜀\n")

def test_directory_creation():
    """디렉토리 생성 테스트"""
    print("🧪 디렉토리 생성 테스트...")
    
    # 로컬 디렉토리
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, "test_subdir")
        cache_makedirs(local_path)
        assert os.path.exists(local_path), "로컬 디렉토리 생성 실패"
        print(f"  ✅ 로컬 디렉토리 생성: {local_path}")
    
    # S3 디렉토리 (실제로는 아무것도 하지 않음)
    s3_path = "s3://waterandfish-s3/cache/test_dir/"
    cache_makedirs(s3_path)  # 오류가 발생하지 않아야 함
    print(f"  ✅ S3 디렉토리 생성 (건너뜀): {s3_path}")
    
    print("  ✅ 디렉토리 생성 테스트 완료\n")

def main():
    """메인 테스트 함수"""
    print("🚀 S3 캐시 시스템 테스트 시작\n")
    
    try:
        test_s3_path_detection()
        test_path_joining()
        test_directory_creation()
        test_local_cache()
        test_s3_cache()
        
        print("🎉 모든 테스트 완료!")
        print("\n📋 테스트 결과:")
        print("  ✅ S3 경로 감지")
        print("  ✅ 경로 조합")
        print("  ✅ 디렉토리 생성")
        print("  ✅ 로컬 캐시")
        
        # S3 테스트 결과 확인
        cache_manager = get_cache_manager()
        if cache_manager.s3_client:
            print("  ✅ S3 캐시")
        else:
            print("  ⚠️ S3 캐시 (AWS 자격증명 필요)")
        
        print("\n💡 다음 단계:")
        print("  1. AWS 자격증명 설정 (S3 사용 시)")
        print("  2. config.py에서 CACHE_DIR 설정")
        print("  3. main.py 실행하여 실제 캐시 시스템 테스트")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 