#!/usr/bin/env python3
"""
S3 호환 캐시 시스템 유틸리티
로컬 파일 시스템과 S3 버킷을 투명하게 처리할 수 있는 캐시 함수들을 제공합니다.
"""

import os
import pickle
import boto3
from urllib.parse import urlparse
from botocore.exceptions import ClientError, NoCredentialsError
import logging
from typing import Optional, Any, Dict

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

# 로깅 설정
logger = logging.getLogger(__name__)

class S3CacheManager:
    """S3와 로컬 파일 시스템을 통합하여 관리하는 캐시 매니저"""
    
    def __init__(self, aws_profile: str = None):
        """
        S3CacheManager 초기화
        
        Args:
            aws_profile: AWS 프로필 이름 (기본값: None - 환경변수 사용)
        """
        self.aws_profile = aws_profile
        self.s3_client = None
        self._setup_s3_client()
    
    def _setup_s3_client(self):
        """S3 클라이언트 설정"""
        # 환경변수 상태 확인
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        session_token = os.getenv('AWS_SESSION_TOKEN')
        
        print(f"🔍 AWS 자격증명 상태 확인:")
        print(f"  - AWS_ACCESS_KEY_ID: {'✅ 설정됨' if access_key else '❌ 미설정'}")
        print(f"  - AWS_SECRET_ACCESS_KEY: {'✅ 설정됨' if secret_key else '❌ 미설정'}")
        print(f"  - AWS_DEFAULT_REGION: {region}")
        print(f"  - AWS_SESSION_TOKEN: {'✅ 설정됨' if session_token else '미설정'}")
        
        try:
            if access_key and secret_key:
                # 환경변수에서 직접 자격증명 사용
                session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token,
                    region_name=region
                )
                print("✅ 환경변수에서 AWS 자격증명 사용")
                logger.info("Using AWS credentials from environment variables")
            else:
                # AWS 프로필 사용
                session = boto3.Session(profile_name=self.aws_profile)
                print(f"✅ AWS 프로필 사용: {self.aws_profile}")
                logger.info(f"Using AWS profile: {self.aws_profile}")
            
            self.s3_client = session.client('s3')
            print("✅ S3 클라이언트 초기화 성공")
            logger.info("S3 client initialized successfully")
            
        except NoCredentialsError as e:
            print(f"❌ AWS 자격증명 오류: {e}")
            print("💡 해결방법:")
            print("  1. .env 파일에 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY 설정")
            print("  2. 또는 AWS CLI로 프로필 설정: aws configure")
            logger.warning(f"S3 client initialization failed: {e}")
            self.s3_client = None
        except Exception as e:
            print(f"❌ S3 클라이언트 초기화 실패: {e}")
            logger.warning(f"S3 client initialization failed: {e}")
            self.s3_client = None
    
    def is_s3_path(self, path: str) -> bool:
        """경로가 S3 경로인지 확인"""
        return path.startswith('s3://')
    
    def parse_s3_path(self, s3_path: str) -> tuple:
        """S3 경로를 버킷과 키로 분리"""
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def join_path(self, base_path: str, *parts: str) -> str:
        """경로 조합 (S3/로컬 호환)"""
        if self.is_s3_path(base_path):
            # S3 경로인 경우 '/'로 조합
            path = base_path.rstrip('/')
            for part in parts:
                path += '/' + str(part).strip('/')
            return path
        else:
            # 로컬 경로인 경우 os.path.join 사용
            return os.path.join(base_path, *parts)
    
    def exists(self, path: str) -> bool:
        """파일/객체 존재 여부 확인"""
        if self.is_s3_path(path):
            if not self.s3_client:
                logger.warning("S3 client not available, assuming file doesn't exist")
                return False
            
            try:
                bucket, key = self.parse_s3_path(path)
                self.s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    return False
                else:
                    logger.error(f"S3 error checking {path}: {e}")
                    return False
            except Exception as e:
                logger.error(f"Error checking S3 path {path}: {e}")
                return False
        else:
            return os.path.exists(path)
    
    def makedirs(self, path: str, exist_ok: bool = True):
        """디렉토리 생성 (S3에서는 불필요하므로 건너뜀)"""
        if self.is_s3_path(path):
            # S3에서는 디렉토리 생성이 불필요
            logger.debug(f"S3 path detected, skipping makedirs for {path}")
            return
        else:
            os.makedirs(path, exist_ok=exist_ok)
    
    def save_pickle(self, path: str, data: Any) -> bool:
        """피클 데이터 저장"""
        try:
            if self.is_s3_path(path):
                if not self.s3_client:
                    logger.error("S3 client not available for saving")
                    return False
                
                # 메모리에서 피클 직렬화
                pickle_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                
                # S3에 업로드
                bucket, key = self.parse_s3_path(path)
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=pickle_data,
                    ContentType='application/octet-stream'
                )
                logger.info(f"Saved pickle to S3: {path}")
                return True
            else:
                # 로컬 파일로 저장
                # 임시 파일 방식으로 원자적 쓰기
                temp_path = path + ".tmp"
                with open(temp_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temp_path, path)
                logger.info(f"Saved pickle to local: {path}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving pickle to {path}: {e}")
            return False
    
    def load_pickle(self, path: str) -> Optional[Any]:
        """피클 데이터 로드"""
        try:
            if self.is_s3_path(path):
                if not self.s3_client:
                    logger.error("S3 client not available for loading")
                    return None
                
                # S3에서 다운로드
                bucket, key = self.parse_s3_path(path)
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                pickle_data = response['Body'].read()
                
                # 피클 역직렬화
                data = pickle.loads(pickle_data)
                logger.info(f"Loaded pickle from S3: {path}")
                return data
            else:
                # 로컬 파일에서 로드
                with open(path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Loaded pickle from local: {path}")
                return data
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.debug(f"S3 object not found: {path}")
            else:
                logger.error(f"S3 error loading {path}: {e}")
            return None
        except FileNotFoundError:
            logger.debug(f"Local file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading pickle from {path}: {e}")
            return None
    
    def remove(self, path: str) -> bool:
        """파일/객체 삭제"""
        try:
            if self.is_s3_path(path):
                if not self.s3_client:
                    logger.error("S3 client not available for deletion")
                    return False
                
                bucket, key = self.parse_s3_path(path)
                self.s3_client.delete_object(Bucket=bucket, Key=key)
                logger.info(f"Deleted S3 object: {path}")
                return True
            else:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Deleted local file: {path}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting {path}: {e}")
            return False

# 전역 캐시 매니저 인스턴스
_cache_manager = None

def get_cache_manager(aws_profile: str = None) -> S3CacheManager:
    """캐시 매니저 싱글톤 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = S3CacheManager(aws_profile)
    return _cache_manager

# 편의 함수들
def is_s3_path(path: str) -> bool:
    """경로가 S3 경로인지 확인"""
    return get_cache_manager().is_s3_path(path)

def cache_join(*parts: str) -> str:
    """캐시 경로 조합"""
    if not parts:
        return ""
    return get_cache_manager().join_path(parts[0], *parts[1:])

def cache_exists(path: str) -> bool:
    """캐시 파일 존재 여부 확인"""
    return get_cache_manager().exists(path)

def cache_makedirs(path: str, exist_ok: bool = True):
    """캐시 디렉토리 생성"""
    get_cache_manager().makedirs(path, exist_ok)

def cache_save_pickle(path: str, data: Any) -> bool:
    """캐시 피클 저장"""
    return get_cache_manager().save_pickle(path, data)

def cache_load_pickle(path: str) -> Optional[Any]:
    """캐시 피클 로드"""
    return get_cache_manager().load_pickle(path)

def cache_remove(path: str) -> bool:
    """캐시 파일 삭제"""
    return get_cache_manager().remove(path) 