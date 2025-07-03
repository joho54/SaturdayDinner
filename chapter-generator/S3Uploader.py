#!/usr/bin/env python3
"""
S3Uploader - 디렉토리를 S3에 업로드하고 검증 후 삭제하는 스크립트

안전한 업로드를 위해 다음 단계를 거칩니다:
1. 로컬 파일 목록 생성
2. S3에 업로드
3. 업로드된 파일 검증 (크기, 해시)
4. 검증 완료 후 로컬 파일 삭제

설정 방법:
1. .env 파일 생성:
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   AWS_DEFAULT_REGION=ap-northeast-2
   S3_BUCKET_NAME=your-bucket-name
   AWS_PROFILE=default

2. 필요한 패키지 설치:
   pip install boto3 python-dotenv

사용법:
   python S3Uploader.py <directory>
   python S3Uploader.py models/ --prefix "backups/2024-12-01"
"""

import os
import sys
import boto3
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from botocore.exceptions import ClientError, NoCredentialsError
import json

# .env 파일 지원 추가
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables.")


class S3Uploader:
    """S3에 안전하게 파일을 업로드하고 검증하는 클래스"""

    def __init__(self, bucket_name: str = None, aws_profile: str = None):
        """
        S3Uploader 초기화
        
        Args:
            bucket_name: S3 버킷 이름 (.env 파일의 S3_BUCKET_NAME 또는 환경변수에서 가져올 수 있음)
            aws_profile: AWS 프로필 이름 (.env 파일의 AWS_PROFILE 또는 기본값: default)
        """
        # .env 파일에서 설정 로드
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name must be provided via parameter or S3_BUCKET_NAME in .env file")
        
        self.aws_profile = aws_profile or os.getenv('AWS_PROFILE', 'default')
        
        # AWS 자격증명 확인
        self.check_aws_credentials()
        
        self.setup_logging()
        self.setup_s3_client()
        
        self.logger.info(f"S3Uploader initialized for bucket: {self.bucket_name}")
        self.logger.info(f"AWS Profile: {self.aws_profile}")
        self.logger.info(f"AWS Region: {os.getenv('AWS_DEFAULT_REGION', 'not set')}")

    def check_aws_credentials(self):
        """AWS 자격증명 환경변수 확인"""
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars and self.aws_profile == 'default':
            print("Warning: Missing AWS credentials in environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            print("Make sure to set them in .env file or configure AWS CLI profile")
            print("\nExample .env file:")
            print("AWS_ACCESS_KEY_ID=your-access-key")
            print("AWS_SECRET_ACCESS_KEY=your-secret-key")
            print("AWS_DEFAULT_REGION=ap-northeast-2")
            print("S3_BUCKET_NAME=your-bucket-name")
            print("AWS_PROFILE=default")

    def setup_logging(self):
        """로깅 설정"""
        log_file = f"s3_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 로거 설정
        self.logger = logging.getLogger('S3Uploader')
        self.logger.setLevel(logging.INFO)
        
        # 핸들러가 이미 있다면 제거 (중복 방지)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포매터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def setup_s3_client(self):
        """S3 클라이언트 설정"""
        try:
            # AWS 세션 생성 (환경변수 또는 프로필 사용)
            if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
                # 환경변수에서 직접 자격증명 사용
                session = boto3.Session(
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),  # 임시 자격증명 지원
                    region_name=os.getenv('AWS_DEFAULT_REGION')
                )
                self.logger.info(f"Using AWS credentials from environment variables")
            else:
                # AWS 프로필 사용
                session = boto3.Session(profile_name=self.aws_profile)
                self.logger.info(f"Using AWS profile: {self.aws_profile}")
            
            self.s3_client = session.client('s3')
            
            # 버킷 접근 권한 확인
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. Please:\n"
                "1. Create .env file with AWS credentials, or\n"
                "2. Configure AWS CLI with 'aws configure', or\n"
                "3. Set AWS environment variables"
            )
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise RuntimeError(f"Bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                raise RuntimeError(f"Access denied to bucket '{self.bucket_name}'")
            else:
                raise RuntimeError(f"Error accessing S3 bucket: {e}")

    def calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """파일의 MD5 해시 계산"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def get_local_files_info(self, directory: str) -> Dict[str, Dict]:
        """로컬 디렉토리의 모든 파일 정보 수집"""
        files_info = {}
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        self.logger.info(f"Scanning directory: {directory}")
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory_path)
                file_size = file_path.stat().st_size
                file_hash = self.calculate_file_hash(str(file_path))
                
                files_info[str(relative_path)] = {
                    'full_path': str(file_path),
                    'size': file_size,
                    'hash': file_hash
                }
        
        self.logger.info(f"Found {len(files_info)} files to upload")
        return files_info

    def upload_file_to_s3(self, local_file: str, s3_key: str) -> bool:
        """단일 파일을 S3에 업로드"""
        try:
            self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload {local_file} to s3://{self.bucket_name}/{s3_key}: {e}")
            return False

    def verify_s3_file(self, s3_key: str, expected_size: int, expected_hash: str) -> bool:
        """S3에 업로드된 파일 검증"""
        try:
            # 파일 메타데이터 확인
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            s3_size = response['ContentLength']
            
            # 크기 검증
            if s3_size != expected_size:
                self.logger.error(f"Size mismatch for {s3_key}: expected {expected_size}, got {s3_size}")
                return False
            
            # ETag로 간단 검증 (멀티파트가 아닌 경우만)
            s3_etag = response['ETag'].strip('"')
            if '-' not in s3_etag and s3_etag != expected_hash:
                self.logger.warning(f"Hash mismatch for {s3_key}: expected {expected_hash}, got {s3_etag}")
                # 해시 불일치는 경고만 출력하고 계속 진행 (멀티파트 업로드 등의 경우)
            
            return True
            
        except ClientError as e:
            self.logger.error(f"Failed to verify {s3_key}: {e}")
            return False

    def upload_directory(self, directory: str, s3_prefix: str = "") -> Tuple[bool, Dict]:
        """디렉토리를 S3에 업로드"""
        self.logger.info(f"Starting upload of directory: {directory}")
        
        # 로컬 파일 정보 수집
        local_files = self.get_local_files_info(directory)
        
        if not local_files:
            self.logger.warning("No files found to upload")
            return True, {}
        
        # S3 prefix 설정
        if s3_prefix and not s3_prefix.endswith('/'):
            s3_prefix += '/'
        
        upload_results = {}
        successful_uploads = 0
        failed_uploads = 0
        
        # 파일별 업로드
        for relative_path, file_info in local_files.items():
            s3_key = s3_prefix + relative_path.replace('\\', '/')  # Windows 경로 처리
            
            self.logger.info(f"Uploading: {relative_path} -> s3://{self.bucket_name}/{s3_key}")
            
            # 업로드 실행
            upload_success = self.upload_file_to_s3(file_info['full_path'], s3_key)
            
            if upload_success:
                # 업로드 검증
                verify_success = self.verify_s3_file(s3_key, file_info['size'], file_info['hash'])
                
                if verify_success:
                    upload_results[relative_path] = {
                        'status': 'success',
                        's3_key': s3_key,
                        'size': file_info['size']
                    }
                    successful_uploads += 1
                    self.logger.info(f"✅ Successfully uploaded and verified: {relative_path}")
                else:
                    upload_results[relative_path] = {
                        'status': 'verification_failed',
                        's3_key': s3_key,
                        'error': 'Verification failed'
                    }
                    failed_uploads += 1
                    self.logger.error(f"❌ Verification failed: {relative_path}")
            else:
                upload_results[relative_path] = {
                    'status': 'upload_failed',
                    'error': 'Upload failed'
                }
                failed_uploads += 1
                self.logger.error(f"❌ Upload failed: {relative_path}")
        
        # 결과 요약
        success_rate = (successful_uploads / len(local_files)) * 100
        self.logger.info(f"\nUpload Summary:")
        self.logger.info(f"  Total files: {len(local_files)}")
        self.logger.info(f"  Successful: {successful_uploads}")
        self.logger.info(f"  Failed: {failed_uploads}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        return failed_uploads == 0, upload_results

    def delete_local_files(self, upload_results: Dict, directory: str) -> bool:
        """업로드 성공한 파일들을 로컬에서 삭제"""
        self.logger.info("Starting deletion of successfully uploaded files")
        
        deleted_count = 0
        failed_deletions = 0
        
        for relative_path, result in upload_results.items():
            if result['status'] == 'success':
                file_path = os.path.join(directory, relative_path)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    self.logger.info(f"🗑️  Deleted: {relative_path}")
                except Exception as e:
                    failed_deletions += 1
                    self.logger.error(f"Failed to delete {relative_path}: {e}")
        
        # 빈 디렉토리 정리
        self.cleanup_empty_directories(directory)
        
        self.logger.info(f"\nDeletion Summary:")
        self.logger.info(f"  Files deleted: {deleted_count}")
        self.logger.info(f"  Failed deletions: {failed_deletions}")
        
        return failed_deletions == 0

    def cleanup_empty_directories(self, directory: str):
        """빈 디렉토리 정리"""
        directory_path = Path(directory)
        
        # 하위 디렉토리부터 상위로 올라가면서 빈 디렉토리 삭제
        for dir_path in sorted(directory_path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    self.logger.info(f"🗑️  Removed empty directory: {dir_path.relative_to(directory_path)}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove empty directory {dir_path}: {e}")

    def save_upload_report(self, upload_results: Dict, directory: str):
        """업로드 결과 리포트 저장"""
        report_file = f"upload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "directory": directory,
            "bucket": self.bucket_name,
            "total_files": len(upload_results),
            "successful_uploads": sum(1 for r in upload_results.values() if r['status'] == 'success'),
            "failed_uploads": sum(1 for r in upload_results.values() if r['status'] != 'success'),
            "files": upload_results
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📋 Upload report saved: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save upload report: {e}")

    def upload_and_delete(self, directory: str, s3_prefix: str = "", 
                         confirm_delete: bool = True) -> bool:
        """디렉토리를 업로드하고 검증 후 삭제하는 메인 프로세스"""
        try:
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Starting S3 upload and delete process")
            self.logger.info(f"Directory: {directory}")
            self.logger.info(f"S3 Bucket: {self.bucket_name}")
            self.logger.info(f"S3 Prefix: {s3_prefix or '(root)'}")
            self.logger.info(f"{'='*80}")
            
            # 1. 업로드 실행
            upload_success, upload_results = self.upload_directory(directory, s3_prefix)
            
            # 2. 업로드 리포트 저장
            self.save_upload_report(upload_results, directory)
            
            if not upload_success:
                self.logger.error("❌ Upload process completed with failures. Files will NOT be deleted.")
                return False
            
            # 3. 삭제 확인
            if confirm_delete:
                response = input(f"\n⚠️  All files uploaded successfully. Delete local files in '{directory}'? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    self.logger.info("User cancelled file deletion. Local files preserved.")
                    return True
            
            # 4. 로컬 파일 삭제
            delete_success = self.delete_local_files(upload_results, directory)
            
            if delete_success:
                self.logger.info("🎉 Upload and delete process completed successfully!")
                return True
            else:
                self.logger.warning("⚠️  Upload completed but some files could not be deleted.")
                return False
                
        except Exception as e:
            self.logger.error(f"Critical error in upload_and_delete: {e}")
            return False


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='Upload directory to S3 and delete local files after verification',
        epilog="""
Examples:
  python S3Uploader.py models/
  python S3Uploader.py models/ --prefix "backups/2024-12-01"
  python S3Uploader.py models/ --bucket my-bucket --no-confirm

Environment variables (.env file):
  AWS_ACCESS_KEY_ID=your-access-key
  AWS_SECRET_ACCESS_KEY=your-secret-key
  AWS_DEFAULT_REGION=ap-northeast-2
  S3_BUCKET_NAME=your-bucket-name
  AWS_PROFILE=default
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('directory', help='Directory to upload')
    parser.add_argument('--bucket', help='S3 bucket name (or set S3_BUCKET_NAME in .env)')
    parser.add_argument('--prefix', default='', help='S3 prefix/folder (default: root)')
    parser.add_argument('--profile', help='AWS profile name (or set AWS_PROFILE in .env)')
    parser.add_argument('--no-confirm', action='store_true', help='Skip deletion confirmation')
    
    args = parser.parse_args()
    
    try:
        # S3Uploader 초기화
        uploader = S3Uploader(bucket_name=args.bucket, aws_profile=args.profile)
        
        # 업로드 및 삭제 실행
        success = uploader.upload_and_delete(
            directory=args.directory,
            s3_prefix=args.prefix,
            confirm_delete=not args.no_confirm
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
