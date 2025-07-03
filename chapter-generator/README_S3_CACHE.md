# S3 호환 캐시 시스템 사용 방법

## 개요
이 프로젝트는 기존 로컬 파일 시스템 캐시를 S3 버킷으로 확장하여 클라우드 환경에서도 캐시를 공유할 수 있도록 업데이트되었습니다.

## 주요 변경사항
- `s3_utils.py`: S3 호환 캐시 관리 유틸리티 추가
- `main.py`: 캐시 함수들을 S3 호환 버전으로 변경
- `config.py`: `CACHE_DIR`을 S3 경로로 변경 가능

## 필요한 의존성
```bash
pip install -r requirements.txt
```

주요 새로운 의존성:
- `boto3`: AWS S3 SDK
- `botocore`: AWS 코어 라이브러리
- `python-dotenv`: 환경 변수 관리 (선택사항)

## AWS 자격증명 설정

### 방법 1: 환경 변수 사용
```bash
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export AWS_DEFAULT_REGION=ap-northeast-2
```

### 방법 2: .env 파일 사용 (권장)
chapter-generator 디렉토리에 `.env` 파일 생성:
```
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_DEFAULT_REGION=ap-northeast-2
S3_BUCKET_NAME=waterandfish-s3
```

**주의**: `.env` 파일은 보안상 Git에 커밋하지 마세요!

### 방법 3: AWS CLI 프로필 사용
```bash
aws configure --profile default
```

## 캐시 디렉토리 설정

### S3 캐시 사용
`config.py`에서:
```python
CACHE_DIR = "s3://waterandfish-s3/cache/"
```

### 로컬 캐시 사용
`config.py`에서:
```python
CACHE_DIR = "cache"
```

## 사용법

### 1. 기본 사용법
기존 코드는 수정 없이 그대로 사용할 수 있습니다:
```bash
python main.py
```

### 2. 캐시 경로 확인
프로그램 실행 시 캐시가 S3 또는 로컬 파일 시스템 중 어느 것을 사용하는지 로그에 표시됩니다:
```
📂 라벨 데이터 캐시 로드 (S3): s3://waterandfish-s3/cache/label_data.pkl
📂 라벨 데이터 캐시 로드 (로컬): cache/label_data.pkl
```

## 기능 특징

### 투명한 호환성
- S3 경로와 로컬 경로를 자동으로 감지
- 동일한 함수로 S3와 로컬 파일 시스템 모두 지원

### 원자적 쓰기
- 로컬: 임시 파일 방식으로 안전한 쓰기
- S3: put_object가 원자적으로 동작

### 오류 처리
- 네트워크 오류 시 자동 재시도
- 손상된 캐시 파일 자동 삭제
- AWS 자격증명 없을 경우 graceful fallback

## 트러블슈팅

### 1. AWS 자격증명 오류
```
S3 client initialization failed: Unable to locate credentials
```
**해결방법**: AWS 자격증명을 올바르게 설정했는지 확인

### 2. S3 버킷 접근 권한 오류
```
S3 error checking s3://bucket/path: Access Denied
```
**해결방법**: 버킷 정책에서 GetObject, PutObject, DeleteObject 권한 확인

### 3. 네트워크 연결 문제
```
S3 error loading path: Connection timeout
```
**해결방법**: 네트워크 연결 확인 또는 로컬 캐시로 fallback

## 성능 비교

| 항목 | 로컬 캐시 | S3 캐시 |
|------|----------|---------|
| 읽기 속도 | 매우 빠름 | 중간 |
| 쓰기 속도 | 매우 빠름 | 중간 |
| 공유 가능성 | 불가능 | 가능 |
| 저장 용량 | 로컬 디스크 제한 | 무제한 |
| 비용 | 없음 | S3 요금 |

## 권장 사항

1. **개발 환경**: 로컬 캐시 사용 (빠른 속도)
2. **프로덕션 환경**: S3 캐시 사용 (공유 가능)
3. **하이브리드**: 작은 캐시는 로컬, 큰 캐시는 S3

## .env 파일 설정 예시

chapter-generator 디렉토리에 `.env` 파일을 생성하세요:

```bash
# .env 파일 내용
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=ap-northeast-2
S3_BUCKET_NAME=waterandfish-s3
```

## 예시 워크플로우

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. .env 파일 생성 및 AWS 자격증명 설정
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=your-actual-access-key
AWS_SECRET_ACCESS_KEY=your-actual-secret-key
AWS_DEFAULT_REGION=ap-northeast-2
S3_BUCKET_NAME=waterandfish-s3
EOF

# 3. .env 파일을 .gitignore에 추가 (보안)
echo ".env" >> .gitignore

# 4. S3 캐시 경로 설정 확인 (config.py)
# CACHE_DIR = "s3://waterandfish-s3/cache/"

# 5. 프로그램 실행
python main.py

# 6. 캐시 확인 (S3 콘솔 또는 AWS CLI)
aws s3 ls s3://waterandfish-s3/cache/
```

## 자격증명 테스트

프로그램 실행 전에 AWS 자격증명이 올바르게 설정되었는지 테스트:

```bash
python test_s3_cache.py
``` 