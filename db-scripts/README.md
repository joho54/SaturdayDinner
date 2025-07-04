# MongoDB Model Mapper

MongoDB의 Lessons 컬렉션에 있는 레코드들의 `sign_text` 필드와 매핑 파일을 기반으로 `model_data_url` 필드를 업데이트하는 스크립트입니다.

## 설치

1. 필요한 Python 패키지 설치:
```bash
pip install -r requirements.txt
```

2. `.env` 파일 생성:
```bash
# .env 파일을 프로젝트 루트에 생성하고 다음 내용을 추가:
MONGODB_URL=mongodb://localhost:27017/your_database_name
```

MongoDB Atlas를 사용하는 경우:
```bash
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/database_name
```

## 사용법

```bash
python ModelMapper.py
```

## 기능

- `label_model_mapping.json` 파일에서 sign_text와 모델 파일 경로 매핑을 읽습니다
- MongoDB의 Lessons 컬렉션에서 모든 레코드를 가져옵니다
- 각 레코드의 `sign_text`와 매핑 파일의 키를 비교합니다
- 일치하는 경우 해당 레코드의 `model_data_url` 필드를 업데이트합니다
- 업데이트 결과를 콘솔에 출력합니다

## 예시

실행 전:
```json
{
  "_id": ObjectId("6862685dcba901ab2b744fd7"),
  "sign_text": "병원",
  "model_data_url": null
}
```

실행 후:
```json
{
  "_id": ObjectId("6862685dcba901ab2b744fd7"),
  "sign_text": "병원",
  "model_data_url": "models/sign_language_model_20250704_031902.keras"
}
```

## 주의사항

- 스크립트 실행 전에 데이터베이스를 백업하는 것을 권장합니다
- MongoDB URL이 올바른지 확인하세요
- `label_model_mapping.json` 파일이 프로젝트 루트에 있는지 확인하세요 