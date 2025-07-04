import json
import os
import glob

def create_label_model_mapping():
    """
    info 디렉토리의 모든 model-info JSON 파일들을 읽어서
    각 labels 요소와 model_path를 매핑하는 JSON 파일을 생성합니다.
    """
    mapper = {}
    
    # info 디렉토리의 모든 model-info 파일들 찾기
    info_files = glob.glob('info/model-info-*.json')
    
    print(f"총 {len(info_files)}개의 model-info 파일을 발견했습니다.")
    
    for file_path in info_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info_path = os.path.basename(file_path)
            labels = data['labels']
            
            print(f"처리 중: {file_path} - {len(labels)}개 라벨")
            
            # 각 라벨을 모델 경로와 매핑
            for label in labels:
                if label in mapper:
                    print(f"  중복 라벨 발견: '{label}' - 기존: {mapper[label]}, 새로운: {info_path}")
                mapper[label] = info_path
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생 {file_path}: {e}")
            continue
    
    # 결과를 JSON 형식으로 저장
    result = {"mapper": mapper}
    
    with open('label_model_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n매핑 파일 생성 완료!")
    print(f"- 총 {len(mapper)}개의 라벨이 매핑되었습니다.")
    print(f"- 생성된 파일: label_model_mapping.json")
    
    # 매핑 결과 미리보기 (처음 5개)
    print("\n매핑 결과 미리보기:")
    for i, (label, info_path) in enumerate(mapper.items()):
        if i >= 5:
            print("...")
            break
        print(f"  '{label}' -> '{info_path}'")

if __name__ == "__main__":
    create_label_model_mapping()