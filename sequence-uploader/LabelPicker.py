#!/usr/bin/env python3
"""
LabelPicker.py - 라벨 CSV 파일에서 유니크한 값을 추출하여 spec.json 형식으로 변환하는 스크립트

사용법:
    python LabelPicker.py [--input INPUT_CSV] [--output OUTPUT_JSON] [--chapter CHAPTER_NAME]

예시:
    python LabelPicker.py --input labels.csv --output spec_labels.json --chapter chapter_01
"""

import csv
import json
import argparse
import sys
from pathlib import Path


def extract_unique_labels(csv_file_path, label_column='한국어'):
    """
    CSV 파일에서 유니크한 라벨을 추출합니다.
    
    Args:
        csv_file_path (str): CSV 파일 경로
        label_column (str): 라벨이 있는 컬럼명 (기본값: '한국어')
    
    Returns:
        list: 정렬된 유니크 라벨 리스트
    """
    unique_labels = set()
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            # 헤더 확인
            if label_column not in csv_reader.fieldnames:
                available_columns = ', '.join(csv_reader.fieldnames)
                raise ValueError(f"컬럼 '{label_column}'을 찾을 수 없습니다. 사용 가능한 컬럼: {available_columns}")
            
            # 유니크 라벨 수집
            for row in csv_reader:
                label = row[label_column].strip()
                if label:  # 빈 문자열이 아닌 경우만 추가
                    unique_labels.add(label)
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_file_path}")
    except UnicodeDecodeError:
        # UTF-8로 읽기 실패시 다른 인코딩 시도
        try:
            with open(csv_file_path, 'r', encoding='cp949') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    label = row[label_column].strip()
                    if label:
                        unique_labels.add(label)
        except UnicodeDecodeError:
            raise UnicodeDecodeError("파일 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949를 시도했습니다.")
    
    # 알파벳 순으로 정렬하여 반환
    return sorted(list(unique_labels))


def create_label_dict(unique_labels, include_none=True):
    """
    유니크 라벨 리스트를 라벨 딕셔너리로 변환합니다.
    
    Args:
        unique_labels (list): 유니크 라벨 리스트
        include_none (bool): 'None' 라벨을 추가할지 여부
    
    Returns:
        dict: 라벨을 키로, 정수를 값으로 하는 딕셔너리
    """
    label_dict = {}
    
    # 라벨을 0부터 시작하는 정수에 매핑
    for idx, label in enumerate(unique_labels):
        label_dict[label] = idx
    
    # None 라벨 추가 (선택사항)
    if include_none and "None" not in label_dict:
        label_dict["None"] = len(label_dict)
    
    return label_dict


def create_spec_json(label_dict, chapter_name="chapter_01"):
    """
    spec.json 형식의 딕셔너리를 생성합니다.
    
    Args:
        label_dict (dict): 라벨 딕셔너리
        chapter_name (str): 챕터 이름
    
    Returns:
        dict: spec.json 형식의 딕셔너리
    """
    return {
        "chapter_name": chapter_name,
        "label_dict": label_dict
    }


def save_json(data, output_file_path):
    """
    데이터를 JSON 파일로 저장합니다.
    
    Args:
        data (dict): 저장할 데이터
        output_file_path (str): 출력 파일 경로
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"✅ JSON 파일이 성공적으로 생성되었습니다: {output_file_path}")
    except Exception as e:
        raise Exception(f"JSON 파일 저장 중 오류가 발생했습니다: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CSV 파일에서 유니크한 라벨을 추출하여 spec.json 형식으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python LabelPicker.py
  python LabelPicker.py --input labels.csv --output spec_labels.json
  python LabelPicker.py --input data/labels.csv --output specs/spec_new.json --chapter chapter_100
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='labels.csv',
        help='입력 CSV 파일 경로 (기본값: labels.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='spec_labels.json',
        help='출력 JSON 파일 경로 (기본값: spec_labels.json)'
    )
    
    parser.add_argument(
        '--chapter', '-c',
        default='chapter_01',
        help='챕터 이름 (기본값: chapter_01)'
    )
    
    parser.add_argument(
        '--no-none',
        action='store_true',
        help='None 라벨을 포함하지 않음'
    )
    
    parser.add_argument(
        '--column',
        default='한국어',
        help='라벨을 추출할 컬럼명 (기본값: 한국어)'
    )
    
    args = parser.parse_args()
    
    try:
        # 1. 입력 파일 존재 확인
        if not Path(args.input).exists():
            print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {args.input}")
            sys.exit(1)
        
        print(f"📂 CSV 파일 읽는 중: {args.input}")
        
        # 2. 유니크 라벨 추출
        unique_labels = extract_unique_labels(args.input, args.column)
        print(f"📊 총 {len(unique_labels)}개의 유니크한 라벨을 발견했습니다.")
        
        # 3. 라벨 딕셔너리 생성
        label_dict = create_label_dict(unique_labels, include_none=not args.no_none)
        print(f"🏷️  라벨 딕셔너리 생성 완료 (총 {len(label_dict)}개 항목)")
        
        # 4. spec.json 형식 데이터 생성
        spec_data = create_spec_json(label_dict, args.chapter)
        
        # 5. JSON 파일로 저장
        save_json(spec_data, args.output)
        
        # 6. 결과 요약 출력
        print(f"\n📋 처리 결과 요약:")
        print(f"   입력 파일: {args.input}")
        print(f"   출력 파일: {args.output}")
        print(f"   챕터명: {args.chapter}")
        print(f"   라벨 개수: {len(label_dict)}")
        print(f"\n🏷️  라벨 매핑 (처음 10개):")
        for i, (label, idx) in enumerate(list(label_dict.items())[:10]):
            print(f"   {label}: {idx}")
        if len(label_dict) > 10:
            print(f"   ... 및 {len(label_dict) - 10}개 더")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
