#!/usr/bin/env python3
"""
독립적인 Info 파일 수정 스크립트
info 디렉토리의 파일들에서 model_path를 실제 모델 파일명과 일치하도록 수정합니다.
"""

import os
import json
import re
from pathlib import Path


def fix_info_files(base_dir=None):
    """info 파일들의 model_path를 실제 모델 파일명과 일치하도록 수정"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    models_dir = os.path.join(base_dir, "models")
    info_dir = os.path.join(base_dir, "info")
    
    print(f"🔧 Fixing info files...")
    print(f"   Models directory: {models_dir}")
    print(f"   Info directory: {info_dir}")
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    if not os.path.exists(info_dir):
        print(f"❌ Info directory not found: {info_dir}")
        return False
    
    # 실제 모델 파일들 목록
    actual_model_files = {f for f in os.listdir(models_dir) if f.endswith('.keras')}
    print(f"📁 Found {len(actual_model_files)} model files:")
    for f in sorted(actual_model_files):
        print(f"   - {f}")
    
    # info 파일들 목록
    info_files = [f for f in os.listdir(info_dir) 
                  if f.startswith('model-info-') and f.endswith('.json')]
    print(f"📄 Found {len(info_files)} info files:")
    for f in sorted(info_files):
        print(f"   - {f}")
    
    fixed_count = 0
    error_count = 0
    
    # info 파일들 처리
    for info_file in info_files:
        info_path = os.path.join(info_dir, info_file)
        
        try:
            # info 파일 읽기
            with open(info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            current_model_path = info_data.get('model_path', '')
            if not current_model_path:
                print(f"⚠️  No model_path in {info_file}")
                continue
            
            # 현재 경로에서 파일명만 추출
            current_model_filename = os.path.basename(current_model_path)
            
            # 실제 존재하는 모델 파일과 매칭 시도
            matched_file = find_matching_model(current_model_filename, actual_model_files)
            
            if matched_file:
                new_model_path = f"models/{matched_file}"
                if current_model_path != new_model_path:
                    info_data['model_path'] = new_model_path
                    
                    # 파일 업데이트
                    with open(info_path, 'w', encoding='utf-8') as f:
                        json.dump(info_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"✅ Fixed {info_file}:")
                    print(f"   Old: {current_model_path}")
                    print(f"   New: {new_model_path}")
                    fixed_count += 1
                else:
                    print(f"✓  {info_file} already correct")
            else:
                print(f"❌ No matching model file found for {info_file}")
                print(f"   Looking for: {current_model_filename}")
                error_count += 1
        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {info_file}: {e}")
            error_count += 1
        except Exception as e:
            print(f"❌ Error processing {info_file}: {e}")
            error_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Fixed: {fixed_count}")
    print(f"   Errors: {error_count}")
    print(f"   Total processed: {len(info_files)}")
    
    return error_count == 0


def find_matching_model(original_filename, actual_files):
    """원본 파일명과 매칭되는 실제 모델 파일 찾기"""
    # 정확한 이름 매칭
    if original_filename in actual_files:
        return original_filename
    
    # 타임스탬프 기반 매칭
    for actual_file in actual_files:
        if is_matching_model_file(original_filename, actual_file):
            return actual_file
    
    return None


def is_matching_model_file(original_filename, actual_filename):
    """두 모델 파일명이 같은 모델인지 확인 (타임스탬프 기준)"""
    try:
        # 여러 타임스탬프 패턴 시도
        patterns = [
            r'(\d{8}_\d{6})',  # YYYYMMDD_HHMMSS
            r'(\d{8})',        # YYYYMMDD만
        ]
        
        for pattern in patterns:
            original_match = re.search(pattern, original_filename)
            actual_match = re.search(pattern, actual_filename)
            
            if original_match and actual_match:
                original_timestamp = original_match.group(1)
                actual_timestamp = actual_match.group(1)
                
                # 날짜 부분이 포함되어 있는지 확인
                if len(original_timestamp) >= 8 and len(actual_timestamp) >= 8:
                    # 최소한 날짜 부분(YYYYMMDD)이 같으면 매칭
                    original_date = original_timestamp[:8]
                    actual_date = actual_timestamp[:8]
                    if original_date == actual_date:
                        return True
        
        # 파일명에서 공통 부분 비교 (sign_language_model_부분)
        original_base = original_filename.replace('.keras', '').split('_')
        actual_base = actual_filename.replace('.keras', '').split('_')
        
        # sign_language_model 부분이 같고, 타임스탬프가 포함되어 있는지 확인
        if (len(original_base) >= 3 and len(actual_base) >= 3 and
            original_base[0] == actual_base[0] and  # sign
            original_base[1] == actual_base[1] and  # language  
            original_base[2] == actual_base[2]):    # model
            
            # 날짜가 포함된 부분 찾기
            for orig_part in original_base[3:]:
                if re.match(r'\d{8}', orig_part):  # 날짜 형식
                    orig_date = orig_part[:8]
                    for actual_part in actual_base[3:]:
                        if re.match(r'\d{8}', actual_part):
                            actual_date = actual_part[:8]
                            if orig_date == actual_date:
                                return True
        
        return False
        
    except Exception:
        return False


def main():
    """메인 실행 함수"""
    print("🔧 Info Files Model Path Fixer")
    print("=" * 50)
    
    try:
        success = fix_info_files()
        if success:
            print("\n🎉 All info files fixed successfully!")
        else:
            print("\n⚠️  Some files had errors")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 