#!/usr/bin/env python3
"""
수어 인식 모델 개선 프로젝트 통합 실행 스크립트

이 스크립트는 리포트에서 제안한 개선 방안들을 적용한 모델을
학습하고 성능을 비교하는 통합 도구입니다.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """필요한 의존성 패키지들을 확인합니다."""
    # 패키지 이름과 import 이름의 매핑
    package_mapping = {
        'opencv-python': 'cv2',
        'mediapipe': 'mediapipe',
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'tqdm': 'tqdm',
        'Pillow': 'PIL',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    installed_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
            installed_packages.append(package_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    print(f"\n📊 설치 상태: {len(installed_packages)}/{len(package_mapping)} 패키지 설치됨")
    
    if missing_packages:
        print(f"\n❌ 다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치해주세요:")
        print("pip install -r requirements_improved.txt")
        print("\n또는 개별 설치:")
        for package in missing_packages:
            print(f"pip install {package}")
        return False
    
    print("✅ 모든 의존성 패키지가 설치되어 있습니다.")
    return True

def install_dependencies():
    """필요한 의존성 패키지들을 설치합니다."""
    print("📦 의존성 패키지 설치를 시작합니다...")
    
    try:
        # requirements 파일이 있는지 확인
        if os.path.exists('requirements_improved.txt'):
            result = subprocess.run(['pip', 'install', '-r', 'requirements_improved.txt'], 
                                  capture_output=True, text=True, check=True)
            print("✅ requirements_improved.txt에서 패키지 설치 완료")
            return True
        else:
            # 개별 패키지 설치
            packages = [
                'opencv-python>=4.5.0',
                'mediapipe>=0.8.0', 
                'tensorflow>=2.8.0',
                'numpy>=1.21.0',
                'scikit-learn>=1.0.0',
                'tqdm>=4.62.0',
                'Pillow>=8.3.0',
                'scipy>=1.7.0',
                'matplotlib>=3.5.0',
                'seaborn>=0.11.0'
            ]
            
            for package in packages:
                print(f"설치 중: {package}")
                subprocess.run(['pip', 'install', package], check=True)
            
            print("✅ 모든 패키지 설치 완료")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 중 오류 발생: {e}")
        return False

def run_command(command, description):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n🚀 {description}")
    print(f"실행 명령어: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 성공적으로 완료되었습니다.")
        if result.stdout:
            print("출력:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        if e.stdout:
            print("표준 출력:")
            print(e.stdout)
        if e.stderr:
            print("오류 출력:")
            print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='수어 인식 모델 개선 프로젝트')
    parser.add_argument('--step', choices=['check', 'install', 'train', 'demo', 'compare', 'all'], 
                       default='all', help='실행할 단계')
    parser.add_argument('--data-path', type=str, 
                       default="/Volumes/Sub_Storage/수어 데이터셋/0001~3000(영상)",
                       help='비디오 데이터 경로')
    parser.add_argument('--auto-install', action='store_true',
                       help='의존성 패키지 자동 설치')
    
    args = parser.parse_args()
    
    print("🎯 수어 인식 모델 개선 프로젝트")
    print("=" * 50)
    
    # 1. 의존성 확인
    if args.step in ['check', 'all']:
        print("🔍 의존성 패키지 확인 중...")
        if not check_dependencies():
            if args.auto_install:
                print("\n🔄 자동 설치를 시작합니다...")
                if install_dependencies():
                    print("✅ 의존성 설치 완료. 다시 확인합니다...")
                    if not check_dependencies():
                        print("❌ 의존성 설치 후에도 문제가 있습니다.")
                        return
                else:
                    print("❌ 자동 설치에 실패했습니다.")
                    return
            else:
                print("\n💡 --auto-install 옵션을 사용하여 자동으로 설치할 수 있습니다.")
                return
    
    # 2. 의존성 설치 (별도 단계)
    if args.step == 'install':
        if install_dependencies():
            print("✅ 의존성 설치가 완료되었습니다.")
        else:
            print("❌ 의존성 설치에 실패했습니다.")
        return
    
    # 3. 데이터 경로 확인
    if not os.path.exists(args.data_path):
        print(f"⚠️ 데이터 경로가 존재하지 않습니다: {args.data_path}")
        print("--data-path 옵션으로 올바른 경로를 지정해주세요.")
        return
    
    # 4. 개선된 모델 학습
    if args.step in ['train', 'all']:
        print(f"\n📁 데이터 경로: {args.data_path}")
        
        # improved_main.py의 데이터 경로 수정
        with open('improved_main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 데이터 경로 업데이트
        updated_content = content.replace(
            'VIDEO_ROOT = "/Volumes/Sub_Storage/수어 데이터셋/0001~3000(영상)"',
            f'VIDEO_ROOT = "{args.data_path}"'
        )
        
        with open('improved_main.py', 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        if not run_command('python improved_main.py', '개선된 Transformer 모델 학습'):
            return
    
    # 5. 실시간 데모 실행 (선택사항)
    if args.step in ['demo', 'all']:
        print("\n🎥 실시간 데모를 실행하시겠습니까? (y/n): ", end='')
        response = input().lower().strip()
        
        if response in ['y', 'yes', '예']:
            if not run_command('python improved_realtime_demo.py', '개선된 실시간 데모'):
                print("⚠️ 실시간 데모 실행 중 오류가 발생했습니다.")
    
    # 6. 모델 성능 비교
    if args.step in ['compare', 'all']:
        print("\n📊 모델 성능 비교를 실행하시겠습니까? (y/n): ", end='')
        response = input().lower().strip()
        
        if response in ['y', 'yes', '예']:
            if not run_command('python model_comparison.py', '모델 성능 비교'):
                print("⚠️ 모델 비교 실행 중 오류가 발생했습니다.")
    
    print("\n🎉 모든 작업이 완료되었습니다!")
    print("\n📋 다음 파일들을 확인해보세요:")
    print("   - improved_transformer_model.keras: 개선된 모델")
    print("   - improved_preprocessed_data.npz: 개선된 전처리 데이터")
    print("   - model_comparison_*.png: 성능 비교 그래프")
    print("   - README_IMPROVEMENTS.md: 개선 사항 상세 설명")

if __name__ == "__main__":
    main() 