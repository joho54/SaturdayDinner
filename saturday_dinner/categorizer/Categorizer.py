import os
import sys
import subprocess

def check_file_exists(file_path):
    """파일이 존재하는지 확인"""
    return os.path.exists(file_path)

def run_python_script(script_path):
    """Python 스크립트 실행"""
    try:
        print(f"실행 중: {script_path}")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(f"✓ {script_path} 성공적으로 완료")
        if result.stdout:
            print(f"출력: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_path} 실행 실패: {e}")
        if e.stderr:
            print(f"오류: {e.stderr}")
        return False

def main():
    # 현재 디렉토리에서 labels.csv 파일 확인
    labels_path = "../labels.csv"  # 현재 디렉토리의 상위 디렉토리에서 확인
    
    print("=== 카테고라이저 시작 ===")
    
    if not check_file_exists(labels_path):
        print(f"오류: {labels_path} 파일을 찾을 수 없습니다.")
        return False
    
    print(f"✓ {labels_path} 파일을 찾았습니다.")
    
    # 실행할 스크립트 목록 (순서대로)
    scripts = [
        "VideoCluster.py",
        "LabelCluster.py", 
        "CrossCategorizer.py"
    ]
    
    # 각 스크립트를 차례로 실행
    for script in scripts:
        if not check_file_exists(script):
            print(f"오류: {script} 파일을 찾을 수 없습니다.")
            return False
        
        if not run_python_script(script):
            print(f"오류: {script} 실행 중 문제가 발생했습니다.")
            return False
    
    print("=== 모든 스크립트가 성공적으로 완료되었습니다! ===")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
