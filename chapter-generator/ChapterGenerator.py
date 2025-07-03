#!/usr/bin/env python3
"""
ChapterGenerator - 여러 챕터의 label_dict를 처리하여 각각의 모델을 생성하는 클래스

이 클래스는 chapter_result.json 파일에서 여러 챕터를 읽어와서
각 챕터별로 개별적인 모델을 학습시키고, 파일명에 챕터명을 포함하여 저장합니다.
"""

import os
import json
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Optional


class ChapterGenerator:
    """여러 챕터의 label_dict를 처리하여 각각의 모델을 생성하는 클래스"""

    def __init__(self, chapter_file: str = "chapter_result.json", base_output_dir: str = None):
        """
        ChapterGenerator 초기화
        
        Args:
            chapter_file: 챕터 정보가 포함된 JSON 파일 경로
            base_output_dir: 출력 디렉토리 기본 경로 (기본값: chapter-generator)
        """
        self.chapter_file = chapter_file
        self.base_output_dir = base_output_dir or os.path.dirname(os.path.abspath(__file__))
        self.main_py_path = os.path.join(self.base_output_dir, "main.py")
        
        # 로깅 설정
        self.setup_logging()
        
        # 기본 디렉토리 생성
        self.setup_directories()
        
        self.logger.info(f"ChapterGenerator 초기화 완료")
        self.logger.info(f"Chapter file: {self.chapter_file}")
        self.logger.info(f"Base output directory: {self.base_output_dir}")

    def setup_logging(self):
        """로깅 설정"""
        log_file = os.path.join(self.base_output_dir, "chapter_generation.log")
        
        # 로거 설정
        self.logger = logging.getLogger('ChapterGenerator')
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

    def setup_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "info"), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, "cache"), exist_ok=True)

    def load_chapters(self) -> Dict:
        """chapter_result.json 파일을 로드"""
        chapter_path = os.path.join(self.base_output_dir, self.chapter_file)
        
        if not os.path.exists(chapter_path):
            raise FileNotFoundError(f"Chapter file not found: {chapter_path}")
            
        try:
            with open(chapter_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.logger.info(f"Loaded chapter file: {len(data.get('chapters', []))} chapters found")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {chapter_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading chapter file: {e}")

    def validate_chapter(self, chapter: Dict) -> bool:
        """챕터 데이터 유효성 검사"""
        required_fields = ['chapter_name', 'label_dict']
        
        for field in required_fields:
            if field not in chapter:
                self.logger.error(f"Missing required field '{field}' in chapter")
                return False
        
        label_dict = chapter['label_dict']
        
        # label_dict가 비어있지 않은지 확인
        if not label_dict:
            self.logger.error(f"Empty label_dict in chapter {chapter['chapter_name']}")
            return False
            
        # None 라벨이 포함되어 있는지 확인
        if 'None' not in label_dict:
            self.logger.error(f"'None' label missing in chapter {chapter['chapter_name']}")
            return False
            
        # 라벨 값이 숫자인지 확인
        for label, value in label_dict.items():
            if not isinstance(value, int):
                self.logger.error(f"Non-integer label value in chapter {chapter['chapter_name']}: {label}={value}")
                return False
                
        return True

    def create_spec_file(self, chapter: Dict) -> str:
        """챕터 정보로부터 spec.json 파일 생성"""
        chapter_name = chapter['chapter_name']
        spec_data = {
            "label_dict": chapter['label_dict']
        }
        
        spec_file = os.path.join(self.base_output_dir, f"spec_{chapter_name}.json")
        
        try:
            with open(spec_file, 'w', encoding='utf-8') as f:
                json.dump(spec_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Created spec file for {chapter_name}: {spec_file}")
            return spec_file
            
        except Exception as e:
            raise RuntimeError(f"Error creating spec file for {chapter_name}: {e}")



    def run_main_for_chapter(self, spec_file: str, chapter_name: str) -> bool:
        """특정 챕터에 대해 main.py 실행"""
        if not os.path.exists(self.main_py_path):
            self.logger.error(f"main.py not found at: {self.main_py_path}")
            return False
            
        try:
            # 현재 디렉토리를 chapter-generator로 변경
            original_cwd = os.getcwd()
            os.chdir(self.base_output_dir)
            
            self.logger.info(f"Starting training for {chapter_name}")
            self.logger.info(f"Command: python main.py {spec_file}")
            print(f"\n{'='*80}")
            print(f"🚀 Starting training for {chapter_name}")
            print(f"Command: python main.py {spec_file}")
            print(f"{'='*80}")
            
            # main.py 실행 (실시간 출력)
            result = subprocess.run(
                [sys.executable, "main.py", spec_file],               
                text=True,
                encoding='utf-8'
            )
            
            # 원래 디렉토리로 복귀
            os.chdir(original_cwd)
            
            print(f"\n{'='*80}")
            if result.returncode == 0:
                print(f"✅ Successfully completed training for {chapter_name}")
                self.logger.info(f"Successfully completed training for {chapter_name}")
                return True
            else:
                print(f"❌ Training failed for {chapter_name} (exit code: {result.returncode})")
                self.logger.error(f"Training failed for {chapter_name} (exit code: {result.returncode})")
                return False
                
        except Exception as e:
            # 원래 디렉토리로 복귀
            os.chdir(original_cwd)
            print(f"❌ Exception during training for {chapter_name}: {e}")
            self.logger.error(f"Exception during training for {chapter_name}: {e}")
            return False
                    
        except Exception as e:
            self.logger.error(f"Error renaming outputs for {chapter_name}: {e}")

    def cleanup_temp_files(self, spec_file: str):
        """임시 파일 정리"""
        try:
            if os.path.exists(spec_file):
                os.remove(spec_file)
                self.logger.info(f"Removed temporary spec file: {spec_file}")
        except Exception as e:
            self.logger.warning(f"Error removing temporary file {spec_file}: {e}")

    def process_all_chapters(self) -> Dict[str, bool]:
        """모든 챕터를 처리"""
        try:
            # 챕터 데이터 로드
            chapter_data = self.load_chapters()
            chapters = chapter_data.get('chapters', [])
            
            if not chapters:
                self.logger.error("No chapters found in the input file")
                return {}
                
            results = {}
            successful_chapters = 0
            failed_chapters = 0
            
            self.logger.info(f"Processing {len(chapters)} chapters...")
            
            for i, chapter in enumerate(chapters, 1):
                chapter_name = chapter.get('chapter_name', f'chapter_{i}')
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing chapter {i}/{len(chapters)}: {chapter_name}")
                self.logger.info(f"{'='*60}")
                
                try:
                    # 챕터 유효성 검사
                    if not self.validate_chapter(chapter):
                        self.logger.error(f"Invalid chapter data for {chapter_name}")
                        results[chapter_name] = False
                        failed_chapters += 1
                        continue
                    
                    # spec 파일 생성
                    spec_file = self.create_spec_file(chapter)
                    
                    # main.py 실행
                    success = self.run_main_for_chapter(spec_file, chapter_name)
                    
                    if success:
                        successful_chapters += 1
                    else:
                        failed_chapters += 1
                    
                    results[chapter_name] = success
                    
                    # 임시 파일 정리
                    self.cleanup_temp_files(spec_file)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {chapter_name}: {e}")
                    results[chapter_name] = False
                    failed_chapters += 1
            
            # 최종 결과 요약
            self.logger.info(f"\n{'='*60}")
            self.logger.info("CHAPTER GENERATION SUMMARY")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Total chapters: {len(chapters)}")
            self.logger.info(f"Successful: {successful_chapters}")
            self.logger.info(f"Failed: {failed_chapters}")
            self.logger.info(f"Success rate: {(successful_chapters/len(chapters)*100):.1f}%")
            
            # 상세 결과
            self.logger.info(f"\nDetailed results:")
            for chapter_name, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                self.logger.info(f"  {chapter_name}: {status}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error in process_all_chapters: {e}")
            return {}

    def generate_report(self, results: Dict[str, bool]):
        """처리 결과 리포트 생성"""
        report_file = os.path.join(self.base_output_dir, "chapter_generation_report.txt")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("CHAPTER GENERATION REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total chapters processed: {len(results)}\n")
                
                successful = sum(1 for success in results.values() if success)
                failed = len(results) - successful
                
                f.write(f"Successful: {successful}\n")
                f.write(f"Failed: {failed}\n")
                f.write(f"Success rate: {(successful/len(results)*100):.1f}%\n\n")
                
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                for chapter_name, success in results.items():
                    status = "SUCCESS" if success else "FAILED"
                    f.write(f"{chapter_name}: {status}\n")
                    
                    if success:
                        # 챕터별 출력 파일 정보 (파일명에서 챕터명으로 찾기)
                        models_dir = os.path.join(self.base_output_dir, "models")
                        info_dir = os.path.join(self.base_output_dir, "info")
                        
                        if os.path.exists(models_dir):
                            model_files = [f for f in os.listdir(models_dir) 
                                         if f.endswith('.keras') and chapter_name in f]
                            f.write(f"  Model files: {len(model_files)}\n")
                            for model_file in model_files:
                                f.write(f"    - {model_file}\n")
                        
                        if os.path.exists(info_dir):
                            info_files = [f for f in os.listdir(info_dir) 
                                        if f.endswith('.json') and chapter_name in f]
                            f.write(f"  Info files: {len(info_files)}\n")
                            for info_file in info_files:
                                f.write(f"    - {info_file}\n")
                    
                    f.write("\n")
                
            self.logger.info(f"Report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")


def main():
    """메인 실행 함수"""
    generator = ChapterGenerator()
    
    try:
        # 모든 챕터 처리
        results = generator.process_all_chapters()
        
        # 리포트 생성
        generator.generate_report(results)
        
        # 성공적으로 완료된 경우
        if results:
            successful_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            if successful_count == total_count:
                print(f"\n🎉 All {total_count} chapters processed successfully!")
                sys.exit(0)
            else:
                print(f"\n⚠️  {successful_count}/{total_count} chapters processed successfully")
                sys.exit(1)
        else:
            print("\n❌ No chapters were processed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
