#!/usr/bin/env python3
"""
ChapterGenerator.py - 여러 챕터의 모델을 대량으로 생성하는 스크립트

명세:
1. 입력파일: chapter_result.json
2. main.py가 하던 일을 대량으로 처리할 수 있도록 리팩터링한 버전
3. 결과를 chapter-models와 chapter-info에 각각 저장

사용법:
    python ChapterGenerator.py chapter_result.json
"""

import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import subprocess
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chapter_generation.log')
    ]
)
logger = logging.getLogger(__name__)


class ChapterGenerator:
    """여러 챕터의 모델을 대량으로 생성하는 클래스"""
    
    def __init__(self, chapter_result_path: str):
        """
        ChapterGenerator 초기화
        
        Args:
            chapter_result_path (str): chapter_result.json 파일 경로
        """
        self.chapter_result_path = chapter_result_path
        self.chapter_data = None
        self.output_models_dir = "chapter-models"
        self.output_info_dir = "chapter-info"
        self.temp_dir = None
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_models_dir, exist_ok=True)
        os.makedirs(self.output_info_dir, exist_ok=True)
        
        logger.info(f"ChapterGenerator 초기화 완료")
        logger.info(f"모델 출력 디렉토리: {self.output_models_dir}")
        logger.info(f"정보 출력 디렉토리: {self.output_info_dir}")
    
    def load_chapter_data(self) -> bool:
        """
        chapter_result.json 파일을 로드합니다.
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if not os.path.exists(self.chapter_result_path):
                logger.error(f"입력 파일이 존재하지 않습니다: {self.chapter_result_path}")
                return False
            
            with open(self.chapter_result_path, 'r', encoding='utf-8') as f:
                self.chapter_data = json.load(f)
            
            if 'chapters' not in self.chapter_data:
                logger.error("chapter_result.json에 'chapters' 키가 없습니다.")
                return False
            
            chapters_count = len(self.chapter_data['chapters'])
            logger.info(f"챕터 데이터 로드 완료: {chapters_count}개 챕터")
            
            return True
            
        except Exception as e:
            logger.error(f"챕터 데이터 로드 중 오류 발생: {e}")
            return False
    
    def validate_chapter(self, chapter: Dict) -> bool:
        """
        개별 챕터 데이터의 유효성을 검증합니다.
        
        Args:
            chapter (Dict): 챕터 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        if 'chapter_name' not in chapter:
            logger.error("챕터에 'chapter_name'이 없습니다.")
            return False
        
        if 'label_dict' not in chapter:
            logger.error(f"챕터 {chapter['chapter_name']}에 'label_dict'가 없습니다.")
            return False
        
        label_dict = chapter['label_dict']
        
        # None 라벨이 반드시 포함되어야 함
        if 'None' not in label_dict:
            logger.error(f"챕터 {chapter['chapter_name']}에 'None' 라벨이 없습니다.")
            return False
        
        # label_dict가 비어있지 않아야 함
        if len(label_dict) == 0:
            logger.error(f"챕터 {chapter['chapter_name']}의 label_dict가 비어있습니다.")
            return False
        
        logger.debug(f"챕터 {chapter['chapter_name']} 검증 통과: {len(label_dict)}개 라벨")
        return True
    
    def create_spec_file(self, chapter: Dict, temp_dir: str) -> str:
        """
        챕터 데이터로부터 spec.json 파일을 생성합니다.
        
        Args:
            chapter (Dict): 챕터 데이터
            temp_dir (str): 임시 디렉토리 경로
            
        Returns:
            str: 생성된 spec.json 파일 경로
        """
        spec_data = {
            "chapter_name": chapter["chapter_name"],
            "label_dict": chapter["label_dict"]
        }
        
        spec_path = os.path.join(temp_dir, f"spec_{chapter['chapter_name']}.json")
        
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(spec_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Spec 파일 생성: {spec_path}")
        return spec_path
    
    def run_main_for_chapter(self, spec_path: str, chapter_name: str) -> bool:
        """
        개별 챕터에 대해 main.py를 실행합니다.
        
        Args:
            spec_path (str): spec.json 파일 경로
            chapter_name (str): 챕터 이름
            
        Returns:
            bool: 실행 성공 여부
        """
        try:
            logger.info(f"챕터 {chapter_name} 모델 학습 시작...")
            
            # main.py 실행
            result = subprocess.run(
                [sys.executable, "main.py", spec_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1시간 타임아웃
            )
            
            if result.returncode == 0:
                logger.info(f"챕터 {chapter_name} 모델 학습 완료")
                return True
            else:
                logger.error(f"챕터 {chapter_name} 모델 학습 실패:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"챕터 {chapter_name} 모델 학습 타임아웃")
            return False
        except Exception as e:
            logger.error(f"챕터 {chapter_name} 모델 학습 중 오류: {e}")
            return False
    
    def move_results(self, chapter_name: str) -> bool:
        """
        생성된 모델과 정보를 적절한 디렉토리로 이동합니다.
        
        Args:
            chapter_name (str): 챕터 이름
            
        Returns:
            bool: 이동 성공 여부
        """
        try:
            # 모델 파일 이동
            model_src = "models/sign_language_model.keras"
            model_dst = os.path.join(self.output_models_dir, f"{chapter_name}_model.keras")
            
            if os.path.exists(model_src):
                shutil.move(model_src, model_dst)
                logger.info(f"모델 파일 이동: {model_src} -> {model_dst}")
            else:
                logger.warning(f"모델 파일이 존재하지 않습니다: {model_src}")
                return False
            
            # 정보 파일 이동
            info_src = "info/model_info.json"
            info_dst = os.path.join(self.output_info_dir, f"{chapter_name}_info.json")
            
            if os.path.exists(info_src):
                shutil.move(info_src, info_dst)
                logger.info(f"정보 파일 이동: {info_src} -> {info_dst}")
            else:
                logger.warning(f"정보 파일이 존재하지 않습니다: {info_src}")
            
            return True
            
        except Exception as e:
            logger.error(f"결과 파일 이동 중 오류: {e}")
            return False
    
    def cleanup_temp_files(self):
        """임시 파일들을 정리합니다."""
        try:
            # models/ 와 info/ 디렉토리의 파일들 정리
            if os.path.exists("models/sign_language_model.keras"):
                os.remove("models/sign_language_model.keras")
            
            if os.path.exists("info/model_info.json"):
                os.remove("info/model_info.json")
                
            logger.debug("임시 파일 정리 완료")
            
        except Exception as e:
            logger.warning(f"임시 파일 정리 중 오류: {e}")
    
    def process_single_chapter(self, chapter: Dict) -> bool:
        """
        단일 챕터를 처리합니다.
        
        Args:
            chapter (Dict): 챕터 데이터
            
        Returns:
            bool: 처리 성공 여부
        """
        chapter_name = chapter['chapter_name']
        
        logger.info(f"{'='*60}")
        logger.info(f"챕터 처리 시작: {chapter_name}")
        logger.info(f"라벨 개수: {len(chapter['label_dict'])}")
        logger.info(f"{'='*60}")
        
        # 1. 챕터 유효성 검증
        if not self.validate_chapter(chapter):
            logger.error(f"챕터 {chapter_name} 유효성 검증 실패")
            return False
        
        # 2. 임시 spec 파일 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = self.create_spec_file(chapter, temp_dir)
            
            # 3. main.py 실행
            if not self.run_main_for_chapter(spec_path, chapter_name):
                logger.error(f"챕터 {chapter_name} 처리 실패")
                return False
            
            # 4. 결과 파일 이동
            if not self.move_results(chapter_name):
                logger.error(f"챕터 {chapter_name} 결과 파일 이동 실패")
                return False
            
            # 5. 임시 파일 정리
            self.cleanup_temp_files()
        
        logger.info(f"챕터 {chapter_name} 처리 완료")
        return True
    
    def process_all_chapters(self) -> Dict[str, bool]:
        """
        모든 챕터를 처리합니다.
        
        Returns:
            Dict[str, bool]: 각 챕터의 처리 결과
        """
        if not self.chapter_data:
            logger.error("챕터 데이터가 로드되지 않았습니다.")
            return {}
        
        results = {}
        total_chapters = len(self.chapter_data['chapters'])
        
        logger.info(f"총 {total_chapters}개 챕터 처리 시작")
        
        for i, chapter in enumerate(self.chapter_data['chapters'], 1):
            chapter_name = chapter.get('chapter_name', f'chapter_{i}')
            
            logger.info(f"\n진행률: {i}/{total_chapters} - {chapter_name}")
            
            try:
                success = self.process_single_chapter(chapter)
                results[chapter_name] = success
                
                if success:
                    logger.info(f"✅ {chapter_name} 성공")
                else:
                    logger.error(f"❌ {chapter_name} 실패")
                    
            except Exception as e:
                logger.error(f"❌ {chapter_name} 처리 중 예외 발생: {e}")
                results[chapter_name] = False
        
        return results
    
    def generate_summary_report(self, results: Dict[str, bool]) -> str:
        """
        처리 결과 요약 리포트를 생성합니다.
        
        Args:
            results (Dict[str, bool]): 각 챕터의 처리 결과
            
        Returns:
            str: 요약 리포트
        """
        total = len(results)
        success_count = sum(1 for success in results.values() if success)
        failure_count = total - success_count
        
        report = f"""
{'='*80}
챕터 생성 완료 보고서
{'='*80}

📊 전체 통계:
   - 총 챕터 수: {total}
   - 성공: {success_count}
   - 실패: {failure_count}
   - 성공률: {(success_count/total*100) if total > 0 else 0:.1f}%

📁 출력 디렉토리:
   - 모델: {self.output_models_dir}/
   - 정보: {self.output_info_dir}/

"""
        
        if failure_count > 0:
            report += "❌ 실패한 챕터:\n"
            for chapter_name, success in results.items():
                if not success:
                    report += f"   - {chapter_name}\n"
        
        if success_count > 0:
            report += "\n✅ 성공한 챕터:\n"
            for chapter_name, success in results.items():
                if success:
                    report += f"   - {chapter_name}\n"
        
        report += f"\n{'='*80}"
        
        return report
    
    def run(self) -> bool:
        """
        전체 처리 과정을 실행합니다.
        
        Returns:
            bool: 전체 처리 성공 여부
        """
        logger.info("ChapterGenerator 실행 시작")
        
        # 1. 챕터 데이터 로드
        if not self.load_chapter_data():
            return False
        
        # 2. 모든 챕터 처리
        results = self.process_all_chapters()
        
        # 3. 결과 요약
        report = self.generate_summary_report(results)
        logger.info(report)
        
        # 4. 리포트 파일 저장
        with open('chapter_generation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 5. 성공 여부 판단
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        if success_count == total_count:
            logger.info("🎉 모든 챕터 처리 완료!")
            return True
        else:
            logger.warning(f"⚠️ 일부 챕터 처리 실패 ({success_count}/{total_count})")
            return False


def main():
    """메인 실행 함수"""
    if len(sys.argv) != 2:
        print("사용법: python ChapterGenerator.py chapter_result.json")
        sys.exit(1)
    
    chapter_result_path = sys.argv[1]
    
    try:
        generator = ChapterGenerator(chapter_result_path)
        success = generator.run()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
