"""
CCTV 다중 카메라 위험 감지 시스템 실행 스크립트
"""
import sys
import os

# 프로젝트 루트와 src 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

# main_multi 모듈 import 및 실행
from src.main_multi import main

if __name__ == "__main__":
    main()