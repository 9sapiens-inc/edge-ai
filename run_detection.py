"""
CCTV 위험 감지 시스템 실행 스크립트
Windows/VSCode 환경에서의 경로 문제 해결
"""
import sys
import os
import locale

# UTF-8 인코딩 설정 (Windows)
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    
# 환경 변수 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 프로젝트 루트와 src 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

# main 모듈 import 및 실행
from src.main import main

if __name__ == "__main__":
    main()