@echo off
echo ===================================
echo CCTV 위험 감지 시스템 시작
echo ===================================
echo.

REM 가상환경 활성화 (있는 경우)
if exist venv\Scripts\activate.bat (
    echo 가상환경 활성화 중...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Python 경로 설정
set PYTHONPATH=%cd%;%cd%\src

REM 프로그램 실행
echo 프로그램 시작...
python run_detection.py %*

REM 종료 시 대기
echo.
echo 프로그램이 종료되었습니다.
pause