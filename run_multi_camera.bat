@echo off
echo ===================================
echo CCTV 다중 카메라 위험 감지 시스템
echo ===================================
echo.

REM 가상환경 활성화 (있는 경우)
if exist venv\Scripts\activate.bat (
    echo 가상환경 활성화 중...
    call venv\Scripts\activate.bat
)

REM Python 경로 설정
set PYTHONPATH=%cd%;%cd%\src

REM 프로그램 실행
echo 다중 카메라 모드로 시작합니다...
python run_multi_camera.py %*

REM 종료 시 대기
echo.
echo 프로그램이 종료되었습니다.
pause