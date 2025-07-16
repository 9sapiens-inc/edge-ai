# CCTV 위험 감지 시스템 실행 스크립트

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "CCTV 위험 감지 시스템 시작" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# 가상환경 활성화
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "가상환경 활성화 중..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

# Python 경로 설정
$env:PYTHONPATH = "$PWD;$PWD\src"

# 프로그램 실행
Write-Host "프로그램 시작..." -ForegroundColor Green
python run_detection.py $args

# 종료 시 대기
Write-Host ""
Write-Host "프로그램이 종료되었습니다." -ForegroundColor Yellow
Write-Host "아무 키나 누르면 창이 닫힙니다..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")