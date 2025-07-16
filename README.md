## 설치 및 실행 가이드

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
venv\Scripts\activate  # Windows
# 또는
source venv/bin/activate  # Linux/Mac
# 빠져나가기
deactivate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 모델 다운로드
프로그램 첫 실행 시 YOLOv8 모델이 자동으로 다운로드됩니다.

### 3. 실행
```bash
# 기본 실행 (화면 표시 포함)
python main.py

# 화면 표시 없이 실행
python main.py --no-display

# 결과 비디오 저장
python main.py --save-output

# 다른 RTSP URL 사용
python main.py --rtsp-url "rtsp://your-camera-url"
```

### 4. 사용 중 단축키
- `q`: 프로그램 종료
- `r`: 통계 초기화
- `s`: 현재 화면 스크린샷 저장

## 주요 기능 설명

### 1. **화재/연기 감지**
- 색상 기반 화염 감지 (HSV 색상 공간)
- 모션 기반 연기 감지
- 열 이상 감지 (밝기 분석)

### 2. **낙상 감지**
- 사람의 종횡비 변화 감지
- 급격한 Y축 이동 감지
- 자세 변화 추적
- 개인별 히스토리 관리

### 3. **제한구역 침입 감지**
- 설정된 위험 구역 모니터링
- 사람 위치 실시간 추적
- 침입 정도 계산

### 4. **안전모 미착용 감지**
- 머리 영역 자동 추출
- 색상 기반 안전모 감지
- 형태 기반 보조 감지

## 성능 최적화 팁

1. **GPU 사용**: CUDA가 설치된 경우 자동으로 GPU 사용
2. **프레임 스킵**: `config.py`의 `frame_skip` 값 조정
3. **해상도 조정**: `resize_width`, `resize_height` 값 변경
4. **모델 선택**: 더 빠른 처리를 원하면 `yolov8n.pt` 사용

## 추가 개선 사항

실제 운영 환경에서는 다음 사항들을 추가로 구현하는 것을 권장합니다:

1. **커스텀 모델 학습**: 화재, 안전모 등 특정 객체 감지를 위한 전용 모델
2. **알림 시스템**: 이메일, SMS, 알람 등 다양한 알림 방식
3. **데이터베이스 연동**: 감지 이력 저장 및 분석
4. **웹 인터페이스**: 원격 모니터링 대시보드
5. **다중 카메라 지원**: 여러 CCTV 동시 모니터링
