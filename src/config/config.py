"""
Configuration settings for CCTV danger detection system
"""
import torch

# RTSP 스트림 설정 (다중 카메라 지원)
# 단일 카메라: enabled를 하나만 True로 설정
# 다중 카메라: 여러 개를 True로 설정
CAMERAS = [
    {
        'id': 1,
        'name': 'Camera 1',
        'url': 'rtsp://admin:P%40ssw0rd@192.168.0.24:554/Streaming/Channels/101/',
        'enabled': True,
        'position': (0, 0)  # 화면 표시 위치 (grid)
    },
    {
        'id': 2,
        'name': 'Camera 2',
        'url': 'rtsp://admin:P%40ssw0rd@192.168.0.25:554/Streaming/Channels/101/',
        'enabled': True,  # False로 변경하면 단일 카메라 모드
        'position': (1, 0)
    },
    {
        'id': 3,
        'name': 'Camera 3',
        'url': '',  # 비어있으면 비활성화
        'enabled': False,
        'position': (0, 1)
    },
    {
        'id': 4,
        'name': 'Camera 4',
        'url': '',
        'enabled': False,
        'position': (1, 1)
    }
]

# 기본 RTSP URL (단일 카메라 모드용)
RTSP_URL = CAMERAS[0]['url']

# 모델 설정
MODEL_CONFIG = {
    'yolo_model': 'yolov8n.pt',  # YOLOv8 nano 모델 (빠른 처리)
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # 자동 감지
}

# 위험 감지 설정 - 더 민감하게 조정
DETECTION_CONFIG = {
    'fire_detection': {
        'enabled': True,
        'min_confidence': 0.5,
        'alert_cooldown': 30,
    },
    'restricted_area': {
        'enabled': True,
        'min_confidence': 0.7,
        'restricted_zones': [
            # (x1, y1, x2, y2) 형식의 위험 구역 좌표
            (100, 100, 300, 300),
            (400, 200, 600, 400),
        ],
        'alert_cooldown': 10,
    },
    'fall_detection': {
        'enabled': True,
        'min_confidence': 0.7,
        'aspect_ratio_threshold': 1.5,  # 가로/세로 비율
        'alert_cooldown': 20,
    },
    'helmet_detection': {
        'enabled': True,
        'min_confidence': 0.5,
        'alert_cooldown': 60,
    }
}

# 비디오 처리 설정
VIDEO_CONFIG = {
    'frame_skip': 2,  # 2프레임마다 처리 (적당한 타협점)
    'resize_width': 640,
    'resize_height': 480,
    'buffer_size': 5,
    'reconnect_delay': 5,
    'multi_camera_layout': 'grid',
    'grid_spacing': 10,
    'display_width': 1920,
    'display_height': 1080,
}

# 알림 설정
ALERT_CONFIG = {
    'console_alerts': True,
    'log_file': 'danger_detection.log',
    'alert_sound': False,
}

# 클래스 매핑 (COCO dataset 기준)
COCO_CLASSES = {
    0: 'person',
    39: 'bottle',  # 화재 감지용 임시
    # 실제로는 별도 화재 감지 모델 필요
}

# 커스텀 클래스 (추가 학습 필요)
CUSTOM_CLASSES = {
    'fire': 80,
    'smoke': 81,
    'helmet': 82,
    'no_helmet': 83,
}