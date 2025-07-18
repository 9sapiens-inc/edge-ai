"""
YOLOv8 기반 화재 및 연기 감지 모듈
Abonia1/YOLOv8-Fire-and-Smoke-Detection 모델 사용
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .yolo_detector import YOLODetector
except ImportError:
    from models.yolo_detector import YOLODetector

class FireDetectorYOLO:
    def __init__(self, model_path: str = 'weights/fire_smoke_best.pt'):
        """
        YOLOv8 기반 화재/연기 감지기
        
        Args:
            model_path: 화재 감지 전용 YOLOv8 모델 경로
        """
        # 화재 감지 전용 YOLO 모델 로드
        try:
            self.model = YOLO(model_path)
            print(f"화재 감지 모델 로드 완료: {model_path}")
            
            # 클래스 이름 확인 (보통 'Fire', 'Smoke' 등)
            self.class_names = self.model.names
            print(f"감지 가능 클래스: {self.class_names}")
            
        except Exception as e:
            print(f"화재 감지 모델 로드 실패: {e}")
            print("기본 모델로 대체합니다.")
            # 실패 시 기본 YOLOv8 사용
            self.model = YOLO('yolov8n.pt')
            self.class_names = {}
        
        # 신뢰도 임계값
        self.confidence_threshold = 0.5
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        프레임에서 화재/연기 감지
        
        Args:
            frame: 입력 이미지
            
        Returns:
            감지 결과 리스트
        """
        detections = []
        
        try:
            # YOLO 추론
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # 클래스 확인
                    class_id = int(box.cls)
                    class_name = self.class_names.get(class_id, 'unknown')
                    
                    # 화재 또는 연기 클래스인지 확인
                    if class_name in ['Fire', 'Smoke']:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        # 감지 영역 크기 계산
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # 너무 작은 감지는 무시
                        if area < 500:
                            continue
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id,
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                            'width': width,
                            'height': height,
                            'area': area
                        }
                        
                        detections.append(detection)
            
        except Exception as e:
            print(f"화재 감지 중 오류: {e}")
        
        return detections
    
    def update_confidence_threshold(self, threshold: float):
        """
        신뢰도 임계값 업데이트
        
        Args:
            threshold: 새로운 임계값 (0.0 ~ 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"화재 감지 신뢰도 임계값 변경: {self.confidence_threshold}")
    
    def get_statistics(self) -> Dict:
        """
        감지 통계 반환
        """
        return {
            'model_name': self.model.model.names if hasattr(self.model, 'model') else 'unknown',
            'confidence_threshold': self.confidence_threshold,
            'classes': self.class_names
        }