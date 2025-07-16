"""
Base YOLO detector class
"""
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import cv2

class YOLODetector:
    def __init__(self, model_path: str = 'yolov8n.pt', 
                 device: str = 'cuda:0',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        YOLO 기반 객체 탐지기
        
        Args:
            model_path: YOLO 모델 경로
            device: 실행 디바이스 (cuda:0 또는 cpu)
            conf_threshold: 신뢰도 임계값
            iou_threshold: IOU 임계값
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # YOLO 모델 로드
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
        
        # 클래스 이름 가져오기
        self.class_names = self.model.names
        
    def detect(self, frame: np.ndarray, 
               target_classes: Optional[List[int]] = None) -> List[Dict]:
        """
        프레임에서 객체 탐지
        
        Args:
            frame: 입력 이미지
            target_classes: 탐지할 클래스 ID 리스트
            
        Returns:
            탐지 결과 리스트
        """
        try:
            # YOLO 추론
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # 클래스 필터링
                    class_id = int(box.cls)
                    if target_classes and class_id not in target_classes:
                        continue
                    
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf),
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, 'unknown'),
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'aspect_ratio': (x2 - x1) / (y2 - y1) if y2 - y1 > 0 else 0
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        탐지 결과를 프레임에 그리기
        
        Args:
            frame: 입력 이미지
            detections: 탐지 결과
            color: 바운딩 박스 색상
            thickness: 선 두께
            
        Returns:
            그려진 이미지
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # 라벨 배경
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, 
                         (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # 라벨 텍스트
            cv2.putText(frame_copy, label,
                       (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
        
        return frame_copy
    
    def check_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        두 바운딩 박스의 겹침 정도 계산
        
        Returns:
            IOU (Intersection over Union) 값
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0