"""
Fire and smoke detection module
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .yolo_detector import YOLODetector
except ImportError:
    from models.yolo_detector import YOLODetector

class FireDetector:
    def __init__(self, yolo_detector: YOLODetector):
        self.yolo_detector = yolo_detector
        
        # 화재 감지를 위한 색상 범위 (HSV)
        self.fire_lower = np.array([0, 50, 50])
        self.fire_upper = np.array([35, 255, 255])
        
        # 연기 감지를 위한 색상 범위
        self.smoke_lower = np.array([0, 0, 50])
        self.smoke_upper = np.array([180, 50, 200])
        
        # 모션 감지를 위한 변수
        self.prev_frame = None
        self.motion_threshold = 30
        
    def detect_fire_color(self, frame: np.ndarray) -> List[Dict]:
        """
        색상 기반 화재 감지
        """
        detections = []
        
        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 화재 색상 마스크
        fire_mask = cv2.inRange(hsv, self.fire_lower, self.fire_upper)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 최소 면적 필터
                x, y, w, h = cv2.boundingRect(contour)
                
                # 추가 검증: 밝기 체크
                roi = frame[y:y+h, x:x+w]
                brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                
                if brightness > 150:  # 밝은 영역
                    detection = {
                        'bbox': [x, y, x+w, y+h],
                        'confidence': min(0.9, area / 5000),  # 면적 기반 신뢰도
                        'class_name': 'fire',
                        'area': area,
                        'brightness': brightness
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_smoke(self, frame: np.ndarray) -> List[Dict]:
        """
        연기 감지 (모션 + 색상)
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 모션 감지
        if self.prev_frame is not None:
            # 프레임 차이 계산
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            _, motion_mask = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # 연기 색상 감지
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            smoke_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)
            
            # 모션과 색상 결합
            combined_mask = cv2.bitwise_and(motion_mask, smoke_mask)
            
            # 블러 적용 (연기의 특성)
            combined_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)
            _, combined_mask = cv2.threshold(combined_mask, 50, 255, cv2.THRESH_BINARY)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 최소 면적
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 종횡비 체크 (연기는 보통 넓게 퍼짐)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 3.0:
                        detection = {
                            'bbox': [x, y, x+w, y+h],
                            'confidence': min(0.8, area / 10000),
                            'class_name': 'smoke',
                            'area': area,
                            'motion_intensity': np.mean(frame_diff[y:y+h, x:x+w])
                        }
                        detections.append(detection)
        
        self.prev_frame = gray
        return detections
    
    def detect_thermal_anomaly(self, frame: np.ndarray) -> List[Dict]:
        """
        열 이상 감지 (밝기 기반)
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 적응형 임계값으로 밝은 영역 찾기
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, -20)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                
                # 평균 밝기와 표준편차 계산
                mean_brightness = np.mean(roi)
                std_brightness = np.std(roi)
                
                # 매우 밝고 균일한 영역 (화염 가능성)
                if mean_brightness > 200 and std_brightness < 30:
                    detection = {
                        'bbox': [x, y, x+w, y+h],
                        'confidence': min(0.85, mean_brightness / 255),
                        'class_name': 'thermal_anomaly',
                        'mean_brightness': mean_brightness,
                        'std_brightness': std_brightness
                    }
                    detections.append(detection)
        
        return detections
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        통합 화재/연기 감지
        """
        all_detections = []
        
        # 색상 기반 화재 감지
        fire_detections = self.detect_fire_color(frame)
        all_detections.extend(fire_detections)
        
        # 연기 감지
        smoke_detections = self.detect_smoke(frame)
        all_detections.extend(smoke_detections)
        
        # 열 이상 감지
        thermal_detections = self.detect_thermal_anomaly(frame)
        all_detections.extend(thermal_detections)
        
        # 중복 제거 및 병합
        merged_detections = self._merge_detections(all_detections)
        
        return merged_detections
    
    def _merge_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        겹치는 감지 결과 병합
        """
        if not detections:
            return []
        
        # 신뢰도로 정렬
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            # 겹치는 감지 찾기
            overlapping = [det1]
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                iou = self.yolo_detector.check_overlap(det1['bbox'], det2['bbox'])
                if iou > iou_threshold:
                    overlapping.append(det2)
                    used.add(j)
            
            # 병합
            if len(overlapping) > 1:
                # 가장 높은 신뢰도 선택
                best = max(overlapping, key=lambda x: x['confidence'])
                best['confidence'] = min(0.95, best['confidence'] * (1 + 0.1 * len(overlapping)))
                merged.append(best)
            else:
                merged.append(det1)
        
        return merged