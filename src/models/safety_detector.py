"""
Safety equipment detection module (helmet, restricted area)
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

class SafetyDetector:
    def __init__(self, yolo_detector: YOLODetector, restricted_zones: List[Tuple] = None):
        self.yolo_detector = yolo_detector
        self.restricted_zones = restricted_zones or []
        
        # 안전모 색상 범위 (HSV)
        self.helmet_colors = {
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'orange': ([10, 100, 100], [20, 255, 255])
        }
        
    def detect_restricted_area_intrusion(self, frame: np.ndarray) -> List[Dict]:
        """
        위험 구역 침입 감지
        """
        # 사람 감지
        person_detections = self.yolo_detector.detect(frame, target_classes=[0])
        
        intrusion_detections = []
        
        for person in person_detections:
            person_bbox = person['bbox']
            person_center = person['center']
            
            # 각 제한 구역에 대해 체크
            for i, zone in enumerate(self.restricted_zones):
                if self._is_in_zone(person_center, person_bbox, zone):
                    intrusion = person.copy()
                    intrusion['class_name'] = 'restricted_area_intrusion'
                    intrusion['zone_id'] = i
                    intrusion['zone_bbox'] = zone
                    intrusion['intrusion_level'] = self._calculate_intrusion_level(person_bbox, zone)
                    intrusion_detections.append(intrusion)
                    break
        
        return intrusion_detections
    
    def detect_helmet(self, frame: np.ndarray) -> List[Dict]:
        """
        안전모 착용 여부 감지
        """
        # 사람 감지
        person_detections = self.yolo_detector.detect(frame, target_classes=[0])
        
        helmet_detections = []
        
        for person in person_detections:
            # 사람의 상단 부분 추출 (머리 영역)
            head_region = self._get_head_region(person['bbox'], frame.shape)
            
            # 머리 영역에서 안전모 감지
            has_helmet, helmet_color = self._detect_helmet_in_region(
                frame[head_region[1]:head_region[3], head_region[0]:head_region[2]]
            )
            
            if not has_helmet:
                no_helmet = person.copy()
                no_helmet['class_name'] = 'no_helmet'
                no_helmet['head_region'] = head_region
                helmet_detections.append(no_helmet)
            else:
                # 안전모 착용 확인 (선택적으로 기록)
                helmet = person.copy()
                helmet['class_name'] = 'helmet_detected'
                helmet['helmet_color'] = helmet_color
                helmet['head_region'] = head_region
                # helmet_detections.append(helmet)  # 필요시 주석 해제
        
        return helmet_detections
    
    def _is_in_zone(self, center: Tuple, bbox: List[int], zone: Tuple) -> bool:
        """
        바운딩 박스가 구역 내에 있는지 확인
        """
        x1, y1, x2, y2 = bbox
        zx1, zy1, zx2, zy2 = zone
        
        # 중심점이 구역 내에 있는지 확인
        if zx1 <= center[0] <= zx2 and zy1 <= center[1] <= zy2:
            return True
        
        # 바운딩 박스가 구역과 겹치는지 확인
        if not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2):
            return True
        
        return False
    
    def _calculate_intrusion_level(self, person_bbox: List[int], zone: Tuple) -> float:
        """
        침입 정도 계산 (0~1)
        """
        x1, y1, x2, y2 = person_bbox
        zx1, zy1, zx2, zy2 = zone
        
        # 교집합 영역 계산
        ix1 = max(x1, zx1)
        iy1 = max(y1, zy1)
        ix2 = min(x2, zx2)
        iy2 = min(y2, zy2)
        
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        person_area = (x2 - x1) * (y2 - y1)
        
        return intersection / person_area if person_area > 0 else 0.0
    
    def _get_head_region(self, person_bbox: List[int], frame_shape: Tuple) -> List[int]:
        """
        사람 바운딩 박스에서 머리 영역 추출
        """
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        
        # 상단 25% 영역을 머리로 가정
        head_height = int(height * 0.25)
        
        # 머리 영역 바운딩 박스
        head_x1 = x1
        head_y1 = max(0, y1 - int(height * 0.1))  # 약간 위로 확장
        head_x2 = x2
        head_y2 = min(frame_shape[0], y1 + head_height)
        
        return [head_x1, head_y1, head_x2, head_y2]
    
    def _detect_helmet_in_region(self, head_region: np.ndarray) -> Tuple[bool, str]:
        """
        머리 영역에서 안전모 감지
        """
        if head_region.size == 0:
            return False, None
        
        # HSV 변환
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # 각 안전모 색상에 대해 체크
        for color_name, (lower, upper) in self.helmet_colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            # 색상 마스크
            mask = cv2.inRange(hsv, lower, upper)
            
            # 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 안전모로 판단할 최소 픽셀 비율
            helmet_ratio = np.sum(mask > 0) / (head_region.shape[0] * head_region.shape[1])
            
            if helmet_ratio > 0.2:  # 20% 이상이 안전모 색상
                # 추가 검증: 원형 또는 타원형 체크
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # 최소 면적
                        # 원형도 체크
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        
                        if circularity > 0.5:  # 어느 정도 원형
                            return True, color_name
        
        # 추가: 형태 기반 감지 (색상 없이)
        # 엣지 검출
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 원형 검출 (Hough 변환)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        if circles is not None and len(circles[0]) > 0:
            return True, 'unknown'
        
        return False, None
    
    def visualize_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        제한 구역 시각화
        """
        frame_copy = frame.copy()
        
        for i, zone in enumerate(self.restricted_zones):
            x1, y1, x2, y2 = zone
            
            # 반투명 빨간색 오버레이
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)
            
            # 테두리
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 라벨
            label = f"Restricted Zone {i+1}"
            cv2.putText(frame_copy, label, (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy