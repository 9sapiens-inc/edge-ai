"""
Fire and smoke detection module
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import sys
import os
from collections import deque

# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .yolo_detector import YOLODetector
except ImportError:
    from models.yolo_detector import YOLODetector

class FireDetector:
    def __init__(self, yolo_detector: YOLODetector):
        self.yolo_detector = yolo_detector
        
        # 개선된 화재 감지 파라미터
        # 화재 색상 범위 (HSV) - 더 엄격하게 조정
        self.fire_lower1 = np.array([0, 120, 70])    # 빨간색 범위 1
        self.fire_upper1 = np.array([10, 255, 255])
        self.fire_lower2 = np.array([170, 120, 70])  # 빨간색 범위 2
        self.fire_upper2 = np.array([180, 255, 255])
        
        # 연기 감지를 위한 색상 범위 - 더 엄격하게
        self.smoke_lower = np.array([0, 0, 80])      # 회색 범위
        self.smoke_upper = np.array([180, 30, 150])
        
        # 모션 감지를 위한 변수
        self.prev_frame = None
        self.motion_threshold = 50
        self.motion_history = deque(maxlen=10)
        
        # 플리커링 감지를 위한 변수
        self.brightness_history = deque(maxlen=15)
        
        # 최소 감지 영역 크기
        self.min_fire_area = 1000
        self.min_smoke_area = 2000
        
    def detect_fire_color(self, frame: np.ndarray) -> List[Dict]:
        """
        개선된 색상 기반 화재 감지
        """
        detections = []
        
        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 두 가지 빨간색 범위 마스크
        fire_mask1 = cv2.inRange(hsv, self.fire_lower1, self.fire_upper1)
        fire_mask2 = cv2.inRange(hsv, self.fire_lower2, self.fire_upper2)
        fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
        
        # 노이즈 제거 - 더 강력하게
        kernel = np.ones((7, 7), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # 가우시안 블러로 부드럽게
        fire_mask = cv2.GaussianBlur(fire_mask, (5, 5), 0)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_fire_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # ROI 추출
                roi = frame[y:y+h, x:x+w]
                roi_hsv = hsv[y:y+h, x:x+w]
                
                # 추가 검증 1: 밝기 체크
                brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                
                # 추가 검증 2: 색상 분포 체크
                hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
                hist = hist.flatten() / hist.sum()
                
                # 빨간색/주황색 영역이 지배적인지 확인
                red_orange_ratio = np.sum(hist[0:20]) + np.sum(hist[160:180])
                
                # 추가 검증 3: 채도 체크
                saturation = np.mean(roi_hsv[:, :, 1])
                
                # 추가 검증 4: 플리커링 체크
                self.brightness_history.append(brightness)
                if len(self.brightness_history) > 5:
                    brightness_std = np.std(self.brightness_history)
                    is_flickering = brightness_std > 10
                else:
                    is_flickering = False
                
                # 종합 판단
                if (brightness > 180 and 
                    red_orange_ratio > 0.3 and 
                    saturation > 100 and
                    (is_flickering or brightness > 200)):
                    
                    confidence = min(0.95, 
                        (brightness / 255) * 0.3 + 
                        red_orange_ratio * 0.3 + 
                        (saturation / 255) * 0.2 +
                        (0.2 if is_flickering else 0))
                    
                    detection = {
                        'bbox': [x, y, x+w, y+h],
                        'confidence': confidence,
                        'class_name': 'fire',
                        'area': area,
                        'brightness': brightness,
                        'flickering': is_flickering
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_smoke(self, frame: np.ndarray) -> List[Dict]:
        """
        개선된 연기 감지 (모션 + 색상 + 패턴)
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 모션 감지
        if self.prev_frame is not None:
            # 프레임 차이 계산
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # 모션 히스토리 업데이트
            self.motion_history.append(motion_mask)
            
            if len(self.motion_history) >= 3:
                # 지속적인 모션 영역 찾기
                accumulated_motion = np.zeros_like(motion_mask)
                for hist_mask in self.motion_history:
                    accumulated_motion = cv2.bitwise_or(accumulated_motion, hist_mask)
                
                # 연기 색상 감지
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                smoke_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)
                
                # 엣지 감지 (연기는 엣지가 부드러움)
                edges = cv2.Canny(gray, 50, 150)
                edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                
                # 연기 특성: 모션이 있고, 회색이며, 엣지가 부드러운 영역
                combined_mask = cv2.bitwise_and(accumulated_motion, smoke_mask)
                combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(edges_dilated))
                
                # 블러 적용 (연기의 특성)
                combined_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)
                _, combined_mask = cv2.threshold(combined_mask, 30, 255, cv2.THRESH_BINARY)
                
                # 윤곽선 찾기
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > self.min_smoke_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 종횡비 체크 (연기는 보통 넓게 퍼짐)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # ROI 분석
                        roi = gray[y:y+h, x:x+w]
                        roi_motion = frame_diff[y:y+h, x:x+w]
                        
                        # 질감 분석 (연기는 균일한 질감)
                        texture_std = np.std(roi)
                        
                        # 상승 모션 체크
                        motion_mean_upper = np.mean(roi_motion[:h//2, :])
                        motion_mean_lower = np.mean(roi_motion[h//2:, :])
                        is_rising = motion_mean_upper > motion_mean_lower
                        
                        if (0.8 < aspect_ratio < 4.0 and 
                            texture_std < 50 and
                            is_rising):
                            
                            confidence = min(0.85, 
                                (area / 20000) * 0.3 +
                                (1 - texture_std / 100) * 0.3 +
                                (0.4 if is_rising else 0))
                            
                            detection = {
                                'bbox': [x, y, x+w, y+h],
                                'confidence': confidence,
                                'class_name': 'smoke',
                                'area': area,
                                'motion_intensity': np.mean(roi_motion),
                                'rising': is_rising
                            }
                            detections.append(detection)
        
        self.prev_frame = gray
        return detections
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        통합 화재/연기 감지 - 더 엄격한 기준 적용
        """
        all_detections = []
        
        # 색상 기반 화재 감지
        fire_detections = self.detect_fire_color(frame)
        
        # 신뢰도 필터링 - 더 높은 임계값
        fire_detections = [d for d in fire_detections if d['confidence'] > 0.7]
        all_detections.extend(fire_detections)
        
        # 연기 감지
        smoke_detections = self.detect_smoke(frame)
        
        # 신뢰도 필터링
        smoke_detections = [d for d in smoke_detections if d['confidence'] > 0.6]
        all_detections.extend(smoke_detections)
        
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
                # 다중 감지는 신뢰도를 높임
                best['confidence'] = min(0.95, best['confidence'] * (1 + 0.05 * len(overlapping)))
                merged.append(best)
            else:
                merged.append(det1)
        
        return merged