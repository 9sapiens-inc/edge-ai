"""
Fall detection module
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import sys
import os

# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .yolo_detector import YOLODetector
except ImportError:
    from models.yolo_detector import YOLODetector

class FallDetector:
    def __init__(self, yolo_detector: YOLODetector):
        self.yolo_detector = yolo_detector
        
        # 사람 추적 정보
        self.person_history = {}  # person_id: deque of positions
        self.fall_status = {}     # person_id: is_fallen
        self.history_size = 10    # 프레임 히스토리 크기
        
        # 낙상 감지 파라미터
        self.aspect_ratio_threshold = 1.5  # 가로/세로 비율
        self.velocity_threshold = 50       # 픽셀/프레임
        self.angle_threshold = 60          # 각도 (도)
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        낙상 감지
        """
        # YOLO로 사람 감지
        person_detections = self.yolo_detector.detect(frame, target_classes=[0])  # 0: person
        
        fall_detections = []
        
        for det in person_detections:
            # 사람 ID 할당 (간단한 위치 기반)
            person_id = self._get_person_id(det)
            
            # 히스토리 업데이트
            self._update_history(person_id, det)
            
            # 낙상 감지 체크
            is_fallen = self._check_fall(person_id, det)
            
            if is_fallen:
                fall_det = det.copy()
                fall_det['class_name'] = 'fall_detected'
                fall_det['person_id'] = person_id
                fall_det['fall_confidence'] = self._calculate_fall_confidence(person_id, det)
                fall_detections.append(fall_det)
        
        # 오래된 추적 정보 제거
        self._cleanup_old_tracks()
        
        return fall_detections
    
    def _get_person_id(self, detection: Dict) -> str:
        """
        간단한 사람 ID 생성 (실제로는 더 정교한 추적 알고리즘 필요)
        """
        cx, cy = detection['center']
        
        # 가장 가까운 기존 추적 찾기
        min_dist = float('inf')
        best_id = None
        
        for pid, history in self.person_history.items():
            if history:
                last_pos = history[-1]['center']
                dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                
                if dist < min_dist and dist < 100:  # 100픽셀 이내
                    min_dist = dist
                    best_id = pid
        
        if best_id is None:
            # 새로운 사람
            best_id = f"person_{len(self.person_history)}_{int(cx)}_{int(cy)}"
        
        return best_id
    
    def _update_history(self, person_id: str, detection: Dict):
        """
        사람 추적 히스토리 업데이트
        """
        if person_id not in self.person_history:
            self.person_history[person_id] = deque(maxlen=self.history_size)
        
        self.person_history[person_id].append({
            'center': detection['center'],
            'bbox': detection['bbox'],
            'aspect_ratio': detection['aspect_ratio'],
            'height': detection['height'],
            'width': detection['width']
        })
    
    def _check_fall(self, person_id: str, detection: Dict) -> bool:
        """
        낙상 여부 체크
        """
        # 1. 종횡비 체크 (누워있으면 가로가 더 김)
        aspect_ratio = detection['aspect_ratio']
        is_horizontal = aspect_ratio > self.aspect_ratio_threshold
        
        # 2. 속도 체크 (급격한 하강)
        velocity = self._calculate_velocity(person_id)
        is_fast_movement = velocity > self.velocity_threshold
        
        # 3. 위치 변화 체크 (Y 좌표의 급격한 증가)
        y_change = self._calculate_y_change(person_id)
        is_downward = y_change > 30
        
        # 4. 자세 변화 체크
        posture_changed = self._check_posture_change(person_id)
        
        # 종합 판단
        if is_horizontal and (is_fast_movement or is_downward) and posture_changed:
            self.fall_status[person_id] = True
            return True
        
        # 이미 넘어진 상태 유지
        if person_id in self.fall_status and self.fall_status[person_id]:
            # 일어났는지 체크
            if aspect_ratio < 0.8:  # 서있는 자세
                self.fall_status[person_id] = False
                return False
            return True
        
        return False
    
    def _calculate_velocity(self, person_id: str) -> float:
        """
        이동 속도 계산
        """
        history = self.person_history.get(person_id, deque())
        
        if len(history) < 2:
            return 0
        
        curr_pos = history[-1]['center']
        prev_pos = history[-2]['center']
        
        velocity = np.sqrt(
            (curr_pos[0] - prev_pos[0])**2 + 
            (curr_pos[1] - prev_pos[1])**2
        )
        
        return velocity
    
    def _calculate_y_change(self, person_id: str) -> float:
        """
        Y 좌표 변화량 계산
        """
        history = self.person_history.get(person_id, deque())
        
        if len(history) < 3:
            return 0
        
        recent_y = [h['center'][1] for h in list(history)[-3:]]
        return recent_y[-1] - recent_y[0]
    
    def _check_posture_change(self, person_id: str) -> bool:
        """
        자세 변화 체크
        """
        history = self.person_history.get(person_id, deque())
        
        if len(history) < 5:
            return False
        
        # 최근 종횡비 변화
        recent_ratios = [h['aspect_ratio'] for h in list(history)[-5:]]
        
        # 초기 평균과 현재 비교
        initial_avg = np.mean(recent_ratios[:3])
        current = recent_ratios[-1]
        
        # 큰 변화가 있으면 자세 변화로 판단
        return abs(current - initial_avg) > 0.5
    
    def _calculate_fall_confidence(self, person_id: str, detection: Dict) -> float:
        """
        낙상 신뢰도 계산
        """
        confidence = 0.0
        
        # 종횡비 기여도
        aspect_ratio = detection['aspect_ratio']
        if aspect_ratio > self.aspect_ratio_threshold:
            confidence += 0.3 * min(1.0, (aspect_ratio - 1) / 2)
        
        # 속도 기여도
        velocity = self._calculate_velocity(person_id)
        if velocity > self.velocity_threshold:
            confidence += 0.3 * min(1.0, velocity / 100)
        
        # Y 변화 기여도
        y_change = self._calculate_y_change(person_id)
        if y_change > 0:
            confidence += 0.2 * min(1.0, y_change / 100)
        
        # 지속 시간 기여도
        if person_id in self.fall_status and self.fall_status[person_id]:
            confidence += 0.2
        
        return min(0.95, confidence)
    
    def _cleanup_old_tracks(self):
        """
        오래된 추적 정보 제거
        """
        # 실제 구현에서는 타임스탬프 기반으로 처리
        # 여기서는 간단히 너무 많은 추적 제거
        if len(self.person_history) > 20:
            oldest = list(self.person_history.keys())[:5]
            for pid in oldest:
                del self.person_history[pid]
                self.fall_status.pop(pid, None)