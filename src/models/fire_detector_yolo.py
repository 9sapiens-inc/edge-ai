"""
YOLOv8 기반 화재 및 연기 감지 모듈 - 시간적 분석 강화 버전
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Deque
from ultralytics import YOLO
from collections import deque, defaultdict
import time

class FireDetectorYOLO:
    def __init__(self, model_path: str = 'weights/fire_smoke_best.pt'):
        """
        YOLOv8 기반 화재/연기 감지기 with 시간적 분석
        """
        # 기본 모델 로드
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"화재 감지 모델 로드 완료: {model_path}")
        print(f"감지 가능 클래스: {self.class_names}")
        
        # 신뢰도 임계값
        self.confidence_threshold = 0.4
        
        # 시간적 분석을 위한 프레임 버퍼
        self.frame_buffer_size = 10  # 최근 10프레임 저장
        self.frame_buffers = defaultdict(lambda: deque(maxlen=self.frame_buffer_size))
        
        # 연기 감지를 위한 시간적 분석 파라미터
        self.temporal_analysis = {
            'min_pixel_change': 5,      # 최소 픽셀 변화량 (픽셀값 범위: 0~255 (그레이스케일))
            'min_changed_ratio': 0.25,   # 최소 변화 픽셀 비율 (15%) - ROI(관심영역) 내 전체 픽셀의 p% 이상이 변해야 함
            'consistency_frames': 5,     # 일관성 확인 프레임 수 - 연기로 확정하기 위해 연속 7프레임 동안 일관된 패턴 필요
            'motion_variance_min': 30,   # 최소 모션 분산 (낮추면: 느리게 움직이는 연기도 감지, 높이면: 빠르게 움직이는 연기만 감지)
            'temporal_score_weight': 0.7 # 시간적 점수 가중치 (낮추면: 주로 색상/모양으로 판단, 높이면: 주로 움직임으로 판단)
        }
        
        # 객체별 시간적 추적
        self.smoke_candidates = defaultdict(lambda: {
            'scores': deque(maxlen=10),
            'temporal_scores': deque(maxlen=10),
            'last_update': 0,
            'confirmed': False
        })
        
        # 디버그 모드
        self.debug_mode = False
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        프레임에서 화재/연기 감지 with 시간적 분석
        """
        # 기본 YOLO 감지
        raw_detections = self._run_yolo_detection(frame)
        
        # 필터링된 감지 결과
        filtered_detections = []
        
        for det in raw_detections:
            if det['class_name'] == 'Fire' or 'fire' in det.get('original_class_name', '').lower():
                # 화재는 기본 검증만
                if self._validate_fire_detection(det, frame):
                    filtered_detections.append(det)
                    
            elif det['class_name'] == 'Smoke' or 'smoke' in det.get('original_class_name', '').lower():
                # 연기는 시간적 분석 포함한 강화된 검증
                if self._validate_smoke_with_temporal_analysis(det, frame):
                    filtered_detections.append(det)
        
        # 오래된 후보 정리
        self._cleanup_old_candidates()
        
        return filtered_detections
    
    def _run_yolo_detection(self, frame: np.ndarray) -> List[Dict]:
        """기본 YOLO 감지 실행"""
        detections = []
        
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        for r in results:
            if r.boxes is None:
                continue
                
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = self.class_names.get(class_id, 'unknown')
                
                if any(keyword in class_name.lower() for keyword in ['fire', 'smoke', '연기', '화재']):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf),
                        'class_name': 'Fire' if 'fire' in class_name.lower() else 'Smoke',
                        'original_class_name': class_name,
                        'class_id': class_id,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)
        
        return detections
    
    def _validate_fire_detection(self, detection: Dict, frame: np.ndarray) -> bool:
        """화재 감지 검증"""
        x1, y1, x2, y2 = detection['bbox']
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # HSV 색상 검증
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 화재 색상 마스크
        fire_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        fire_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
        
        fire_pixel_ratio = np.sum(fire_mask > 0) / (roi.shape[0] * roi.shape[1])
        brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        
        return fire_pixel_ratio > 0.3 and brightness > 150
    
    def _validate_smoke_with_temporal_analysis(self, detection: Dict, frame: np.ndarray) -> bool:
        """연기 감지 검증 - 시간적 분석 포함"""
        x1, y1, x2, y2 = detection['bbox']
        
        # 안전한 범위 체크
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        roi = frame[y1:y2, x1:x2]
        
        # 후보 ID 생성 (더 정밀하게 - 바운딩 박스 기반)
        candidate_id = f"{x1}_{y1}_{x2}_{y2}"
        
        # 1. 기본 연기 특성 검증
        basic_score = self._check_basic_smoke_properties(roi)
        
        # 2. 시간적 변화 분석
        temporal_score = self._analyze_temporal_changes(roi, candidate_id, detection['bbox'])
        
        # 3. 종합 점수 계산
        total_score = (1 - self.temporal_analysis['temporal_score_weight']) * basic_score + \
                      self.temporal_analysis['temporal_score_weight'] * temporal_score
        
        # 후보 정보 업데이트
        candidate = self.smoke_candidates[candidate_id]
        candidate['scores'].append(total_score)
        candidate['temporal_scores'].append(temporal_score)
        candidate['last_update'] = time.time()
        candidate['bbox'] = detection['bbox']  # 바운딩 박스 저장
        
        # 디버그 출력
        if self.debug_mode:
            print(f"\n[연기 검증] ID: {candidate_id}")
            print(f"  - 기본 점수: {basic_score:.3f}")
            print(f"  - 시간적 점수: {temporal_score:.3f}")
            print(f"  - 종합 점수: {total_score:.3f}")
            print(f"  - 가중치: {self.temporal_analysis['temporal_score_weight']}")
            print(f"  - 버퍼 크기: {len(self.frame_buffers[candidate_id])}")
        
        # 4. 최종 판단 - temporal_score_weight가 높을 때는 더 엄격하게
        min_frames_required = 3
        
        # 충분한 프레임이 수집되었을 때
        if len(candidate['temporal_scores']) >= min_frames_required:
            avg_temporal = np.mean(list(candidate['temporal_scores']))
            
            # temporal_score_weight가 높으면 시간적 점수를 엄격하게 평가
            if self.temporal_analysis['temporal_score_weight'] > 0.7:
                # 시간적 점수가 낮으면 즉시 거부
                if avg_temporal < 0.2:
                    if self.debug_mode:
                        print(f"  → 거부: 시간적 변화 부족 (avg_temporal: {avg_temporal:.3f})")
                    return False
                
                # 시간적 점수가 충분히 높아야만 승인
                if avg_temporal > 0.4 and total_score > 0.5:
                    candidate['confirmed'] = True
                    return True
            else:
                # 일반적인 평가
                avg_score = np.mean(list(candidate['scores']))
                if avg_temporal > 0.3 and avg_score > 0.5:
                    candidate['confirmed'] = True
                    return True
        
        # 초기 프레임에서는 거부 (temporal_score_weight가 높을 때)
        if self.temporal_analysis['temporal_score_weight'] > 0.7:
            if self.debug_mode:
                print(f"  → 대기: 프레임 부족 ({len(candidate['temporal_scores'])}/{min_frames_required})")
            return False
        
        # temporal_score_weight가 낮을 때만 기본 점수 사용
        return basic_score > 0.8  # 더 엄격한 임계값
    
    def _check_basic_smoke_properties(self, roi: np.ndarray) -> float:
        """기본 연기 특성 검증"""
        if roi.size == 0:
            return 0.0
        
        score = 0.0
        
        # 1. 색상 검증 (회색 계열)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 채도가 낮은지 확인
        mean_saturation = np.mean(hsv[:, :, 1])
        if mean_saturation < 60:
            score += 0.3
        
        # 2. 엣지 밀도 (연기는 부드러움)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        if edge_density < 0.2:
            score += 0.3
        
        # 3. 텍스처 복잡도
        texture_std = np.std(gray)
        if 15 < texture_std < 80:
            score += 0.2
        
        # 4. 밝기 분포
        brightness_mean = np.mean(gray)
        if 60 < brightness_mean < 200:
            score += 0.2
        
        return min(1.0, score)
    
    def _analyze_temporal_changes(self, roi: np.ndarray, candidate_id: str, bbox: List[int]) -> float:
        """시간적 변화 분석"""
        # ROI를 그레이스케일로 변환
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 고정 크기로 리사이즈 (모든 프레임 동일하게)
        fixed_size = (64, 64)  # 또는 (32, 32) for faster processing
        roi_resized = cv2.resize(roi_gray, fixed_size)
        
        # 프레임 버퍼에 추가
        buffer = self.frame_buffers[candidate_id]
        buffer.append(roi_resized.copy())
        
        if len(buffer) < 2:
            return 0.0  # 초기값을 0으로 (변화 없음)
        
        temporal_score = 0.0
        
        # 1. 프레임 간 차이 분석
        frame_diffs = []
        for i in range(1, len(buffer)):
            # 이제 모든 프레임이 같은 크기이므로 직접 비교 가능
            diff = cv2.absdiff(buffer[i], buffer[i-1])
            
            # 변화 픽셀 비율 계산
            changed_pixels = np.sum(diff > self.temporal_analysis['min_pixel_change'])
            change_ratio = changed_pixels / diff.size
            frame_diffs.append(change_ratio)
        
        if frame_diffs:
            avg_change_ratio = np.mean(frame_diffs)
            
            # 적절한 변화가 있는지 확인
            if self.temporal_analysis['min_changed_ratio'] < avg_change_ratio < 0.5:
                temporal_score += 0.4
        
        # 2. 시간에 따른 분산 분석
        if len(buffer) >= 3:
            try:
                # 모든 버퍼가 같은 크기인지 확인
                buffer_array = np.array(list(buffer))
                
                # 각 픽셀의 시간적 분산 계산
                temporal_variance = np.var(buffer_array, axis=0)
                mean_variance = np.mean(temporal_variance)
                
                if mean_variance > self.temporal_analysis['motion_variance_min']:
                    temporal_score += 0.3
            except Exception as e:
                if self.debug_mode:
                    print(f"[경고] 분산 계산 오류: {e}")
        
        # 3. 광학 흐름 분석 (선택적)
        if len(buffer) >= 2:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    buffer[-2], buffer[-1], None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # 흐름의 크기 계산
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                mean_magnitude = np.mean(magnitude)
                
                # 연기는 일정한 흐름을 가짐
                if 0.5 < mean_magnitude < 10:
                    temporal_score += 0.3
            except Exception as e:
                if self.debug_mode:
                    print(f"[경고] 광학 흐름 계산 오류: {e}")
        
        return min(1.0, temporal_score)
    
    def _cleanup_old_candidates(self):
        """오래된 후보 정리"""
        current_time = time.time()
        to_remove = []
        
        for candidate_id, info in self.smoke_candidates.items():
            if current_time - info['last_update'] > 5.0:  # 5초 이상 미감지
                to_remove.append(candidate_id)
        
        for candidate_id in to_remove:
            del self.smoke_candidates[candidate_id]
            if candidate_id in self.frame_buffers:
                del self.frame_buffers[candidate_id]
    
    def set_debug_mode(self, enabled: bool):
        """디버그 모드 설정"""
        self.debug_mode = enabled
        
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'temporal_params': self.temporal_analysis,
            'active_candidates': len(self.smoke_candidates),
            'confirmed_smoke': sum(1 for c in self.smoke_candidates.values() if c['confirmed'])
        }