"""
YOLOv8 기반 화재 및 연기 감지 모듈 - 디버깅 강화 버전
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import sys
import os

class FireDetectorYOLO:
    def __init__(self, model_path: str = 'weights/fire_smoke_best.pt'):
        """
        YOLOv8 기반 화재/연기 감지기
        """
        # 화재 감지 전용 YOLO 모델 로드
        try:
            self.model = YOLO(model_path)
            print(f"화재 감지 모델 로드 완료: {model_path}")
            
            # 클래스 이름 확인 - 상세 출력
            self.class_names = self.model.names
            print(f"감지 가능 클래스: {self.class_names}")
            print(f"클래스 ID와 이름 매핑:")
            for idx, name in self.class_names.items():
                print(f"  - ID {idx}: '{name}'")
            
        except Exception as e:
            print(f"화재 감지 모델 로드 실패: {e}")
            raise
        
        # 신뢰도 임계값
        self.confidence_threshold = 0.3  # 0.5에서 낮춤
        
        # 디버깅 모드
        self.debug_mode = True
        self.detection_count = {'fire': 0, 'smoke': 0, 'other': 0}
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        프레임에서 화재/연기 감지
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
                
                if self.debug_mode and len(boxes) > 0:
                    print(f"\n[디버그] 감지된 객체 수: {len(boxes)}")
                
                for box in boxes:
                    # 클래스 정보 추출
                    class_id = int(box.cls)
                    class_name = self.class_names.get(class_id, 'unknown')
                    confidence = float(box.conf)
                    
                    if self.debug_mode:
                        print(f"[디버그] 감지: ID={class_id}, Name='{class_name}', Conf={confidence:.3f}")
                    
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # 모든 가능한 화재/연기 관련 클래스 이름 체크
                    fire_keywords = ['fire', 'flame', '화재', '불']
                    smoke_keywords = ['smoke', '연기', 'smog']
                    
                    is_fire = any(keyword in class_name.lower() for keyword in fire_keywords)
                    is_smoke = any(keyword in class_name.lower() for keyword in smoke_keywords)
                    
                    if is_fire or is_smoke:
                        # 너무 작은 감지는 필터링 (조정 가능)
                        min_area = 300 if is_smoke else 500  # 연기는 더 작은 영역도 허용
                        
                        if area < min_area:
                            if self.debug_mode:
                                print(f"[디버그] 너무 작은 영역 무시: {area:.0f} < {min_area}")
                            continue
                        
                        # 감지 유형 결정
                        detection_type = 'Fire' if is_fire else 'Smoke'
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_name': detection_type,
                            'class_id': class_id,
                            'original_class_name': class_name,  # 원본 클래스 이름 저장
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                            'width': width,
                            'height': height,
                            'area': area
                        }
                        
                        detections.append(detection)
                        
                        # 통계 업데이트
                        if is_fire:
                            self.detection_count['fire'] += 1
                        else:
                            self.detection_count['smoke'] += 1
                    else:
                        self.detection_count['other'] += 1
                        if self.debug_mode and confidence > 0.5:
                            print(f"[디버그] 화재/연기 아님: '{class_name}' (conf={confidence:.3f})")
            
            # 주기적으로 통계 출력
            total_detections = sum(self.detection_count.values())
            if self.debug_mode and total_detections % 100 == 0 and total_detections > 0:
                print(f"\n[통계] 총 감지 수: {total_detections}")
                print(f"  - Fire: {self.detection_count['fire']}")
                print(f"  - Smoke: {self.detection_count['smoke']}")
                print(f"  - Other: {self.detection_count['other']}")
            
        except Exception as e:
            print(f"화재 감지 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    def update_confidence_threshold(self, threshold: float):
        """신뢰도 임계값 업데이트"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"화재 감지 신뢰도 임계값 변경: {self.confidence_threshold}")
    
    def set_debug_mode(self, enabled: bool):
        """디버그 모드 설정"""
        self.debug_mode = enabled
        print(f"디버그 모드: {'활성화' if enabled else '비활성화'}")
    
    def get_statistics(self) -> Dict:
        """감지 통계 반환"""
        return {
            'model_name': str(self.model.model) if hasattr(self.model, 'model') else 'unknown',
            'confidence_threshold': self.confidence_threshold,
            'classes': self.class_names,
            'detection_count': self.detection_count.copy(),
            'debug_mode': self.debug_mode
        }
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """감지 결과를 프레임에 시각화"""
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            original_name = det.get('original_class_name', class_name)
            
            # 색상 설정
            if class_name == 'Fire':
                color = (0, 0, 255)  # 빨간색
            else:  # Smoke
                color = (128, 128, 128)  # 회색
            
            # 바운딩 박스
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨
            label = f"{class_name}: {confidence:.2f}"
            if self.debug_mode and original_name != class_name:
                label += f" ({original_name})"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame