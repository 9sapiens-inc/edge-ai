"""
CCTV 실시간 위험 감지 시스템 메인 프로그램
"""
import cv2
import time
import argparse
import signal
import sys
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List, Tuple, Optional

# 프로젝트 모듈
from config.config import *
from utils.video_stream import VideoStream
from utils.alert_manager import AlertManager
from utils.multi_camera_manager import MultiCameraManager
from models.yolo_detector import YOLODetector
from models.fire_detector import FireDetector
from models.fall_detector import FallDetector
from models.safety_detector import SafetyDetector

class DangerDetectionSystem:
    def __init__(self, multi_camera=False):
        print("="*60)
        print("CCTV 실시간 위험 감지 시스템 초기화 중...")
        print("="*60)
        
        # 모드 설정
        self.multi_camera_mode = multi_camera
        
        # 컴포넌트 초기화
        self.video_stream = None  # 단일 카메라 모드용
        self.multi_camera_manager = None  # 다중 카메라 모드용
        self.alert_manager = AlertManager(ALERT_CONFIG['log_file'])
        self.yolo_detector = None
        self.fire_detector = None
        self.fall_detector = None
        self.safety_detector = None
        
        # 통계
        self.frame_count = 0
        self.start_time = time.time()
        
        # 다중 카메라 처리용 스레드 풀
        if multi_camera:
            self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def initialize_models(self):
        """AI 모델 초기화"""
        try:
            print("\n[1/4] YOLO 모델 로딩 중...")
            self.yolo_detector = YOLODetector(
                model_path=MODEL_CONFIG['yolo_model'],
                device=MODEL_CONFIG['device'],
                conf_threshold=MODEL_CONFIG['confidence_threshold'],
                iou_threshold=MODEL_CONFIG['iou_threshold']
            )
            print("✓ YOLO 모델 로드 완료")
            
            print("[2/4] 화재 감지 모듈 초기화 중...")
            self.fire_detector = FireDetector(self.yolo_detector)
            print("✓ 화재 감지 모듈 준비 완료")
            
            print("[3/4] 낙상 감지 모듈 초기화 중...")
            self.fall_detector = FallDetector(self.yolo_detector)
            print("✓ 낙상 감지 모듈 준비 완료")
            
            print("[4/4] 안전 장비 감지 모듈 초기화 중...")
            self.safety_detector = SafetyDetector(
                self.yolo_detector,
                restricted_zones=DETECTION_CONFIG['restricted_area']['restricted_zones']
            )
            print("✓ 안전 장비 감지 모듈 준비 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            return False
    
    def connect_camera(self, rtsp_url=None):
        """카메라 연결"""
        if self.multi_camera_mode:
            # 다중 카메라 모드
            print("\n다중 카메라 모드로 시작합니다...")
            self.multi_camera_manager = MultiCameraManager(
                CAMERAS, 
                self.alert_manager,
                VIDEO_CONFIG
            )
            return self.multi_camera_manager.initialize_cameras()
        else:
            # 단일 카메라 모드
            url = rtsp_url or RTSP_URL
            print(f"\n카메라 연결 중: {url}")
            self.video_stream = VideoStream(url, VIDEO_CONFIG['buffer_size'])
            
            if self.video_stream.start():
                print("✓ 카메라 연결 성공")
                time.sleep(2)  # 스트림 안정화 대기
                return True
            else:
                print("❌ 카메라 연결 실패")
                return False
    
    def process_frame(self, frame: np.ndarray):
        """프레임 처리 및 위험 감지"""
        detections = {
            'fire': [],
            'fall': [],
            'restricted_area': [],
            'no_helmet': []
        }
        
        # 프레임 리사이즈 (성능 향상)
        resized = self.video_stream.resize_frame(
            frame, 
            VIDEO_CONFIG['resize_width'],
            VIDEO_CONFIG['resize_height']
        )
        
        # 1. 화재/연기 감지
        if DETECTION_CONFIG['fire_detection']['enabled']:
            fire_detections = self.fire_detector.detect(resized)
            for det in fire_detections:
                if det['confidence'] >= DETECTION_CONFIG['fire_detection']['min_confidence']:
                    detections['fire'].append(det)
                    self.alert_manager.send_alert(
                        'fire',
                        f"화재 또는 연기 감지됨 - {det['class_name']}",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['confidence'],
                        cooldown=DETECTION_CONFIG['fire_detection']['alert_cooldown']
                    )
        
        # 2. 낙상 감지
        if DETECTION_CONFIG['fall_detection']['enabled']:
            fall_detections = self.fall_detector.detect(resized)
            for det in fall_detections:
                if det['fall_confidence'] >= DETECTION_CONFIG['fall_detection']['min_confidence']:
                    detections['fall'].append(det)
                    self.alert_manager.send_alert(
                        'fall',
                        f"낙상 감지 - Person ID: {det.get('person_id', 'Unknown')}",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['fall_confidence'],
                        cooldown=DETECTION_CONFIG['fall_detection']['alert_cooldown']
                    )
        
        # 3. 제한구역 침입 감지
        if DETECTION_CONFIG['restricted_area']['enabled']:
            intrusion_detections = self.safety_detector.detect_restricted_area_intrusion(resized)
            for det in intrusion_detections:
                if det['confidence'] >= DETECTION_CONFIG['restricted_area']['min_confidence']:
                    detections['restricted_area'].append(det)
                    self.alert_manager.send_alert(
                        'restricted_area',
                        f"제한구역 침입 감지 - Zone {det['zone_id']+1}",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['confidence'],
                        cooldown=DETECTION_CONFIG['restricted_area']['alert_cooldown']
                    )
        
        # 4. 안전모 미착용 감지
        if DETECTION_CONFIG['helmet_detection']['enabled']:
            helmet_detections = self.safety_detector.detect_helmet(resized)
            for det in helmet_detections:
                if det['class_name'] == 'no_helmet' and det['confidence'] >= DETECTION_CONFIG['helmet_detection']['min_confidence']:
                    detections['no_helmet'].append(det)
                    self.alert_manager.send_alert(
                        'no_helmet',
                        "안전모 미착용 감지",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['confidence'],
                        cooldown=DETECTION_CONFIG['helmet_detection']['alert_cooldown']
                    )
        
        return detections, resized
        """단일 카메라 프레임 처리 (다중 카메라용)"""
        detections = {
            'fire': [],
            'fall': [],
            'restricted_area': [],
            'no_helmet': []
        }
        
        # 프레임 리사이즈
        resized = cv2.resize(frame, 
            (VIDEO_CONFIG['resize_width'], VIDEO_CONFIG['resize_height'])
        )
        
        # 각 감지 수행 (process_frame과 동일한 로직)
        if DETECTION_CONFIG['fire_detection']['enabled']:
            fire_detections = self.fire_detector.detect(resized)
            for det in fire_detections:
                if det['confidence'] >= DETECTION_CONFIG['fire_detection']['min_confidence']:
                    detections['fire'].append(det)
                    self.alert_manager.send_alert(
                        f'fire_cam{cam_id}',
                        f"[카메라 {cam_id}] 화재 또는 연기 감지됨 - {det['class_name']}",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['confidence'],
                        cooldown=DETECTION_CONFIG['fire_detection']['alert_cooldown']
                    )
        
        # 나머지 감지도 동일하게 처리...
        # (낙상, 제한구역, 안전모)
        
        return detections
        """프레임 처리 및 위험 감지"""
        detections = {
            'fire': [],
            'fall': [],
            'restricted_area': [],
            'no_helmet': []
        }
        
        # 프레임 리사이즈 (성능 향상)
        resized = self.video_stream.resize_frame(
            frame, 
            VIDEO_CONFIG['resize_width'],
            VIDEO_CONFIG['resize_height']
        )
        
        # 1. 화재/연기 감지
        if DETECTION_CONFIG['fire_detection']['enabled']:
            fire_detections = self.fire_detector.detect(resized)
            for det in fire_detections:
                if det['confidence'] >= DETECTION_CONFIG['fire_detection']['min_confidence']:
                    detections['fire'].append(det)
                    self.alert_manager.send_alert(
                        'fire',
                        f"화재 또는 연기 감지됨 - {det['class_name']}",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['confidence'],
                        cooldown=DETECTION_CONFIG['fire_detection']['alert_cooldown']
                    )
        
        # 2. 낙상 감지
        if DETECTION_CONFIG['fall_detection']['enabled']:
            fall_detections = self.fall_detector.detect(resized)
            for det in fall_detections:
                if det['fall_confidence'] >= DETECTION_CONFIG['fall_detection']['min_confidence']:
                    detections['fall'].append(det)
                    self.alert_manager.send_alert(
                        'fall',
                        f"낙상 감지 - Person ID: {det.get('person_id', 'Unknown')}",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['fall_confidence'],
                        cooldown=DETECTION_CONFIG['fall_detection']['alert_cooldown']
                    )
        
        # 3. 제한구역 침입 감지
        if DETECTION_CONFIG['restricted_area']['enabled']:
            intrusion_detections = self.safety_detector.detect_restricted_area_intrusion(resized)
            for det in intrusion_detections:
                if det['confidence'] >= DETECTION_CONFIG['restricted_area']['min_confidence']:
                    detections['restricted_area'].append(det)
                    self.alert_manager.send_alert(
                        'restricted_area',
                        f"제한구역 침입 감지 - Zone {det['zone_id']+1}",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['confidence'],
                        cooldown=DETECTION_CONFIG['restricted_area']['alert_cooldown']
                    )
        
        # 4. 안전모 미착용 감지
        if DETECTION_CONFIG['helmet_detection']['enabled']:
            helmet_detections = self.safety_detector.detect_helmet(resized)
            for det in helmet_detections:
                if det['class_name'] == 'no_helmet' and det['confidence'] >= DETECTION_CONFIG['helmet_detection']['min_confidence']:
                    detections['no_helmet'].append(det)
                    self.alert_manager.send_alert(
                        'no_helmet',
                        "안전모 미착용 감지",
                        location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                        confidence=det['confidence'],
                        cooldown=DETECTION_CONFIG['helmet_detection']['alert_cooldown']
                    )
        
        return detections, resized
    
    def visualize_detections(self, frame, detections):
        """감지 결과 시각화"""
        vis_frame = frame.copy()
        
        # 제한구역 표시
        if DETECTION_CONFIG['restricted_area']['enabled']:
            vis_frame = self.safety_detector.visualize_zones(vis_frame)
        
        # 화재 감지 표시 (빨간색)
        for det in detections['fire']:
            vis_frame = self.yolo_detector.draw_detections(
                vis_frame, [det], color=(0, 0, 255), thickness=3
            )
        
        # 낙상 감지 표시 (파란색)
        for det in detections['fall']:
            vis_frame = self.yolo_detector.draw_detections(
                vis_frame, [det], color=(255, 0, 0), thickness=3
            )
        
        # 제한구역 침입 표시 (노란색)
        for det in detections['restricted_area']:
            vis_frame = self.yolo_detector.draw_detections(
                vis_frame, [det], color=(0, 255, 255), thickness=3
            )
        
        # 안전모 미착용 표시 (보라색)
        for det in detections['no_helmet']:
            vis_frame = self.yolo_detector.draw_detections(
                vis_frame, [det], color=(255, 0, 255), thickness=3
            )
        
        # 정보 오버레이 추가
        self.add_info_overlay(vis_frame)
        
        return vis_frame
    
    def add_info_overlay(self, frame):
        """정보 오버레이 추가"""
        height, width = frame.shape[:2]
        
        # 상단 정보 바
        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
        
        # FPS 정보
        fps = self.video_stream.get_fps()
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        info_text = f"FPS: {fps:.1f} (Avg: {avg_fps:.1f}) | "
        info_text += f"Frame: {self.frame_count} | "
        info_text += f"Time: {datetime.now().strftime('%H:%M:%S')}"
        
        cv2.putText(frame, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 통계 정보
        stats = self.alert_manager.get_statistics()
        if stats['total_alerts'] > 0:
            # 하단 정보 바
            cv2.rectangle(frame, (0, height-60), (width, height), (0, 0, 0), -1)
            
            alert_text = f"Total Alerts: {stats['total_alerts']} | "
            for alert_type, count in stats['alert_breakdown'].items():
                alert_text += f"{alert_type}: {count} | "
            
            cv2.putText(frame, alert_text[:100], (10, height-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if stats['active_alerts']:
                active_text = f"Active: {', '.join(stats['active_alerts'])}"
                cv2.putText(frame, active_text, (10, height-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def run(self, display=True, save_output=False):
        """메인 실행 루프"""
        print("\n시스템 시작...")
        print("종료하려면 Ctrl+C를 누르세요\n")
        
        if self.multi_camera_mode:
            self.run_multi_camera(display, save_output)
        else:
            self.run_single_camera(display, save_output)
    
    def run_single_camera(self, display=True, save_output=False):
        """단일 카메라 모드 실행"""
        
        # 비디오 저장 설정
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_writer = cv2.VideoWriter(
                output_filename,
                fourcc,
                20.0,
                (VIDEO_CONFIG['resize_width'], VIDEO_CONFIG['resize_height'])
            )
            print(f"비디오 저장: {output_filename}")
        
        try:
            while self.video_stream.is_running():
                # 프레임 읽기
                ret, frame = self.video_stream.read()
                if not ret:
                    continue
                
                self.frame_count += 1
                
                # 프레임 스킵 (성능 최적화)
                if self.frame_count % VIDEO_CONFIG['frame_skip'] != 0:
                    continue
                
                # 위험 감지 처리
                detections, processed_frame = self.process_frame(frame)
                
                # 시각화
                if display or save_output:
                    vis_frame = self.visualize_detections(processed_frame, detections)
                    
                    if display:
                        cv2.imshow('CCTV Danger Detection', vis_frame)
                        
                        # 키 입력 처리
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('r'):
                            # 통계 리셋
                            self.alert_manager.reset_statistics()
                            print("통계가 초기화되었습니다.")
                        elif key == ord('s'):
                            # 스크린샷 저장
                            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(filename, vis_frame)
                            print(f"스크린샷 저장: {filename}")
                    
                    if save_output and video_writer:
                        video_writer.write(vis_frame)
                
                # 주기적 상태 출력 (10초마다)
                if self.frame_count % (30 * 10) == 0:  # 30fps 기준
                    self.print_status()
                    
        except Exception as e:
            print(f"\n오류 발생: {e}")
            
        finally:
            self.cleanup(video_writer)
    
    def print_status(self):
        """시스템 상태 출력"""
        stats = self.alert_manager.get_statistics()
        print(f"\n[상태] 프레임: {self.frame_count} | "
              f"총 알림: {stats['total_alerts']} | "
              f"활성 알림: {len(stats['active_alerts'])}")
    
    def cleanup(self, video_writer=None):
        """리소스 정리"""
        print("\n시스템 종료 중...")
        
        if video_writer:
            video_writer.release()
            
        if self.video_stream:
            self.video_stream.stop()
            
        cv2.destroyAllWindows()
        
        # 최종 통계 출력
        print("\n=== 최종 통계 ===")
        stats = self.alert_manager.get_statistics()
        print(f"총 처리 프레임: {self.frame_count}")
        print(f"총 실행 시간: {time.time() - self.start_time:.1f}초")
        print(f"총 알림 횟수: {stats['total_alerts']}")
        
        if stats['alert_breakdown']:
            print("\n알림 유형별 통계:")
            for alert_type, count in stats['alert_breakdown'].items():
                print(f"  - {alert_type}: {count}회")
        
        print("\n시스템이 안전하게 종료되었습니다.")
    
    def signal_handler(self, sig, frame):
        """시그널 핸들러"""
        print("\n\n종료 신호를 받았습니다...")
        self.cleanup()
        sys.exit(0)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CCTV 실시간 위험 감지 시스템')
    parser.add_argument('--no-display', action='store_true', 
                       help='화면 표시 없이 실행')
    parser.add_argument('--save-output', action='store_true',
                       help='감지 결과를 비디오로 저장')
    parser.add_argument('--rtsp-url', type=str, default=RTSP_URL,
                       help='RTSP 스트림 URL')
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = DangerDetectionSystem()
    
    # URL 업데이트
    rtsp_url = args.rtsp_url
    if rtsp_url != RTSP_URL:
        print(f"RTSP URL 변경: {rtsp_url}")
    
    # 모델 초기화
    if not system.initialize_models():
        print("모델 초기화 실패. 프로그램을 종료합니다.")
        return
    
    # 카메라 연결
    if not system.connect_camera(rtsp_url):
        print("카메라 연결 실패. 프로그램을 종료합니다.")
        return
    
    # 메인 루프 실행
    system.run(display=not args.no_display, save_output=args.save_output)

if __name__ == "__main__":
    main()