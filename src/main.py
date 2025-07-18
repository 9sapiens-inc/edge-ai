"""
CCTV 실시간 위험 감지 시스템 - 통합 버전 (단일/다중 카메라 지원)
"""
import cv2
import time
import argparse
import signal
import sys
import os
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Optional

# 프로젝트 모듈
from config.config import *
from utils.alert_manager import AlertManager
from utils.multi_camera_manager import MultiCameraManager
from models.yolo_detector import YOLODetector
from models.fire_detector import FireDetector
from models.fall_detector import FallDetector
from models.safety_detector import SafetyDetector

class DangerDetectionSystem:
    def __init__(self):
        print("="*60)
        print("CCTV 위험 감지 시스템 초기화 중...")
        print("="*60)
        
        # 활성화된 카메라 수 확인
        self.active_cameras = [cam for cam in CAMERAS if cam['enabled'] and cam['url']]
        self.is_single_camera = len(self.active_cameras) == 1
        
        if self.is_single_camera:
            print(f"단일 카메라 모드 - {self.active_cameras[0]['name']}")
        else:
            print(f"다중 카메라 모드 - {len(self.active_cameras)}대 활성화")
        
        # 컴포넌트 초기화
        self.alert_manager = AlertManager(ALERT_CONFIG['log_file'])
        self.multi_camera_manager = None
        
        # AI 모델
        self.yolo_detector = None        # 일반 객체 감지용 (사람, 차량 등)
        self.fire_detector = None        # 화재 전용 모델
        self.fall_detector = None        # 낙상 감지 (사람 감지 필요)
        self.safety_detector = None      # 안전모 감지 (사람 감지 필요)
        
        # 통계
        self.frame_counts = {}  # 카메라별 프레임 카운트
        self.start_time = time.time()
        
        # 스레드 풀 (다중 카메라용)
        self.executor = ThreadPoolExecutor(max_workers=max(4, len(self.active_cameras)))
        
        # 시그널 핸들러
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
            # 전용 화재 감지 모델이 있는지 확인
            fire_model_path = 'weights/fire_smoke_best.pt'
            if os.path.exists(fire_model_path):
                from models.fire_detector_yolo import FireDetectorYOLO
                self.fire_detector = FireDetectorYOLO(fire_model_path)
                
                # 시간적 분석 가중치 설정
                temporal_weight = DETECTION_CONFIG['fire_detection'].get('temporal_weight', 0.9)
                self.fire_detector.set_temporal_weight(temporal_weight)
                
                # 디버그 모드 활성화 (선택사항)
                self.fire_detector.set_debug_mode(True)
                
                print("✓ YOLOv8 화재 감지 전용 모델 로드 완료")
                print(f"  - 시간적 분석 가중치: {temporal_weight}")
                print(f"  - 모델 정보: {self.fire_detector.get_statistics()}")
            else:
                from models.fire_detector import FireDetector
                self.fire_detector = FireDetector(self.yolo_detector)
                print("✓ 색상 기반 화재 감지 모듈 준비 완료")
            
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
    
    def connect_cameras(self, custom_url: Optional[str] = None):
        """카메라 연결"""
        # 커스텀 URL이 제공된 경우 첫 번째 카메라 URL 업데이트
        if custom_url:
            if self.active_cameras:
                self.active_cameras[0]['url'] = custom_url
            else:
                self.active_cameras = [{
                    'id': 1,
                    'name': 'Custom Camera',
                    'url': custom_url,
                    'enabled': True,
                    'position': (0, 0)
                }]
            print(f"커스텀 URL 사용: {custom_url}")
        
        self.multi_camera_manager = MultiCameraManager(
            self.active_cameras,
            self.alert_manager,
            VIDEO_CONFIG
        )
        
        return self.multi_camera_manager.initialize_cameras()
    
    def process_camera(self, cam_id: int, frame: np.ndarray) -> Tuple[int, Dict]:
        """단일 카메라 프레임 처리"""
        detections = {
            'fire': [],
            'fall': [],
            'restricted_area': [],
            'no_helmet': []
        }
        
        try:
            # 프레임 리사이즈
            resized = cv2.resize(frame, 
                (VIDEO_CONFIG['resize_width'], VIDEO_CONFIG['resize_height'])
            )
            
            # 1. 화재/연기 감지
            if DETECTION_CONFIG['fire_detection']['enabled']:
                fire_detections = self.fire_detector.detect(resized)
                for det in fire_detections:
                    if det['confidence'] >= DETECTION_CONFIG['fire_detection']['min_confidence']:
                        detections['fire'].append(det)
                        # 단일 카메라 모드에서는 카메라 ID 표시 안 함
                        # 화재/연기 구분하여 알림
                        alert_prefix = "" if self.is_single_camera else f"[카메라 {cam_id}] "
                        
                        if det['class_name'] == 'Fire':
                            alert_type = f'fire_cam{cam_id}'
                            alert_message = f"{alert_prefix}화재 감지됨"
                        elif det['class_name'] == 'Smoke':
                            alert_type = f'smoke_cam{cam_id}'
                            alert_message = f"{alert_prefix}연기 감지됨"
                        else:
                            alert_type = f'fire_cam{cam_id}'
                            alert_message = f"{alert_prefix}{det['class_name']} 감지됨"
                        
                        self.alert_manager.send_alert(
                            alert_type,
                            alert_message,
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
                        alert_prefix = "" if self.is_single_camera else f"[카메라 {cam_id}] "
                        self.alert_manager.send_alert(
                            f'fall_cam{cam_id}',
                            f"{alert_prefix}낙상 감지 - Person ID: {det.get('person_id', 'Unknown')}",
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
                        alert_prefix = "" if self.is_single_camera else f"[카메라 {cam_id}] "
                        self.alert_manager.send_alert(
                            f'restricted_cam{cam_id}',
                            f"{alert_prefix}제한구역 침입 감지 - Zone {det['zone_id']+1}",
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
                        alert_prefix = "" if self.is_single_camera else f"[카메라 {cam_id}] "
                        self.alert_manager.send_alert(
                            f'helmet_cam{cam_id}',
                            f"{alert_prefix}안전모 미착용 감지",
                            location={'x': det['bbox'][0], 'y': det['bbox'][1]},
                            confidence=det['confidence'],
                            cooldown=DETECTION_CONFIG['helmet_detection']['alert_cooldown']
                        )
            
        except Exception as e:
            print(f"[카메라 {cam_id}] 처리 중 오류: {e}")
        
        return cam_id, detections
    
    def visualize_detections_on_frame(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """프레임에 감지 결과 시각화"""
        vis_frame = frame.copy()
        
        # 제한구역 표시 (단일 카메라 모드일 때만)
        if self.is_single_camera and DETECTION_CONFIG['restricted_area']['enabled']:
            vis_frame = self.safety_detector.visualize_zones(vis_frame)
        
        # 화재/연기 감지 표시 - Fire와 Smoke 구분
        for det in detections['fire']:
            if det.get('class_name') == 'Smoke':
                # 연기는 주황색으로 표시
                vis_frame = self.yolo_detector.draw_detections(
                    vis_frame, [det], color=(0, 165, 255), thickness=3  # 주황색
                )
            else:
                # 화재는 빨간색으로 표시
                vis_frame = self.yolo_detector.draw_detections(
                    vis_frame, [det], color=(0, 0, 255), thickness=3  # 빨간색
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
        
        return vis_frame
    
    def add_single_camera_overlay(self, frame: np.ndarray, cam_id: int):
        """단일 카메라용 오버레이"""
        height, width = frame.shape[:2]
        
        # 상단 정보 바
        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
        
        # FPS 정보
        fps = self.multi_camera_manager.video_streams[cam_id].get_fps()
        elapsed = time.time() - self.start_time
        total_frames = self.frame_counts.get(cam_id, 0)
        avg_fps = total_frames / elapsed if elapsed > 0 else 0
        
        info_text = f"FPS: {fps:.1f} (Avg: {avg_fps:.1f}) | "
        info_text += f"Frame: {total_frames} | "
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
                # 단일 카메라에서는 _cam1 접미사 제거
                display_type = alert_type.replace('_cam1', '')
                alert_text += f"{display_type}: {count} | "
            
            cv2.putText(frame, alert_text[:100], (10, height-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if stats['active_alerts']:
                active_text = f"Active: {', '.join(stats['active_alerts'])}"
                cv2.putText(frame, active_text, (10, height-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def run(self, display=True, save_output=False):
        """메인 실행 루프"""
        if self.is_single_camera:
            print("\n단일 카메라 시스템 시작...")
            print("종료: Q | 스크린샷: S | 통계 리셋: R\n")
        else:
            print("\n다중 카메라 시스템 시작...")
            print("종료: Q | 스크린샷: S | 통계 리셋: R | 카메라 토글: 1-4\n")
        
        # 비디오 저장 설정
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if self.is_single_camera:
                output_filename = f"output_{timestamp}.mp4"
                output_size = (VIDEO_CONFIG['resize_width'], VIDEO_CONFIG['resize_height'])
            else:
                output_filename = f"multi_output_{timestamp}.mp4"
                output_size = (self.multi_camera_manager.display_width, 
                             self.multi_camera_manager.display_height)
            
            video_writer = cv2.VideoWriter(output_filename, fourcc, 20.0, output_size)
            print(f"비디오 저장: {output_filename}")
        
        try:
            while True:
                # 모든 카메라에서 프레임 읽기
                frames = self.multi_camera_manager.read_all_frames()
                
                if not frames:
                    print("활성 카메라가 없습니다.")
                    time.sleep(1)
                    continue
                
                # 병렬로 각 카메라 처리
                all_detections = {}
                futures = []
                
                for cam_id, frame in frames.items():
                    # 프레임 카운트 업데이트
                    self.frame_counts[cam_id] = self.frame_counts.get(cam_id, 0) + 1
                    
                    # 프레임 스킵
                    if self.frame_counts[cam_id] % VIDEO_CONFIG['frame_skip'] != 0:
                        continue
                    
                    # 비동기 처리 제출
                    future = self.executor.submit(self.process_camera, cam_id, frame)
                    futures.append(future)
                
                # 결과 수집
                for future in as_completed(futures):
                    cam_id, detections = future.result()
                    all_detections[cam_id] = detections
                
                # 시각화된 프레임 생성
                visualized_frames = {}
                for cam_id, frame in frames.items():
                    if cam_id in all_detections:
                        vis_frame = self.visualize_detections_on_frame(
                            frame, all_detections[cam_id]
                        )
                        visualized_frames[cam_id] = vis_frame
                    else:
                        visualized_frames[cam_id] = frame
                
                # 디스플레이 처리
                if self.is_single_camera:
                    # 단일 카메라: 전체 화면 사용
                    cam_id = list(visualized_frames.keys())[0]
                    display_frame = visualized_frames[cam_id]
                    self.add_single_camera_overlay(display_frame, cam_id)
                else:
                    # 다중 카메라: 그리드 뷰
                    display_frame = self.multi_camera_manager.create_multi_view(
                        visualized_frames, all_detections
                    )
                
                if display:
                    window_name = 'CCTV Danger Detection' if self.is_single_camera else 'Multi-Camera Danger Detection'
                    cv2.imshow(window_name, display_frame)
                    
                    # 키 입력 처리
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        break
                    elif key == ord('r') or key == ord('R'):
                        self.alert_manager.reset_statistics()
                        print("통계가 초기화되었습니다.")
                    elif key == ord('s') or key == ord('S'):
                        prefix = "screenshot" if self.is_single_camera else "multi_screenshot"
                        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"스크린샷 저장: {filename}")
                    elif not self.is_single_camera and ord('1') <= key <= ord('4'):
                        # 다중 카메라 모드에서만 카메라 토글
                        cam_id = key - ord('0')
                        self.multi_camera_manager.toggle_camera(cam_id)
                
                if save_output and video_writer:
                    video_writer.write(display_frame)
                
                # 주기적 상태 출력
                total_frames = sum(self.frame_counts.values())
                if total_frames % (30 * 10) == 0:  # 10초마다
                    self.print_status()
                    
        except Exception as e:
            print(f"\n오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup(video_writer)
    
    def print_status(self):
        """시스템 상태 출력"""
        stats = self.alert_manager.get_statistics()
        active_cams = len(self.multi_camera_manager.video_streams)
        total_frames = sum(self.frame_counts.values())
        
        if self.is_single_camera:
            print(f"\n[상태] 프레임: {total_frames} | "
                  f"총 알림: {stats['total_alerts']} | "
                  f"활성 알림: {len(stats['active_alerts'])}")
        else:
            print(f"\n[상태] 활성 카메라: {active_cams} | "
                  f"총 프레임: {total_frames} | "
                  f"총 알림: {stats['total_alerts']} | "
                  f"활성 알림: {len(stats['active_alerts'])}")
    
    def cleanup(self, video_writer=None):
        """리소스 정리"""
        print("\n시스템 종료 중...")
        
        if video_writer:
            video_writer.release()
        
        if self.multi_camera_manager:
            self.multi_camera_manager.stop_all()
        
        self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()
        
        # 최종 통계
        print("\n=== 최종 통계 ===")
        stats = self.alert_manager.get_statistics()
        total_frames = sum(self.frame_counts.values())
        print(f"총 처리 프레임: {total_frames}")
        print(f"총 실행 시간: {time.time() - self.start_time:.1f}초")
        print(f"총 알림 횟수: {stats['total_alerts']}")
        
        if stats['alert_breakdown']:
            print("\n알림 유형별 통계:")
            for alert_type, count in stats['alert_breakdown'].items():
                if self.is_single_camera:
                    # 단일 카메라에서는 _cam1 접미사 제거
                    display_type = alert_type.replace('_cam1', '')
                else:
                    display_type = alert_type
                print(f"  - {display_type}: {count}회")
        
        if self.frame_counts:
            if self.is_single_camera:
                print(f"\n총 프레임 수: {total_frames}")
            else:
                print("\n카메라별 프레임 수:")
                for cam_id, count in self.frame_counts.items():
                    print(f"  - 카메라 {cam_id}: {count} 프레임")
        
        print("\n시스템이 안전하게 종료되었습니다.")
    
    def signal_handler(self, sig, frame):
        """시그널 핸들러"""
        print("\n\n종료 신호를 받았습니다...")
        self.cleanup()
        sys.exit(0)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CCTV 위험 감지 시스템')
    parser.add_argument('--no-display', action='store_true',
                       help='화면 표시 없이 실행')
    parser.add_argument('--save-output', action='store_true',
                       help='감지 결과를 비디오로 저장')
    parser.add_argument('--rtsp-url', type=str,
                       help='RTSP 스트림 URL (단일 카메라 모드)')
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = DangerDetectionSystem()
    
    # 모델 초기화
    if not system.initialize_models():
        print("모델 초기화 실패. 프로그램을 종료합니다.")
        return
    
    # 카메라 연결
    if not system.connect_cameras(custom_url=args.rtsp_url):
        print("카메라 연결 실패. 프로그램을 종료합니다.")
        return
    
    # 메인 루프 실행
    system.run(display=not args.no_display, save_output=args.save_output)

if __name__ == "__main__":
    main()