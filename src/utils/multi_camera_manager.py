"""
Multi-camera management system
"""
import cv2
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import time

from .video_stream import VideoStream
from .alert_manager import AlertManager

class MultiCameraManager:
    def __init__(self, cameras: List[Dict], alert_manager: AlertManager, 
                 display_config: Dict):
        self.cameras = cameras
        self.alert_manager = alert_manager
        self.display_config = display_config
        
        # 각 카메라별 비디오 스트림
        self.video_streams = {}
        
        # 각 카메라별 상태
        self.camera_status = {}
        
        # 각 카메라별 마지막 프레임
        self.last_frames = {}
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=len(cameras))
        
        # 디스플레이 설정
        self.grid_width = display_config.get('resize_width', 640)
        self.grid_height = display_config.get('resize_height', 480)
        self.spacing = display_config.get('grid_spacing', 10)
        
        # 전체 디스플레이 크기 계산
        self._calculate_display_size()
        
    def _calculate_display_size(self):
        """디스플레이 크기 계산"""
        active_cameras = [cam for cam in self.cameras if cam['enabled']]
        
        if not active_cameras:
            self.display_width = self.grid_width
            self.display_height = self.grid_height
            return
        
        # 최대 grid 위치 찾기
        max_col = max(cam['position'][0] for cam in active_cameras) + 1
        max_row = max(cam['position'][1] for cam in active_cameras) + 1
        
        self.display_width = (self.grid_width * max_col) + (self.spacing * (max_col - 1))
        self.display_height = (self.grid_height * max_row) + (self.spacing * (max_row - 1))
        
    def initialize_cameras(self) -> bool:
        """모든 카메라 초기화"""
        success_count = 0
        
        for camera in self.cameras:
            if not camera['enabled'] or not camera['url']:
                continue
                
            cam_id = camera['id']
            print(f"\n[카메라 {cam_id}] {camera['name']} 연결 중...")
            
            # 비디오 스트림 생성
            stream = VideoStream(camera['url'], buffer_size=10)
            
            if stream.start():
                self.video_streams[cam_id] = stream
                self.camera_status[cam_id] = 'connected'
                self.last_frames[cam_id] = None
                print(f"✓ [카메라 {cam_id}] 연결 성공")
                success_count += 1
            else:
                self.camera_status[cam_id] = 'disconnected'
                print(f"❌ [카메라 {cam_id}] 연결 실패")
        
        print(f"\n총 {success_count}/{len([c for c in self.cameras if c['enabled']])}개 카메라 연결됨")
        return success_count > 0
    
    def read_frame(self, cam_id: int) -> Tuple[bool, Optional[np.ndarray]]:
        """특정 카메라에서 프레임 읽기"""
        if cam_id not in self.video_streams:
            return False, None
            
        stream = self.video_streams[cam_id]
        ret, frame = stream.read()
        
        if ret:
            self.last_frames[cam_id] = frame
            self.camera_status[cam_id] = 'connected'
        else:
            self.camera_status[cam_id] = 'reconnecting'
            
        return ret, frame
    
    def read_all_frames(self) -> Dict[int, np.ndarray]:
        """모든 카메라에서 프레임 읽기"""
        frames = {}
        
        for cam_id in self.video_streams:
            ret, frame = self.read_frame(cam_id)
            if ret and frame is not None:
                # 크기 조정
                frame = cv2.resize(frame, (self.grid_width, self.grid_height))
                frames[cam_id] = frame
            elif self.last_frames.get(cam_id) is not None:
                # 마지막 프레임 사용
                frames[cam_id] = self.last_frames[cam_id]
                
        return frames
    
    def create_multi_view(self, frames: Dict[int, np.ndarray], 
                         detections: Dict[int, Dict] = None) -> np.ndarray:
        """다중 카메라 뷰 생성"""
        # 전체 캔버스 생성
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        canvas.fill(30)  # 어두운 회색 배경
        
        for camera in self.cameras:
            if not camera['enabled']:
                continue
                
            cam_id = camera['id']
            col, row = camera['position']
            
            # 그리드 위치 계산
            x = col * (self.grid_width + self.spacing)
            y = row * (self.grid_height + self.spacing)
            
            if cam_id in frames:
                # 프레임 배치
                frame = frames[cam_id].copy()
                
                # 카메라 정보 오버레이
                self._add_camera_overlay(frame, camera, detections.get(cam_id) if detections else None)
                
                # 캔버스에 프레임 배치
                canvas[y:y+self.grid_height, x:x+self.grid_width] = frame
            else:
                # 연결 끊김 표시
                self._draw_disconnected_view(canvas, x, y, camera)
        
        # 전체 시스템 정보 오버레이
        self._add_system_overlay(canvas)
        
        return canvas
    
    def _add_camera_overlay(self, frame: np.ndarray, camera: Dict, 
                           detections: Optional[Dict] = None):
        """카메라별 정보 오버레이"""
        height, width = frame.shape[:2]
        
        # 상단 정보 바
        overlay_height = 30
        cv2.rectangle(frame, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        
        # 카메라 이름과 상태
        status = self.camera_status.get(camera['id'], 'unknown')
        status_color = (0, 255, 0) if status == 'connected' else (0, 0, 255)
        
        text = f"{camera['name']} - Camera {camera['id']}"
        cv2.putText(frame, text, (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 상태 표시
        cv2.circle(frame, (width - 20, 15), 8, status_color, -1)
        
        # 감지 정보
        if detections:
            total_detections = sum(len(d) for d in detections.values())
            if total_detections > 0:
                alert_text = f"Alerts: {total_detections}"
                cv2.putText(frame, alert_text, (5, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def _draw_disconnected_view(self, canvas: np.ndarray, x: int, y: int, 
                               camera: Dict):
        """연결 끊김 뷰 그리기"""
        # 어두운 사각형
        cv2.rectangle(canvas, (x, y), (x + self.grid_width, y + self.grid_height),
                     (50, 50, 50), -1)
        
        # 테두리
        cv2.rectangle(canvas, (x, y), (x + self.grid_width, y + self.grid_height),
                     (100, 100, 100), 2)
        
        # 텍스트
        text1 = f"{camera['name']}"
        text2 = "Disconnected"
        
        # 텍스트 중앙 정렬
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        
        (tw1, th1), _ = cv2.getTextSize(text1, font, scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(text2, font, scale, thickness)
        
        cx = x + self.grid_width // 2
        cy = y + self.grid_height // 2
        
        cv2.putText(canvas, text1, (cx - tw1//2, cy - 10),
                   font, scale, (200, 200, 200), thickness)
        cv2.putText(canvas, text2, (cx - tw2//2, cy + 20),
                   font, scale, (100, 100, 255), thickness)
    
    def _add_system_overlay(self, canvas: np.ndarray):
        """전체 시스템 정보 오버레이"""
        height, width = canvas.shape[:2]
        
        # 하단 상태 바
        bar_height = 40
        cv2.rectangle(canvas, (0, height - bar_height), (width, height),
                     (40, 40, 40), -1)
        
        # 시스템 정보
        connected = sum(1 for status in self.camera_status.values() 
                       if status == 'connected')
        total = len(self.camera_status)
        
        info_text = f"Multi-Camera System | Connected: {connected}/{total} | "
        info_text += f"Time: {time.strftime('%H:%M:%S')}"
        
        cv2.putText(canvas, info_text, (10, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 단축키 안내
        help_text = "Q: Quit | S: Screenshot | R: Reset Stats | 1-4: Toggle Camera"
        cv2.putText(canvas, help_text, (width - 400, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def toggle_camera(self, cam_id: int) -> bool:
        """카메라 활성화/비활성화 토글"""
        for camera in self.cameras:
            if camera['id'] == cam_id:
                if camera['enabled'] and cam_id in self.video_streams:
                    # 비활성화
                    self.video_streams[cam_id].stop()
                    del self.video_streams[cam_id]
                    del self.camera_status[cam_id]
                    camera['enabled'] = False
                    print(f"[카메라 {cam_id}] 비활성화됨")
                    return False
                elif not camera['enabled'] and camera['url']:
                    # 활성화
                    stream = VideoStream(camera['url'], buffer_size=10)
                    if stream.start():
                        self.video_streams[cam_id] = stream
                        self.camera_status[cam_id] = 'connected'
                        camera['enabled'] = True
                        print(f"[카메라 {cam_id}] 활성화됨")
                        return True
        return False
    
    def stop_all(self):
        """모든 카메라 중지"""
        print("\n모든 카메라 연결 종료 중...")
        
        # 스레드 풀 종료
        self.executor.shutdown(wait=True)
        
        # 모든 스트림 중지
        for cam_id, stream in self.video_streams.items():
            stream.stop()
            print(f"[카메라 {cam_id}] 연결 종료됨")
        
        self.video_streams.clear()
        self.camera_status.clear()
        self.last_frames.clear()