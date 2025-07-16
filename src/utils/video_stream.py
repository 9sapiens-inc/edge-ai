"""
Video stream handler for RTSP camera
"""
import cv2
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np

class VideoStream:
    def __init__(self, rtsp_url: str, buffer_size: int = 10):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture = None
        self.thread = None
        self.running = False
        self.last_frame = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
    def start(self) -> bool:
        """스트림 시작"""
        try:
            self.capture = cv2.VideoCapture(self.rtsp_url)
            
            # RTSP 최적화 설정
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.capture.isOpened():
                print(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False
                
            self.running = True
            self.thread = threading.Thread(target=self._update)
            self.thread.daemon = True
            self.thread.start()
            
            print(f"Successfully connected to RTSP stream: {self.rtsp_url}")
            return True
            
        except Exception as e:
            print(f"Error starting video stream: {e}")
            return False
    
    def _update(self):
        """백그라운드에서 프레임 업데이트"""
        while self.running:
            try:
                ret, frame = self.capture.read()
                
                if not ret:
                    print("Failed to read frame, attempting to reconnect...")
                    self._reconnect()
                    continue
                
                # 프레임 카운트 및 FPS 계산
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time > 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # 큐가 가득 차면 오래된 프레임 제거
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                self.frame_queue.put(frame)
                self.last_frame = frame
                
            except Exception as e:
                print(f"Error in video stream update: {e}")
                time.sleep(0.1)
    
    def _reconnect(self):
        """연결 재시도"""
        print("Attempting to reconnect to RTSP stream...")
        self.capture.release()
        time.sleep(5)
        
        try:
            self.capture = cv2.VideoCapture(self.rtsp_url)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.capture.isOpened():
                print("Successfully reconnected to RTSP stream")
            else:
                print("Failed to reconnect to RTSP stream")
                
        except Exception as e:
            print(f"Error during reconnection: {e}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 읽기"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get(timeout=0.1)
                return True, frame
            elif self.last_frame is not None:
                return True, self.last_frame
            else:
                return False, None
        except:
            return False, None
    
    def get_fps(self) -> float:
        """현재 FPS 반환"""
        return self.fps
    
    def stop(self):
        """스트림 중지"""
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            
        if self.capture is not None:
            self.capture.release()
            
        print("Video stream stopped")
    
    def resize_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """프레임 리사이즈"""
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def is_running(self) -> bool:
        """스트림 실행 상태 확인"""
        return self.running and self.capture is not None and self.capture.isOpened()