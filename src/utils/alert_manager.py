"""
Alert management system for danger detection
"""
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import threading

class AlertManager:
    def __init__(self, log_file: str = 'danger_detection.log'):
        self.log_file = log_file
        self.last_alert_time = defaultdict(float)
        self.alert_count = defaultdict(int)
        self.active_alerts = set()
        self.lock = threading.Lock()
        
        # 로깅 설정 - UTF-8 인코딩 명시
        self._setup_logging()
        
    def _setup_logging(self):
        """로깅 설정"""
        # 기존 핸들러 제거
        logger = logging.getLogger(__name__)
        logger.handlers = []
        
        # UTF-8 인코딩으로 파일 핸들러 생성
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 로거 설정
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    def check_alert_cooldown(self, alert_type: str, cooldown_seconds: int) -> bool:
        """알림 쿨다운 체크"""
        with self.lock:
            current_time = time.time()
            last_time = self.last_alert_time.get(alert_type, 0)
            
            if current_time - last_time >= cooldown_seconds:
                self.last_alert_time[alert_type] = current_time
                return True
            return False
    
    def send_alert(self, alert_type: str, message: str, 
                   location: Optional[Dict] = None, 
                   confidence: Optional[float] = None,
                   cooldown: int = 30):
        """위험 알림 전송"""
        if not self.check_alert_cooldown(alert_type, cooldown):
            return
        
        with self.lock:
            self.alert_count[alert_type] += 1
            self.active_alerts.add(alert_type)
        
        # 알림 메시지 구성
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"\n{'='*60}\n"
        alert_msg += f"⚠️  위험 감지: {alert_type.upper()}\n"
        alert_msg += f"시간: {timestamp}\n"
        alert_msg += f"내용: {message}\n"
        
        if location:
            alert_msg += f"위치: X={location.get('x', 'N/A')}, Y={location.get('y', 'N/A')}\n"
        
        if confidence:
            alert_msg += f"신뢰도: {confidence:.2%}\n"
        
        alert_msg += f"누적 감지 횟수: {self.alert_count[alert_type]}\n"
        alert_msg += f"{'='*60}\n"
        
        # 콘솔 출력 (색상 추가)
        self._print_colored_alert(alert_type, alert_msg)
        
        # 로그 기록
        self.logger.warning(f"{alert_type}: {message}")
    
    def _print_colored_alert(self, alert_type: str, message: str):
        """색상이 있는 알림 출력"""
        color_codes = {
            'fire': '\033[91m',  # 빨간색
            'restricted_area': '\033[93m',  # 노란색
            'fall': '\033[94m',  # 파란색
            'no_helmet': '\033[95m',  # 보라색
        }
        
        color = color_codes.get(alert_type, '\033[0m')
        reset = '\033[0m'
        
        print(f"{color}{message}{reset}")
    
    def get_statistics(self) -> Dict:
        """알림 통계 반환"""
        with self.lock:
            return {
                'total_alerts': sum(self.alert_count.values()),
                'alert_breakdown': dict(self.alert_count),
                'active_alerts': list(self.active_alerts),
                'last_alert_times': dict(self.last_alert_time)
            }
    
    def clear_alert(self, alert_type: str):
        """활성 알림 해제"""
        with self.lock:
            self.active_alerts.discard(alert_type)
    
    def reset_statistics(self):
        """통계 초기화"""
        with self.lock:
            self.alert_count.clear()
            self.last_alert_time.clear()
            self.active_alerts.clear()