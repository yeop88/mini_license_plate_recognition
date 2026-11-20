"""
Model — 카메라 장치: 카메라 열기/검색, 백그라운드 프레임 읽기(CameraStream), 이미지 저장
(기존 camera_utils.py 의 장치 부분)
"""

import os
import re
import sys
import threading
import time
from datetime import datetime

import cv2

from app.paths import CAPTURE_DIR

# 없는 카메라 번호를 열 때 나오는 OpenCV 경고 숨기기
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


def open_camera(index, width, height, backend=None, fourcc="MJPG"):
    # Windows에서는 CAP_DSHOW 백엔드가 USB 카메라를 더 빠르고 안정적으로 연다
    if backend is None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    # USB 카메라는 기본(YUY2) 형식이면 720p에서 5~10 FPS로 떨어지는 경우가 많다. MJPG로 요청하면 30 FPS
    # 순서 주의: DSHOW 는 해상도를 먼저 정한 뒤 형식을 지정해야 MJPG 가 적용된다
    #   (형식을 먼저 지정하면 해상도를 바꿀 때 YUY2 로 돌아감 — ELP 5MP 카메라 실측)
    # FPS 는 지정하지 않는다: 지정하면 이 카메라는 YUY2 로 바뀌었다
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지원하면 버퍼를 줄여 지연 감소
    wake_frame_rate(cap)
    return cap


def wake_frame_rate(cap):
    """
    ELP 5MP(Global Shutter Camera) + DSHOW: 열거나 모드를 바꾼 뒤 게인 값을 한 번 '읽기만' 하면 60fps,
    읽지 않으면 약 40fps 로 나온다 (2026-11-20 실측, 1920x1080 MJPG: 읽지 않음 39.5 / 읽음 60.0, 연결마다 다시 필요).
    값은 바꾸지 않는다.
    """
    cap.get(cv2.CAP_PROP_GAIN)


def fourcc_text(value):
    v = int(value)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)) if v > 0 else "?"


class CameraStream:
    """
    별도 스레드에서 카메라 프레임을 계속 읽고 최신 프레임만 보관한다.
    카메라가 느려도 GUI가 멈추지 않고, 분석/표시가 프레임 읽기를 기다리지 않는다.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.fps = 0.0
        self.failed = False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        count, since = 0, time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.failed = True
                break
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            # 0.5초마다 실제로 받은 프레임 수로 카메라 FPS 계산
            count += 1
            now = time.time()
            if now - since >= 0.5:
                self.fps = count / (now - since)
                count, since = 0, now

    def latest(self):
        """(프레임 번호, 최신 프레임) 반환. 아직 프레임이 없으면 (0, None)"""
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


def get_camera_names():
    # DirectShow 장치 이름 (순서가 OpenCV CAP_DSHOW 번호와 같다). pygrabber가 없으면 빈 목록
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def find_cameras(max_index=5):
    """사용 가능한 카메라를 [(번호, 표시 이름), ...] 으로 반환"""
    names = get_camera_names()
    found = []
    for i in range(max(max_index, len(names))):
        cap = open_camera(i, 640, 480)
        if cap is None:
            continue
        ok, _ = cap.read()
        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = names[i] if i < len(names) else f"Camera {i}"
            found.append((i, f"[{i}] {name} ({w}x{h})"))
        cap.release()
    return found


def plain_name(label):
    """'[1] Global Shutter Camera (640x480)' 또는 '[1] Global Shutter Camera' → 'Global Shutter Camera'"""
    name = re.sub(r"^\[\d+\]\s*", "", label)
    return re.sub(r"\s*\(\d+x\d+\)$", "", name).strip()


def safe_name(name):
    """파일 이름에 쓸 수 있게 카메라 이름 정리 (예: Global Shutter Camera → Global_Shutter_Camera)"""
    return re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", name).strip("_") or "camera"


def write_image(path, image):
    # cv2.imwrite는 Windows 한글 경로에서 실패하므로 imencode 후 직접 저장
    ok, buf = cv2.imencode(os.path.splitext(path)[1], image)
    if ok:
        buf.tofile(path)
    return ok


def save_frame(frame, folder=CAPTURE_DIR):
    os.makedirs(folder, exist_ok=True)
    filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")[:-3] + ".png"
    path = os.path.join(folder, filename)
    return path if write_image(path, frame) else None
