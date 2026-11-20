# -*- coding: utf-8 -*-
"""
pipeline (Model) — 입력(카메라 또는 영상 파일) + 번호판 감지 + 추적 + 글자 인식 작업 스레드

    프레임 → 번호판 감지 → 추적(tracker, 화면 ID) → 감지가 끊긴 번호판은 영상으로 따라감
           → 읽는 중인 번호판만 잘라서 선명도 확인(흐리면 건너뜀) → (보정) → 글자 인식
           → lock_frames 번 읽으면 자리별 다수결로 글자 고정 → 결과 이미지
    GUI 와는 queue 로만 주고받음 (ViewModel 이 tick() 에서 꺼내 씀)

입력
    camera : camera_frame() 이 주는 최신 카메라 프레임 (ViewModel 이 보정 화면이 켜져 있으면 왜곡을 편 프레임을 줌).
             새 프레임이 올 때마다 하나씩 처리하고, 처리가 느리면 중간 프레임은 건너뜀 (항상 최신 프레임)
    video  : 영상 파일 (재생·처음부터·재생바 이동)

모델 (plate_models.py 에 등록된 것)
    감지기가 없으면 인식을 시작할 수 없음. 글자 인식기가 없으면 감지·추적만 함

명령 (cmd queue): ("models", (감지기 키, 글자 인식 키)) / ("source", ("camera" | "video", 영상 경로))
                  / ("restart", None) / ("seek", 프레임 번호) / ("quit", None)
보내는 메시지 (out queue): ("log", 글) / ("ready", (감지기 키, 글자 인식 키)) / ("frame", dict) / ("preview", 이미지)
                           / ("capdir", 폴더) / ("end", None)
"""
import inspect
import os
import queue
import threading
import time

import cv2

from app import paths
from app.models import plate_enhance as PE
from app.models import plate_models as PM
from app.models import tracker as TR
from app.models.frame_saver import CAPTURE_KINDS, FrameSaver, capture_name
from app.models.render import draw_tracks


def pick_device():
    """GPU 가 있으면 cuda:0 (torch 가 없으면 cpu)"""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def video_info(path):
    """영상 정보 (너비, 높이, 프레임 수, fps). 못 열면 None"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    return info


class RecognitionPipeline(threading.Thread):
    def __init__(self, out_q, camera_frame):
        """camera_frame: 인자 없는 함수 → (프레임 구분값, BGR 프레임) 또는 (None, None). 작업 스레드에서 호출됨"""
        super().__init__(daemon=True)
        self.q = out_q
        self.camera_frame = camera_frame
        self.cmd = queue.Queue()
        self.device = None                 # 처음 모델을 불러올 때 정함
        # 입력
        self.mode = "camera"
        self.video = None
        self.W, self.H, self.n, self.vfps = 1920, 1080, 0, 30
        # 설정 (ViewModel 이 바꿈)
        self.playing, self.realtime, self.show_entering = False, True, False
        self.entry_margin, self.lock_frames = TR.ENTRY_MARGIN, TR.LOCK_FRAMES
        self.sharp_select = True           # 흐린 프레임 건너뛰기 + 선명도 가중 다수결
        self.enhance_contrast = True       # 글자 인식 전 대비 보정 (CLAHE) — 종합 시험에서 저대비·초점 흐림에 효과
        self.enhance_sharpen = True        # 글자 인식 전 선명화 (언샤프 마스크)
        self.hold_track = True             # 고정 후 오래 기다림 + 감지가 끊기면 영상으로 따라가기
        self.low_conf = True               # 추적 중인 번호판은 낮은 신뢰도 감지로도 이어 감
        self.reid = True                   # 끊긴 뒤 같은 번호가 읽히면 이전 화면 ID·고정값 되살림
        self._det_has_conf = False         # 감지기가 detect(frame, conf=...) 를 받는지
        self.capture_on = False            # [결과 캡처] 체크 상태
        self.capture_dir = None            # 지금 저장 중인 폴더 (시작할 때 만들어짐)
        self.capture_kind = CAPTURE_KINDS[0]
        # 상태
        self.det_key = self.read_key = None
        self.detect = self.read = None
        self.tracker = None
        self.saver = FrameSaver()
        self.saver.start()
        self.reset_stats()

    # ---------------- 상태 ----------------
    def new_tracker(self, keep_numbering=False):
        """keep_numbering: 재생바 이동·해상도 변경처럼 추적만 새로 할 때 화면 ID 를 이어서 붙임"""
        nxt = self.tracker.next_no if (keep_numbering and self.tracker is not None) else 1
        self.tracker = TR.Tracker(self.W, self.H, self.entry_margin, self.lock_frames, next_no=nxt)

    def reset_stats(self):
        self.stats = {"frames": 0, "detections": 0, "locked": 0, "plates": []}
        self.new_tracker()

    def log(self, msg):
        self.q.put(("log", msg))

    def ensure_capture_dir(self):
        """캡처가 켜져 있고 폴더가 아직 없으면 지금 모델 이름으로 만든다 (시작 시점)"""
        if self.capture_on and self.capture_dir is None and self.detect is not None:
            src = "cam" if self.mode == "camera" else "video"
            d = os.path.join(paths.REC_CAPTURE_DIR,
                             f"{time.strftime('%Y%m%d_%H%M%S')}_{src}_{capture_name(self.det_key, self.read_key)}")
            os.makedirs(d, exist_ok=True)
            self.capture_dir = d
            self.q.put(("capdir", d))

    def load(self, dk, rk):
        if (dk, rk) != (self.det_key, self.read_key):
            self.capture_dir = None        # 모델이 바뀌면 지금 캡처 폴더는 닫음
        if self.device is None and (dk or rk):
            self.device = pick_device()
        if dk != self.det_key:
            self.detect = None
            if dk:
                self.log(f"번호판 감지기 {PM.describe(PM.DETECTORS, dk)} 불러오는 중...")
                self.detect = PM.DETECTORS[dk]["build"](self.device)
            self.det_key = dk
            try:
                self._det_has_conf = self.detect is not None and "conf" in inspect.signature(self.detect).parameters
            except (TypeError, ValueError):
                self._det_has_conf = False
        if rk != self.read_key:
            self.read = None
            if rk:
                self.log(f"글자 인식 {PM.describe(PM.READERS, rk)} 불러오는 중...")
                self.read = PM.READERS[rk]["build"](self.device)
            self.read_key = rk
        self.reset_stats()
        if self.detect is None:
            self.log("번호판 감지기가 없습니다. app\\models\\plate_models.py 에 감지기를 등록하세요.")
        else:
            reader = PM.describe(PM.READERS, rk) if self.read else "없음 (감지·추적만)"
            backend = getattr(self.detect, "backend", "")
            backend = f", {backend}" if backend else ""
            self.log(f"인식 준비됨: 감지 {PM.describe(PM.DETECTORS, dk)}{backend} / 글자 인식 {reader} ({self.device})")
        self.q.put(("ready", (dk, rk)))

    def set_size(self, w, h):
        """입력 해상도가 바뀌면 (카메라 해상도 변경·카메라 전환) 추적을 새로 시작"""
        if (w, h) != (self.W, self.H):
            self.W, self.H = w, h
            self.new_tracker(keep_numbering=True)

    # ---------------- 프레임 처리 ----------------
    def _detect(self, frame):
        """감지. '낮은 감지로 추적 잇기' 가 켜져 있고 감지기가 conf 인자를 받으면 낮은 기준으로 감지"""
        if self.low_conf and self._det_has_conf:
            return self.detect(frame, conf=PM.LOW_CONF)
        return self.detect(frame)

    def process(self, frame, fi, save):
        t = time.perf_counter()
        xyxy, conf = self._detect(frame)
        det = (time.perf_counter() - t) * 1000
        s = self.stats
        s["frames"] += 1
        s["detections"] += sum(1 for c in conf if c >= PM.YOLO_CONF)
        tk_ = self.tracker
        tk_.entry_margin, tk_.lock_frames = self.entry_margin, self.lock_frames
        tk_.hold, tk_.reid = self.hold_track, self.reid
        if self.low_conf and self._det_has_conf:
            pairs = tk_.update(xyxy, fi, conf=conf, high=PM.YOLO_CONF)
        else:
            pairs = tk_.update(xyxy, fi)
        # 감지가 끊긴 번호판은 직전 번호판 영상으로 따라감 (고정 후 추적 유지)
        t = time.perf_counter()
        bridged = []
        if self.hold_track:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bridged = tk_.bridge(gray)
            tk_.remember(gray, pairs)
        hold_ms = (time.perf_counter() - t) * 1000
        # 읽는 중인 번호판만 글자 인식 (고정된 번호판은 다시 읽지 않음)
        meas = [(tr, di) for tr, di in pairs if tr.state == "measuring"]
        t = time.perf_counter()
        if meas and self.read is not None:
            crops = [PM.crop_plate(frame, xyxy[di]) for _, di in meas]
            sharps = [PE.sharpness(c) for c in crops] if self.sharp_select else [None] * len(crops)
            todo = []
            for (tr, di), crop, sh in zip(meas, crops, sharps):
                if sh is not None:
                    tr.best_sharp = max(tr.best_sharp, sh)
                    if sh < tr.best_sharp * PE.SKIP_RATIO:      # 이 번호판이 가장 선명했을 때보다 많이 흐리면 읽지 않음
                        tr.last = "(흐림)"
                        continue
                if self.enhance_contrast or self.enhance_sharpen:
                    crop = PE.enhance(crop, self.enhance_contrast, self.enhance_sharpen)
                todo.append((tr, crop, sh))
            if todo:
                for (tr, _, sh), (text, c) in zip(todo, self.read([c for _, c, _ in todo])):
                    tr.last = text
                    if tk_.add_prediction(tr, text, c, fi, sharp=sh) == "lock":
                        self._on_lock(tr)
        read = (time.perf_counter() - t) * 1000
        # 다 읽기 전에 화면 가장자리에 닿으면(나가는 중) 모은 만큼으로 고정
        for tr, di in pairs:
            if tr.state == "measuring" and tk_.touches_edge(tr.box) and tr.preds:
                tr.lock(fi, "가장자리")
                self._on_lock(tr)
        rows = []
        shown = [(tr, tr.box, float(conf[di])) for tr, di in pairs] + [(tr, tr.box, 0.0) for tr in bridged]
        for tr, box, dc in sorted(shown, key=lambda p: (p[1][1], p[1][0])):
            rows.append({"id": tr.id, "no": tr.no, "state": tr.state, "box": box, "det_conf": dc,
                         "n": len(tr.preds), "need": tk_.lock_frames, "last": getattr(tr, "last", ""),
                         "text": tr.text, "ratio": tr.ratio, "conf": tr.conf, "lock_frame": tr.lock_frame,
                         "reason": tr.lock_reason, "bridged": tr in bridged})
        total = det + read + hold_ms
        cam = self.mode == "camera"
        hud = f"{self.det_key}+{self.read_key or '-'}  {total:.0f} ms ({1000 / max(total, 1e-3):.0f} fps)  " + \
              (f"recog #{fi}" if cam else f"frame {fi}")
        vis = draw_tracks(frame, rows, self.show_entering, hud, hud_bottom_right=cam)
        if save and self.capture_dir:
            self.saver.put((self.capture_dir, fi, self.capture_kind, vis, frame, rows))
        self.q.put(("frame", {"img": vis, "fi": fi, "det": det, "read": read, "rows": rows,
                              "stats": {k: (list(v) if isinstance(v, list) else v) for k, v in s.items()},
                              "mode": self.mode}))

    def _on_lock(self, tr):
        s = self.stats
        s["locked"] += 1
        s["plates"] = (s["plates"] + [(tr.no, tr.text)])[-50:]     # 최근 고정된 번호판 (화면 목록용)

    def _open_source(self, kind, path):
        """입력 바꾸기. 영상이면 첫 프레임을 미리 보여 줌 (재생 여부는 ViewModel 이 정함 — 여기서 바꾸면 바로 누른 재생이 꺼짐)"""
        self.capture_dir = None
        self.mode = kind
        if kind == "video":
            info = video_info(path)
            if info is None:
                self.log(f"[오류] 영상을 열 수 없습니다: {path}")
                self.mode = "camera"
                return None
            self.video = path
            self.W, self.H, self.n, self.vfps = info
            cap = cv2.VideoCapture(path)
            ok, first = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if ok:
                self.q.put(("preview", first))
            self.reset_stats()
            self.log(f"입력: 영상 {os.path.basename(path)} ({self.W}x{self.H}, {self.n} 프레임)")
            return cap
        self.reset_stats()
        self.log("입력: 카메라 (보정 화면이 켜져 있으면 왜곡을 편 영상으로 인식)")
        return None

    def run(self):
        cap = None
        fi = 0
        last_cam = None
        while True:
            seek_to = None
            try:
                while True:
                    c, arg = self.cmd.get_nowait()
                    if c == "models":
                        try:
                            self.load(*arg)
                        except Exception as e:
                            self.log(f"[오류] 모델 불러오기 실패: {type(e).__name__}: {e}")
                            self.q.put(("ready", (self.det_key, self.read_key)))
                    elif c == "source":
                        if cap is not None:
                            cap.release()
                        cap = self._open_source(*arg)
                        fi, last_cam, seek_to = 0, None, None
                    elif c == "restart":
                        seek_to = None
                        fi = 0
                        self.reset_stats()
                        if self.mode == "video" and cap is not None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            if not self.playing:
                                seek_to = 0        # 멈춘 상태면 첫 프레임을 한 장 보여 줌
                    elif c == "seek":
                        if self.mode == "video":
                            seek_to = int(arg)     # 드래그 중 여러 번 오면 마지막 것만
                    elif c == "quit":
                        return
            except queue.Empty:
                pass
            ready = self.detect is not None

            # ---------------- 카메라 ----------------
            if self.mode == "camera":
                if not self.playing or not ready:
                    time.sleep(0.02)
                    continue
                key, frame = self.camera_frame()
                if frame is None or key == last_cam:
                    time.sleep(0.003)
                    continue
                last_cam = key
                self.set_size(frame.shape[1], frame.shape[0])
                self.ensure_capture_dir()
                self.process(frame, fi, save=True)
                fi += 1
                while self.q.qsize() > 3 and self.playing:   # 화면이 못 따라가면 기다림
                    time.sleep(0.005)
                continue

            # ---------------- 영상 파일 ----------------
            if cap is None:
                time.sleep(0.02)
                continue
            if seek_to is not None:
                fi = max(0, min(self.n - 1, seek_to))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                self.new_tracker(keep_numbering=True)   # 시간대가 바뀌면 이전 추적은 의미 없음 (ID 번호는 이어감)
                if not (self.playing and ready):           # 멈춘 상태: 그 위치 한 장만 보여 줌 (캡처 안 함)
                    ok, frame = cap.read()
                    if ok:
                        if ready:
                            self.process(frame, fi, save=False)
                        else:
                            self.q.put(("preview", frame))     # 감지기가 없으면 원본만
                        fi += 1
                continue
            if not self.playing or not ready:
                time.sleep(0.02)
                continue
            t0 = time.perf_counter()
            self.ensure_capture_dir()
            ok, frame = cap.read()
            if not ok:
                self.playing = False
                self.q.put(("end", None))
                continue
            self.process(frame, fi, save=True)
            fi += 1
            if self.realtime:
                wait = 1 / self.vfps - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
            while self.q.qsize() > 3:      # 화면이 못 따라가면 기다림 (메모리 쌓이지 않게)
                time.sleep(0.005)
