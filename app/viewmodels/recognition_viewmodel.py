# -*- coding: utf-8 -*-
"""
RecognitionViewModel — 번호판 인식 화면 상태(Observable)와 명령.
작업 스레드(RecognitionPipeline)와는 queue 로만 주고받고, tick() (GUI 스레드) 에서 결과를 반영한다.

입력은 두 가지
    camera : CameraViewModel 이 연 카메라의 최신 프레임 (보정 화면이 켜져 있으면 왜곡을 편 프레임)
    video  : 영상 파일 (videos\\ 목록 또는 [열기…]), 재생바
"""
import collections
import os
import queue
import time

from app import paths
from app.models import plate_models as PM
from app.models import tracker as TR
from app.models.frame_saver import CAPTURE_KINDS, capture_name
from app.models.pipeline import RecognitionPipeline, video_info
from app.mvvm import Event, Observable

SOURCES = {"camera": "카메라", "video": "영상 파일"}
NO_READER = "(글자 인식 없음 — 감지·추적만)"


def _first(registry):
    return next(iter(registry), None)


class RecognitionViewModel:
    CAPTURE_KINDS = CAPTURE_KINDS

    def __init__(self, cam_vm, det=None, reader=None, source="camera", video=None, auto_play=False):
        """
        cam_vm    : CameraViewModel (카메라 프레임·보정 상태를 읽음)
        det/reader: plate_models 에 등록된 키. None 이면 첫 번째 등록 모델
        auto_play : 처음 모델이 준비되면 바로 시작 (영상 파일로 시작할 때)
        """
        self.cam = cam_vm
        self.q = queue.Queue()
        self.pipe = RecognitionPipeline(self.q, self._camera_frame)
        # ---- 화면 상태 ----
        self.detector_choices = [(k, PM.describe(PM.DETECTORS, k)) for k in PM.DETECTORS]
        self.reader_choices = [(None, NO_READER)] + [(k, PM.describe(PM.READERS, k)) for k in PM.READERS]
        self.source = Observable("camera")
        self.videos = Observable([os.path.basename(v) for v in paths.list_videos()])
        self.video_path = Observable(video or paths.default_video())
        self.total = Observable(0)                 # 영상 프레임 수
        self.video_fps = 30.0
        default_det = PM.DEFAULT_DETECTOR if PM.DEFAULT_DETECTOR in PM.DETECTORS else _first(PM.DETECTORS)
        self.det_key = Observable(det or default_det)
        self.read_key = Observable(reader or _first(PM.READERS))
        self.ready = Observable(False)             # 감지기 준비됨
        self.image = Observable(None)              # 결과 이미지 (BGR, 원본 크기)
        self.position = Observable(0)              # 영상: 지금 보이는 프레임 번호
        self.playing = Observable(False)           # 영상: 재생 중 / 카메라: 인식 켜짐
        self.info = Observable({})                 # 시간·고정 번호판 수
        self.rows = Observable([])                 # 현재 화면 번호판
        self.status = Observable("")               # 위쪽 도구 줄에 보여 줄 요약
        self.capture_on = Observable(False)
        self.capture_kind = Observable(CAPTURE_KINDS[0])
        self.capture_status = Observable(("", "#707070"))
        self.capture_dir = Observable(None)        # 마지막으로 만든 캡처 폴더
        self.log = Event()                         # 메시지 (아래 메시지 상자)
        self.ended = Event()
        self.options = {"realtime": True, "show_entering": False,
                        "entry_margin": TR.ENTRY_MARGIN, "lock_frames": TR.LOCK_FRAMES,
                        # 화질·추적 유지 (pipeline 의 같은 이름 속성으로 넘어감)
                        "sharp_select": True, "enhance_contrast": False, "enhance_sharpen": False,
                        "hold_track": True, "low_conf": True, "reid": True}
        self.last_image_time = 0.0                 # 마지막 결과 이미지를 받은 시각 (View 가 카메라 화면과 고를 때 씀)
        self._loading = False
        self._auto_play = auto_play
        self._fps_hist = collections.deque(maxlen=30)
        self.screen_fps = 0.0
        self.pipe.start()
        if source == "video":
            self.set_source("video", self.video_path.value)
        self.select_models()
        self._update_capture_status()

    @property
    def has_detector(self):
        return bool(PM.DETECTORS)

    # ---------------- 카메라 프레임 (작업 스레드에서 호출) ----------------
    def _camera_frame(self):
        cam = self.cam
        stream = cam.stream
        if stream is None:
            return None, None
        fid, frame = stream.latest()
        if frame is None:
            return None, None
        if cam.undistort_on.value:                 # 보정 화면이 켜져 있으면 왜곡을 편 영상으로 인식
            frame = cam.analyzer.undistort(frame)
        return (id(stream), fid), frame

    # ---------------- 명령 ----------------
    def set_source(self, kind, video=None):
        """입력 바꾸기: camera / video (video 는 경로 또는 videos\\ 안 파일 이름)"""
        if kind == "video":
            path = video or self.video_path.value or paths.default_video()
            if path and not os.path.isabs(path):
                path = os.path.join(paths.VIDEOS_DIR, path)
            info = video_info(path) if path and os.path.exists(path) else None
            if info is None:
                self.log.emit(f"영상을 열 수 없습니다: {path or '(videos 폴더에 영상 없음 — [열기…] 로 고르세요)'}")
                self.source.set(self.source.value, force=True)   # 라디오 버튼을 지금 입력으로 되돌림
                return
            self.video_path.value = path
            self.total.value = info[2]
            self.video_fps = info[3]
            self.position.value = 0
        self.set_playing(False)
        self.source.value = kind
        self.pipe.cmd.put(("source", (kind, self.video_path.value if kind == "video" else None)))
        self._update_capture_status()

    def select_models(self, det="keep", reader="keep"):
        """고르는 순간 불러오기 (reader=None 이면 글자 인식 없이 감지·추적만)"""
        if det != "keep":
            self.det_key.value = det
        if reader != "keep":
            self.read_key.value = reader
        self.ready.value = False
        self._loading = True
        self.pipe.cmd.put(("models", (self.det_key.value, self.read_key.value)))
        self._update_capture_status()

    def toggle_play(self):
        self.set_playing(not self.playing.value)

    def set_playing(self, on):
        on = bool(on)
        if on and not self.has_detector:
            self.log.emit("등록된 번호판 감지기가 없어 인식을 시작할 수 없습니다. "
                          "app\\models\\plate_models.py 의 DETECTORS 에 감지기를 등록하세요.")
            on = False
        if on and self.source.value == "camera" and self.cam.stream is None:
            self.log.emit("카메라가 열려 있지 않습니다. [카메라 설정] 에서 카메라를 고르세요.")
            on = False
        if on and self._loading:
            self.log.emit("모델을 불러오는 중입니다. 준비되면 바로 시작합니다.")
        if self.source.value == "camera" and on != self.playing.value:
            self.log.emit("카메라 인식 켜짐" if on else "카메라 인식 꺼짐")
        self.pipe.playing = on
        self.playing.value = on

    def restart(self):
        self.pipe.cmd.put(("restart", None))

    def seek(self, frame_index):
        """재생바: 그 프레임으로 이동 (멈춘 상태면 그 위치 한 장을 처리해서 보여 줌)"""
        fi = int(max(0, min(self.total.value - 1, frame_index)))
        self.pipe.cmd.put(("seek", fi))
        self.position.value = fi

    def set_option(self, name, value):
        self.options[name] = value
        setattr(self.pipe, name, value)

    def set_capture(self, on):
        self.capture_on.value = bool(on)
        self.pipe.capture_kind = self.capture_kind.value
        self.pipe.capture_on = bool(on)
        if not on:
            self.pipe.capture_dir = None          # 끄면 지금 폴더는 닫음
        self._update_capture_status()

    def set_capture_kind(self, kind):
        self.capture_kind.value = kind
        self.pipe.capture_kind = kind

    def open_capture_dir(self):
        d = self.capture_dir.value or paths.REC_CAPTURE_DIR
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def close(self):
        self.pipe.playing = False
        self.pipe.cmd.put(("quit", None))

    # ---------------- 표시용 ----------------
    def time_text(self, fi=None):
        fi = self.position.value if fi is None else fi

        def mmss(f):
            s = f / self.video_fps
            return f"{int(s // 60):02d}:{s % 60:04.1f}"
        return f"{mmss(fi)} / {mmss(self.total.value)}"

    def _update_capture_status(self):
        name = capture_name(self.det_key.value, self.read_key.value)
        src = "cam" if self.source.value == "camera" else "video"
        d = self.pipe.capture_dir
        if d:
            n = self.pipe.saver.counts[d]
            extra = f", 못 따라가 버린 프레임 {self.pipe.saver.dropped}" if self.pipe.saver.dropped else ""
            wait = self.pipe.saver.pending()
            st = (f"캡처 저장 중: {n}장{f' (대기 {wait})' if wait else ''}{extra}\n{os.path.relpath(d, paths.ROOT)}",
                  "#c00000")
        elif self.capture_on.value:
            st = (f"캡처 준비: 시작하면 폴더를 만들고 저장\ncaptures\\recognition\\<시각>_{src}_{name}", "#a06000")
        else:
            last = f"\n마지막 폴더: {os.path.relpath(self.capture_dir.value, paths.ROOT)}" if self.capture_dir.value else ""
            st = (f"캡처 꺼짐 (폴더 이름: <시각>_{src}_{name}){last}", "#707070")
        self.capture_status.value = st

    def _update_status(self):
        if not self.has_detector:
            text = "인식: 번호판 모델 없음"
        elif self._loading:
            text = "인식: 모델 불러오는 중"
        elif not self.playing.value:
            text = "인식: 꺼짐" if self.source.value == "camera" else "영상: 멈춤"
        else:
            what = "인식 중" if self.source.value == "camera" else "영상 재생"
            text = f"{what} {self.screen_fps:4.1f} fps  [{self.det_key.value}+{self.read_key.value or '-'}]"
        self.status.value = text

    def tick(self):
        """GUI 스레드에서 주기적으로 호출: 작업 스레드 결과를 화면 상태로 옮김"""
        last = None
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "frame":
                    last = data
                elif kind == "preview":
                    self.last_image_time = time.time()
                    self.image.set(data, force=True)
                elif kind == "log":
                    self.log.emit(data)
                elif kind == "ready":
                    self._loading = False
                    self.ready.value = self.pipe.detect is not None
                    if self._auto_play and self.ready.value:   # 처음 한 번은 모델 준비되면 자동 시작
                        self._auto_play = False
                        self.set_playing(True)
                elif kind == "capdir":
                    self.capture_dir.value = data
                elif kind == "end":
                    self.playing.value = False
                    self.log.emit("영상 끝 — [처음부터] 를 누르거나 재생바로 이동하세요")
                    self.ended.emit()
        except queue.Empty:
            pass
        if last is not None and last["mode"] == self.source.value:
            self._fps_hist.append(time.perf_counter())
            h = self._fps_hist
            self.screen_fps = (len(h) - 1) / (h[-1] - h[0]) if len(h) > 1 and h[-1] > h[0] else 0.0
            self.last_image_time = time.time()
            # 이미지·목록은 numpy 배열이 들어 있어 값 비교를 하지 않고 항상 알림 (force)
            self.image.set(last["img"], force=True)
            if last["mode"] == "video":
                self.position.value = last["fi"]
            self.info.set({**{k: last[k] for k in ("fi", "det", "read", "mode")}, "stats": last["stats"],
                           "screen_fps": self.screen_fps}, force=True)
            self.rows.set(last["rows"], force=True)
        self._update_capture_status()
        self._update_status()

    def recognition_active(self, within=1.0):
        """카메라 인식 결과가 최근(within 초 안)에 들어오고 있는지 → View 가 결과 화면을 보여 줌"""
        return (self.source.value == "camera" and self.playing.value
                and time.time() - self.last_image_time < within)
