"""
ViewModel — 카메라 화면의 상태와 명령

상태 (Observable, View 가 구독):
    cameras, selected_camera, resolution_options, resolution, board, capture_count, frame_id,
    analyze_on, auto_board_on, auto_correct_on, undistort_on, guide_on, guide_mode
메시지 (Event): log(msg)
명령 (View 가 버튼/키에서 호출):
    start, refresh, select_camera, switch_to, next_camera, set_resolution, reprobe, capture,
    toggle_analyze, toggle_auto_board, toggle_auto_correct, toggle_undistort, reset_analysis,
    save_calibration, save_checkerboard, toggle_guide, next_guide_mode, shutdown
tick(): View 의 타이머가 GUI 스레드에서 자주 호출. 백그라운드 결과(분석·스펙 측정·프레임)를 상태에 반영한다
render_state(): 영상 위에 그릴 정보 묶음 (View 의 overlays 가 사용)
"""

import os
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from app.models.camera import (CAPTURE_DIR, CameraStream, find_cameras, fourcc_text, open_camera, plain_name,
                               safe_name, save_frame, write_image)
from app.models.camera_profile import (PRESETS, choose_mode, parse_size, probe_camera, record_mode,
                                       resolution_options)
from app.models.focus import PEAKING_LEVELS, FocusMeter
from app.models.lens_distortion import (DistortionAnalyzer, calib_is_stable, calibrate_from_images, classify_lens,
                                        detect_board_size, make_checkerboard)

CAPTURE_CALIB_MAX = 10       # 캡처로 보정할 때 쓰는 최근 캡처 수 (현재 해상도와 같은 것만)

FOCUS_INTERVAL = 1 / 30      # 초점 선명도 계산 간격 (초). 60fps 카메라여도 초당 30번이면 충분
MIN_RECORD_RATIO = 0.3       # 처음 쓰는 해상도의 실측 FPS 가 카메라가 알려 준 최대 FPS 의 30% 미만이면 기록 안 함
MIN_RECORD_FPS = 5.0         # 카메라가 최대 FPS 를 알려 주지 않을 때의 기준
from app.mvvm import Event, Observable
from app.paths import PROJECT_DIR

DEFAULT_BOARD = (9, 6)       # 체커보드 이미지 저장 시 기본 크기
AUTO_SEARCH_DELAY = 2.0      # 현재 크기로 보드를 이 시간(초) 이상 못 찾으면 크기 자동 검색
AUTO_SEARCH_INTERVAL = 0.5   # 자동 검색 간격 (초)
GUIDE_MODES = ["ALL", "GRID", "FINE", "THIRDS", "CENTER", "DIAG", "BOARD"]   # 가이드 종류 (V 로 차례대로)
GUIDE_MODE_NAMES = {"ALL": "전체 (격자+체커보드)", "GRID": "정사각 격자", "FINE": "촘촘한 정사각 격자",
                    "THIRDS": "3분할", "CENTER": "중심·동심원", "DIAG": "대각선", "BOARD": "체커보드"}
GUIDE_MODE_HELP = {
    "ALL": "정사각 격자 + 체커보드 직선·코너 오차",
    "GRID": "짧은 변 6칸 정사각 격자, 선 하나가 영상 중심을 지남",
    "FINE": "짧은 변 12칸 정사각 격자 (작은 휨까지 보기)",
    "THIRDS": "가로·세로 3등분선 (구도)",
    "CENTER": "중심 십자 + 동심원 (렌즈 중심·광축 정렬, 원이 찌그러지면 기울어짐·왜곡)",
    "DIAG": "모서리를 잇는 대각선 + 중심 십자 (카메라 수평·정렬)",
    "BOARD": "체커보드 각 줄의 직선과 코너 오차 (초록 < 1px, 주황 < 2.5px, 빨강)",
}
RESOLUTION_NAMES = {"request": "기본", "max_fps": "최고 프레임", "max_quality": "최고 화질"}


class CameraViewModel:
    def __init__(self, profiles, start_camera=None, width=1920, height=1080, board=None, resolution=None,
                 save_dir=PROJECT_DIR, capture_dir=CAPTURE_DIR):
        """
        profiles   : ProfileRepository (카메라 스펙 저장소)
        resolution : 해상도 선택 키 ("request" / "max_fps" / "max_quality" / "2048x1536"). None 이면 카메라별 지난번 값
        save_dir   : 보정값·체커보드 이미지를 저장할 폴더
        """
        self.profiles = profiles
        self.request_size = (width, height)
        self.start_camera = start_camera
        self.save_dir = save_dir
        self.capture_dir = capture_dir
        self.capture_calib_job = None       # 캡처로 보정 (백그라운드) {"result", "done", "reason"}

        # ---------- 상태 (View 가 구독) ----------
        self.cameras = Observable([])                 # [(번호, 표시 이름)]
        self.selected_camera = Observable(None)       # 목록에서 선택된 카메라 번호
        self.resolution_options = Observable([])      # [(키, 표시 이름)]
        self.resolution = Observable(resolution or "request")
        self.board = Observable(board)                # 체커보드 내부 코너 (가로, 세로) 또는 None
        self.capture_count = Observable(0)
        self.frame_id = Observable(0)                 # 새 프레임이 오면 바뀜 → View 가 다시 그림
        # 시작할 때는 모든 기능을 끈다 (영상만 띄움). 필요한 기능은 버튼·키로 켠다.
        # 왜곡 분석·보드 자동 인식은 매 프레임/반복 계산이라 고해상도에서 CPU 부담이 크다
        self.analyze_on = Observable(False)
        self.auto_board_on = Observable(False)
        self.auto_correct_on = Observable(False)
        self.undistort_on = Observable(False)
        self.guide_on = Observable(False)
        self.guide_mode = Observable("ALL")
        self.focus_on = Observable(False)             # 초점 맞추기 도우미 (선명도 막대·판정)
        self.peaking_on = Observable(False)           # 포커스 피킹 (초점 맞은 윤곽을 색으로)
        self.peaking_level = Observable("보통")        # 피킹 감도: 높음 / 보통 / 낮음
        self.focus_text = Observable("")              # 위쪽 줄에 보여 줄 선명도 요약
        self.log = Event()
        self.focus = FocusMeter()
        self.focus_last = 0.0
        self.focus_last_verdict = None

        # ---------- 내부 상태 ----------
        self.resolution_explicit = resolution is not None   # 사용자가 고른 해상도면 카메라를 바꿔도 유지
        self.running = True
        self.stream = None
        self.frame = None
        self.frame_no = 0
        self.shown_id = 0
        self.black_warned = False
        self.reopen_tried = False
        self.slow_warned = False
        self.started = time.time()
        self.cam_key = None              # 지금 열린 카메라 이름 (스펙·보정값 저장 키)
        self.cam_name = "-"
        self.cam_fps = 0.0
        self.cam_fourcc = "?"
        self.backend = "-"
        self.open_spec = None            # 지금 연 설정 (choose_mode 결과)
        self.mode_recorded = True        # 미측정 해상도면 False → 3초 뒤 실측 FPS 를 스펙에 기록
        self.probe_job = None            # 진행 중인 스펙 측정

        self.analyzer = DistortionAnalyzer(board)
        self.analyzer_lock = threading.Lock()   # 분석 스레드와 GUI 스레드가 analyzer 를 함께 씀
        self.analysis_out = None
        self.last_res = None
        self.analysis_fps = 0.0
        self.board_visible = False
        self.lens = None
        self.auto_busy = False
        self.auto_result = None
        self.auto_last_try = 0.0
        self.auto_candidate = None
        self.last_seen = 0.0
        threading.Thread(target=self._analysis_loop, daemon=True).start()

    # =================================================================
    # 시작 / 카메라 목록
    # =================================================================
    def start(self):
        self.log.emit("프로그램 시작. 카메라를 검색합니다... (기능은 모두 꺼진 상태로 시작: "
                      "D 왜곡 분석, A 보드 자동 인식, C 자동 보정, G 가이드선, F 초점 도우미, P 피킹)")

    def refresh(self):
        self.release()
        self.log.emit("카메라 검색 중...")
        cams = find_cameras()
        self.cameras.value = cams
        if not cams:
            self.selected_camera.value = None
            self.log.emit("사용 가능한 카메라가 없습니다. 연결 상태를 확인하고 [새로고침]을 누르세요.")
            return
        self.log.emit(f"카메라 {len(cams)}개 발견: " + ", ".join(l for _, l in cams))
        indexes = [i for i, _ in cams]
        if self.start_camera in indexes:
            index = self.start_camera
        else:
            # --camera 를 주지 않았으면 지난번에 쓴 카메라(이름 기준), 없으면 첫 번째
            names = [plain_name(l) for _, l in cams]
            last = self.profiles.last_camera
            index = indexes[names.index(last)] if last in names else indexes[0]
            if last in names:
                self.log.emit(f"지난번에 쓴 카메라 '{last}' 를 선택합니다.")
        self.select_camera(index)

    def switch_to(self, index):
        indexes = [i for i, _ in self.cameras.value]
        if index not in indexes:
            self.log.emit(f"카메라 {index}번은 목록에 없습니다. 사용 가능: {indexes}")
            return
        if self.stream is not None and index == self.start_camera:
            return
        self.select_camera(index)

    def next_camera(self):
        cams = self.cameras.value
        if not cams:
            self.log.emit("전환할 카메라가 없습니다.")
            return
        if len(cams) == 1:
            self.log.emit("연결된 카메라가 1개뿐이라 전환할 수 없습니다.")
            return
        indexes = [i for i, _ in cams]
        pos = indexes.index(self.start_camera) if self.start_camera in indexes else -1
        self.switch_to(indexes[(pos + 1) % len(indexes)])

    # =================================================================
    # 카메라 열기 (스펙 확인 → 측정 또는 저장된 스펙으로)
    # =================================================================
    def select_camera(self, index, retry=False, force_probe=False):
        """카메라 선택. 처음 보는 카메라(이름 기준)는 스펙을 먼저 측정·저장하고, 저장된 스펙이 있으면 그것으로 연다"""
        self.reopen_tried = retry
        if self.probe_job is not None:          # 진행 중인 측정은 그만둔다
            self.probe_job["stop"] = True
            self.probe_job = None
        self.release()
        self.selected_camera.value = index
        name = plain_name(dict(self.cameras.value).get(index, f"Camera {index}"))
        if (force_probe or not self.profiles.has(name) or not self.profiles.get(name).get("best")) and not retry:
            self._start_probe(index, name, "스펙 다시 측정" if force_probe else "처음 보는 카메라")
            return
        self._open_selected(index, name, retry)

    def reprobe(self):
        if self.start_camera not in dict(self.cameras.value):
            self.log.emit("스펙을 측정할 카메라가 없습니다.")
            return
        self.select_camera(self.start_camera, force_probe=True)

    def set_resolution(self, key):
        """[해상도] 목록에서 고른 값으로 카메라를 다시 연다"""
        if key == self.resolution.value and self.stream is not None:
            return
        self.resolution.value = key
        self.resolution_explicit = True
        label = dict(self.resolution_options.value).get(key, key)
        self.log.emit(f"해상도: {label} → 카메라를 다시 엽니다.")
        if self.start_camera in dict(self.cameras.value) and self.probe_job is None:
            self.select_camera(self.start_camera)

    def _start_probe(self, index, name, why):
        """스펙 측정은 백그라운드 스레드에서. 끝나면 tick() 이 결과를 받아 저장하고 카메라를 연다"""
        self.start_camera = index
        w, h = self.request_size
        job = {"index": index, "name": name, "result": None, "done": False, "stop": False, "msgs": []}
        self.probe_job = job
        self.log.emit(f"{why}: '{name}' 스펙을 측정합니다 — 최고 화질 / 최고 프레임 / "
                      f"기본 해상도({w}x{h})의 가장 빠른 형식 (30초~1분). 끝나면 저장하고 자동으로 엽니다.")

        def work():
            try:
                job["result"] = probe_camera(index, w, h, stop=lambda: job["stop"] or not self.running,
                                             log=job["msgs"].append)
            except Exception as e:      # 측정 실패해도 프로그램은 계속
                job["result"] = {"error": str(e)}
            job["done"] = True
        threading.Thread(target=work, daemon=True).start()

    def _finish_probe(self, job):
        prof = job["result"]
        if prof is None:
            return                              # 중간에 그만둔 측정
        if "error" in prof:
            self.log.emit(f"스펙 측정 실패: {prof['error']} → 기본 설정(MJPG)으로 엽니다.")
            self._open_selected(job["index"], job["name"], False, profile_ok=False)
            return
        old = self.profiles.get(job["name"]) or {}
        for k in ("calibration_file", "model", "last_resolution"):   # 사용자가 적어 둔 값은 유지
            if old.get(k):
                prof[k] = old[k]
        self.profiles.put(job["name"], prof)
        self.profiles.save()
        b = prof.get("best") or {}
        fmt = lambda m: f"{m['fourcc']} {m['width']}x{m['height']} {m['fps']:.0f}fps" if m else "없음"
        color = {True: "컬러", False: "흑백"}.get(prof["color"], "색상 미확인")
        modes = prof.get("modes", [])
        bad = [m for m in modes if not m.get("ok")]
        self.log.emit(f"'{job['name']}' 호환성 시험 끝 — 쓸 수 있는 모드 {len(modes) - len(bad)}/{len(modes)}개. "
                      f"스펙 저장 ({os.path.basename(self.profiles.path)}, {color}): "
                      f"최고 화질 {fmt(b.get('max_quality'))} / 최고 프레임 {fmt(b.get('max_fps'))} / "
                      f"기본 해상도 {fmt(b.get('request'))}")
        if bad:
            self.log.emit("사용 불가 모드: " + "; ".join(f"{m['request']} ({m['problem']})" for m in bad))
        self._open_selected(job["index"], job["name"], False)

    def _open_selected(self, index, name, retry, profile_ok=True):
        prof = self.profiles.get(name) if profile_ok else None
        if not self.resolution_explicit and prof and prof.get("last_resolution"):
            self.resolution.value = prof["last_resolution"]        # 카메라별 지난번 해상도
        key = self.resolution.value
        if key not in PRESETS and parse_size(key) is None:
            key = self.resolution.value = "request"
        spec = choose_mode(prof, key, self.request_size)
        fourcc, ow, oh = spec["request_fourcc"], spec["width"], spec["height"]
        cap = None
        if retry and sys.platform == "win32":
            # 재연결은 MSMF 백엔드로 시도 (DSHOW에서 속도가 들쭉날쭉한 가상 카메라 대응). 실패하면 DSHOW
            cap = open_camera(index, ow, oh, cv2.CAP_MSMF, fourcc=fourcc)
            if cap is not None and not cap.read()[0]:
                cap.release()
                cap = None
        if cap is None:
            cap = open_camera(index, ow, oh, fourcc=fourcc)
        self.black_warned = False
        if cap is None:
            self.log.emit(f"카메라 {index}번을 열 수 없습니다.")
            return
        if not retry:
            last = self.profiles.last_camera
            if last and last != name:
                self.log.emit(f"지난번 카메라 '{last}' 와 다른 카메라입니다 → '{name}' 의 저장된 스펙으로 엽니다.")
            fps = f"측정 {spec['fps']:.0f}fps" if spec["fps"] else "미측정 → 3초 뒤 실측해 스펙에 추가"
            model = f", 모델 {prof['model']}" if prof and prof.get("model") else ""
            self.log.emit(f"카메라 설정 [{RESOLUTION_NAMES.get(key, key)}]: {fourcc} {ow}x{oh} ({fps}{model})")
        if name != self.cam_key:
            self.clear_calibration()  # 렌즈가 다르면 보정값도 다르다
            self.load_saved_calibration(name)
        self.cam_key = name
        self.open_spec = spec
        self.mode_recorded = spec["measured"] or prof is None
        if prof is not None:
            self.profiles.mark_used(name, key)
        self.resolution_options.value = resolution_options(prof, self.request_size)
        self.start_camera = index

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cam_name = dict(self.cameras.value).get(index, f"[{index}]").rsplit(" (", 1)[0]
        self.cam_fps = cap.get(cv2.CAP_PROP_FPS)
        self.cam_fourcc = fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        self.backend = cap.getBackendName()
        self.frame_no = 0
        self.shown_id = 0
        self.started = time.time()
        self.last_res = None
        self.analysis_out = None
        self.slow_warned = False
        self.reset_focus_peak(quiet=True)       # 해상도·카메라가 바뀌면 선명도 기준도 달라진다
        self.stream = CameraStream(cap)
        cam_fps = f"{self.cam_fps:.0f}" if self.cam_fps > 0 else "알 수 없음"
        self.log.emit(f"카메라 {index}번 연결 ({w}x{h} {self.cam_fourcc}, 설정 FPS {cam_fps}). "
                      "Space/s = 캡처, 숫자키/Tab = 카메라 전환, d = 분석, a = 보드 자동 인식, "
                      "c = 자동 보정, u = 보정 화면, q/ESC = 종료")

    def release(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.frame = None
        self.last_res = None

    # =================================================================
    # 토글 / 보정값
    # =================================================================
    def toggle_guide(self):
        self.guide_on.value = not self.guide_on.value
        if self.guide_on.value:
            m = self.guide_mode.value
            self.log.emit(f"가이드선 켜짐: {GUIDE_MODE_NAMES[m]} — {GUIDE_MODE_HELP[m]}. "
                          "보정값이 있으면 원본 화면에서는 렌즈 휨을 예측한 곡선, 보정 화면에서는 반듯한 선으로 그립니다.")
        else:
            self.log.emit("가이드선 꺼짐")

    def set_guide_mode(self, mode):
        """가이드 종류를 고르면 가이드선을 켠다"""
        if mode not in GUIDE_MODES:
            return
        self.guide_mode.value = mode
        self.guide_on.value = True
        self.log.emit(f"가이드선 종류: {GUIDE_MODE_NAMES[mode]} — {GUIDE_MODE_HELP[mode]}")

    def next_guide_mode(self):
        i = GUIDE_MODES.index(self.guide_mode.value)
        self.set_guide_mode(GUIDE_MODES[(i + 1) % len(GUIDE_MODES)])

    # ---------- 초점 맞추기 도우미 ----------
    def toggle_focus_assist(self):
        self.focus_on.value = not self.focus_on.value
        if self.focus_on.value:
            self.reset_focus_peak(quiet=True)
            self.log.emit("초점 도우미 켜짐: 가운데 상자 안의 선명도를 막대로 보여 줍니다. 렌즈 링을 끝에서 끝까지 "
                          "한 번 천천히 돌려 최고값을 잡은 뒤, 막대가 최고값(흰 선)에 닿는 위치로 돌아오면 초점이 맞은 것입니다. "
                          "(이 카메라는 렌즈 모터가 없어 오토포커스는 불가)")
        else:
            self.focus_text.value = ""
            self.log.emit("초점 도우미 꺼짐")

    def toggle_peaking(self):
        self.peaking_on.value = not self.peaking_on.value
        self.log.emit("포커스 피킹 켜짐: 초점이 맞은 윤곽을 빨간색으로 칠합니다. 빨간 윤곽이 가장 많아지는 위치가 초점입니다."
                      if self.peaking_on.value else "포커스 피킹 꺼짐")

    def next_peaking_level(self):
        levels = list(PEAKING_LEVELS)
        self.peaking_level.value = levels[(levels.index(self.peaking_level.value) + 1) % len(levels)]
        self.peaking_on.value = True
        self.log.emit(f"피킹 감도: {self.peaking_level.value} (높음 = 약한 윤곽까지 칠함, 낮음 = 아주 선명한 윤곽만)")

    def reset_focus_peak(self, quiet=False):
        self.focus.reset()
        self.focus_last_verdict = None
        if not quiet:
            self.log.emit("선명도 최고값을 초기화했습니다. 렌즈 링을 다시 끝에서 끝까지 돌려 최고값을 잡으세요.")

    def _update_focus(self):
        now = time.time()
        if not self.focus_on.value or self.frame is None or now - self.focus_last < FOCUS_INTERVAL:
            return
        self.focus_last = now
        self.focus.update(self.frame)
        f = self.focus
        en, ko = f.verdict()
        arrow = "▲" if f.trend > f.score * 0.002 else "▼" if f.trend < -f.score * 0.002 else "-"
        self.focus_text.value = f"선명도 {f.score:.0f} / 최고 {f.peak:.0f} ({f.ratio * 100:.0f}%) {arrow} {ko}"
        if ko != self.focus_last_verdict and f.peak > 0 and now - self.started > 2:
            if ko == "초점 맞음" and self.focus_last_verdict is not None:
                self.log.emit(f"초점 맞음 (선명도 {f.score:.0f}, 최고값의 {f.ratio * 100:.0f}%)")
            self.focus_last_verdict = ko

    def toggle_analyze(self):
        self.analyze_on.value = not self.analyze_on.value
        self.log.emit(f"왜곡 분석 {'켜짐' if self.analyze_on.value else '꺼짐'}")
        if self.analyze_on.value and self.analyzer.board is None and not self.auto_board_on.value:
            # 체커보드 크기를 모르면 분석할 수 없으므로 크기 자동 인식을 함께 켠다
            self.toggle_auto_board()
            self.log.emit("체커보드 크기를 아직 몰라 자동 인식도 켰습니다. 보드 전체가 화면에 보이게 비춰 주세요.")

    def toggle_auto_board(self):
        self.auto_board_on.value = not self.auto_board_on.value
        self.auto_candidate = None
        if self.auto_board_on.value:
            self.log.emit("체커보드 크기 자동 인식 켜짐")
        elif self.analyzer.board is None:
            with self.analyzer_lock:
                self.analyzer.set_board(DEFAULT_BOARD)
            self.board.value = self.analyzer.board
            self.log.emit(f"체커보드 크기 자동 인식 꺼짐 → 기본 크기 {DEFAULT_BOARD[0]}x{DEFAULT_BOARD[1]}로 고정")
        else:
            b = self.analyzer.board
            self.log.emit(f"체커보드 크기 자동 인식 꺼짐 → 현재 크기 {b[0]}x{b[1]}로 고정")

    def toggle_auto_correct(self):
        self.auto_correct_on.value = not self.auto_correct_on.value
        self.log.emit(f"자동 보정 {'켜짐 (보정값이 안정되면 보정 화면을 자동으로 켭니다)' if self.auto_correct_on.value else '꺼짐'}")
        if self.auto_correct_on.value and self.analyzer.calib is not None and calib_is_stable(self.analyzer.calib):
            self.set_undistort(True)

    def toggle_undistort(self):
        if self.analyzer.calib is None:
            self.log.emit(f"보정값이 아직 없습니다. 체커보드를 여러 위치에서 비춰 샘플을 "
                          f"{DistortionAnalyzer.MIN_SAMPLES}장 이상 모으세요. "
                          f"(현재 {len(self.analyzer.samples)}장)")
            return
        self.set_undistort(not self.undistort_on.value)

    def set_undistort(self, on):
        if on == self.undistort_on.value:
            return
        self.undistort_on.value = on
        self.log.emit(f"보정(왜곡 제거) 화면 {'켜짐' if on else '꺼짐'}")

    def clear_calibration(self):
        with self.analyzer_lock:
            self.analyzer.reset()
        self.last_res = None
        self.analysis_out = None
        self.lens = None
        self.undistort_on.value = False

    def reset_analysis(self):
        self.clear_calibration()
        self.log.emit("왜곡 분석 샘플과 보정값을 초기화했습니다.")

    def calibration_path(self, name):
        prof = self.profiles.get(name) or {}
        return os.path.join(self.save_dir, prof.get("calibration_file") or f"calibration_{safe_name(name)}.json")

    def load_saved_calibration(self, name):
        """카메라 이름별로 저장된 보정값이 있으면 불러온다"""
        path = self.calibration_path(name)
        if not os.path.exists(path):
            return
        try:
            with self.analyzer_lock:
                self.analyzer.load(path)
        except (OSError, ValueError, KeyError) as e:
            self.log.emit(f"보정값 불러오기 실패: {path} ({e})")
            return
        self.board.value = self.analyzer.board
        calib = self.analyzer.calib
        self.lens = classify_lens(calib) if calib_is_stable(calib) else None
        self.log.emit(f"'{name}' 보정값을 불러왔습니다: {os.path.basename(path)} "
                      f"(RMS {calib['rms']:.3f}px, {calib['size'][0]}x{calib['size'][1]}, U = 보정 화면)")
        if self.auto_correct_on.value and self.lens is not None:
            self.set_undistort(True)

    def save_calibration(self):
        if self.analyzer.calib is None:
            self.log.emit("저장할 보정값이 없습니다.")
            return
        name = self.cam_key or f"Camera {self.start_camera}"
        fname = f"calibration_{safe_name(name)}.json"      # 카메라 이름별 (번호는 연결 순서에 따라 바뀜)
        path = os.path.join(self.save_dir, fname)
        self.analyzer.save(path)
        self.profiles.ensure(name)["calibration_file"] = fname
        self.profiles.save()
        self.log.emit(f"보정값 저장: {path} ('{name}' 을 다시 열면 자동으로 불러옵니다)")

    def save_checkerboard(self):
        board = self.analyzer.board or DEFAULT_BOARD
        path = os.path.join(self.save_dir, f"checkerboard_{board[0]}x{board[1]}.png")
        if write_image(path, make_checkerboard(board)):
            self.log.emit(f"체커보드 이미지 저장: {path} (인쇄하거나 다른 화면에 띄워 카메라에 비추세요)")
        else:
            self.log.emit("체커보드 이미지 저장 실패")

    def capture(self):
        if self.frame is None:
            self.log.emit("캡처할 영상이 없습니다.")
            return
        path = save_frame(self.frame, self.capture_dir)
        if path:
            self.capture_count.value += 1
            self.log.emit(f"저장 완료 ({self.capture_count.value}): {path}")
            self.calibrate_from_captures(auto=True)     # 최근 캡처 기준으로 화면 보정 갱신 (보드가 있을 때만)
        else:
            self.log.emit("저장 실패")

    # ---------- 캡처로 보정 ----------
    def calibrate_from_captures(self, auto=False):
        """
        최근 캡처(현재 해상도와 같은 것, 최대 CAPTURE_CALIB_MAX 장)로 렌즈 보정을 계산해 화면 보정에 쓴다.
        auto=True (캡처 직후): 체커보드가 보이는 캡처가 없으면 조용히 넘어간다
        """
        if self.capture_calib_job is not None:
            if not auto:
                self.log.emit("캡처로 보정 계산 중입니다. 잠시만 기다리세요.")
            return
        if self.frame is None:
            self.log.emit("카메라가 열려 있어야 캡처로 보정할 수 있습니다 (현재 해상도 기준).")
            return
        h, w = self.frame.shape[:2]
        try:
            files = sorted((os.path.join(self.capture_dir, f) for f in os.listdir(self.capture_dir)
                            if f.lower().endswith(".png")), key=os.path.getmtime, reverse=True)
        except OSError:
            files = []
        job = {"result": None, "done": False, "auto": auto, "size": (w, h)}
        self.capture_calib_job = job
        if not auto:
            self.log.emit(f"최근 캡처로 보정 계산 중 ({w}x{h} 캡처 최대 {CAPTURE_CALIB_MAX}장)...")

        def work():
            imgs = []
            for p in files:
                img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[1::-1] == (w, h):
                    imgs.append((os.path.basename(p), img))
                if len(imgs) >= CAPTURE_CALIB_MAX:
                    break
            job["result"] = calibrate_from_images(imgs) if imgs else {"error": f"{w}x{h} 캡처가 없습니다"}
            job["done"] = True
        threading.Thread(target=work, daemon=True).start()

    def _finish_capture_calib(self, job):
        r = job["result"]
        if "error" in r:
            if not job["auto"]:
                self.log.emit(f"캡처로 보정 못 함: {r['error']}. 체커보드가 화면 가득 보이게 캡처(Space)하세요.")
            return
        if self.frame is None or self.frame.shape[1::-1] != r["calib"]["size"]:
            self.log.emit("해상도가 바뀌어 캡처 보정 결과를 적용하지 않았습니다.")
            return
        with self.analyzer_lock:
            self.analyzer.set_calibration(r["calib"], r["board"])
        self.board.value = self.analyzer.board
        calib = self.analyzer.calib
        self.lens = classify_lens(calib) if calib_is_stable(calib) else None
        b, a = r["err_before"], r["err_after"]
        k = calib["dist"]
        self.log.emit(f"최근 캡처 {len(r['used'])}장으로 화면 보정 ({', '.join(r['used'][:3])}{' …' if len(r['used']) > 3 else ''}): "
                      f"직선 오차 최대 {b[0]:.1f}px·평균 {b[1]:.1f}px → 최대 {a[0]:.1f}px·평균 {a[1]:.1f}px, "
                      f"k1 {k[0]:+.4f} k2 {k[1]:+.4f}"
                      + (f" (보드가 안 보여 뺀 캡처 {len(r['skipped'])}장)" if r["skipped"] else ""))
        self.save_calibration()
        self.set_undistort(True)

    def shutdown(self):
        self.running = False
        if self.probe_job is not None:
            self.probe_job["stop"] = True
        self.release()

    # =================================================================
    # 체커보드 크기 자동 인식
    # =================================================================
    def _auto_board_step(self, frame, found):
        """현재 크기로 보드를 못 찾는 동안 백그라운드에서 크기를 검색하고, 연속 2번 같으면 적용"""
        now = time.time()
        if found:
            self.last_seen = now
            self.auto_candidate = None
            return
        if not self.auto_board_on.value:
            return

        if self.auto_result is not None:
            size = self.auto_result[0]
            self.auto_result = None
            if size is None or size == self.analyzer.board:
                self.auto_candidate = None
            elif size == self.auto_candidate:
                old = self.analyzer.board
                with self.analyzer_lock:
                    self.analyzer.set_board(size)
                self.last_res = None
                self.analysis_out = None
                self.lens = None
                self.undistort_on.value = False
                self.auto_candidate = None
                self.last_seen = now
                self.board.value = size
                change = f" ({old[0]}x{old[1]} → 변경, 샘플 초기화)" if old else ""
                self.log.emit(f"체커보드 자동 인식: 내부 코너 {size[0]}x{size[1]} "
                              f"(칸 {size[0] + 1}x{size[1] + 1}){change}")
                return
            else:
                self.auto_candidate = size

        if self.analyzer.board is not None and now - self.last_seen < AUTO_SEARCH_DELAY:
            return
        if self.auto_busy or now - self.auto_last_try < AUTO_SEARCH_INTERVAL:
            return
        self.auto_busy = True
        self.auto_last_try = now
        threading.Thread(target=self._auto_search, args=(frame.copy(),), daemon=True).start()

    def _auto_search(self, frame):
        self.auto_result = [detect_board_size(frame)]
        self.auto_busy = False

    # =================================================================
    # 왜곡 분석 (백그라운드) / 결과 반영
    # =================================================================
    def _analysis_loop(self):
        """분석 스레드: 새 프레임이 들어오면 왜곡 분석. 분석이 느려도 영상 표시 FPS에는 영향 없음"""
        last_id, count, since = -1, 0, time.time()
        while self.running:
            stream = self.stream
            if stream is None or not self.analyze_on.value:
                time.sleep(0.02)
                continue
            fid, frame = stream.latest()
            if frame is None or fid == last_id:
                time.sleep(0.003)
                continue
            last_id = fid
            with self.analyzer_lock:
                res = self.analyzer.analyze(frame)
            res["frame"] = frame
            res["stream"] = stream  # 카메라 전환 전 결과는 버리기 위해
            self.analysis_out = res
            count += 1
            now = time.time()
            if now - since >= 0.5:
                self.analysis_fps = count / (now - since)
                count, since = 0, now

    def _post_analysis(self, res):
        """분석 결과에 따른 후처리 (메시지/자동 인식/자동 보정). GUI 스레드(tick)에서 호출"""
        self._auto_board_step(res["frame"], res["found"])
        if res["found"] != self.board_visible:
            self.board_visible = res["found"]
            self.log.emit("체커보드 감지됨 - 분석 중" if res["found"] else "체커보드가 화면에서 사라졌습니다")
        if res["new_sample"]:
            n = len(self.analyzer.samples)
            need = DistortionAnalyzer.MIN_SAMPLES
            more = f" (보정까지 {need - n}장 더)" if n < need else ""
            self.log.emit(f"샘플 {n}/{DistortionAnalyzer.MAX_SAMPLES}장 수집{more}. 위치/각도를 바꿔 보세요.")
        new = self.analyzer.new_calib
        if new is not None:
            self.analyzer.new_calib = None
            self._on_calibrated(new)

    def _on_calibrated(self, new):
        if "error" in new:
            self.log.emit(f"보정 실패: {new['error']}")
            return
        if "rms" not in new or self.analyzer.calib is not new:
            return
        k1, k2, p1, p2, k3 = new["dist"]
        K = new["K"]
        self.log.emit(f"보정 계산 ({new['n']}장): RMS {new['rms']:.3f}px, "
                      f"k1={k1:+.4f} k2={k2:+.4f} p1={p1:+.5f} p2={p2:+.5f} k3={k3:+.4f}, "
                      f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
        if not calib_is_stable(new):
            self.lens = None
            self.log.emit("주의: 보정값이 불안정합니다. 체커보드를 앞뒤/좌우로 기울이고 "
                          "화면 가장자리까지 옮겨 가며 샘플을 더 모으세요. (필요하면 R로 초기화)")
            return
        self.lens = classify_lens(new)
        L = self.lens
        msg = (f"렌즈 분석: {L['kind']}, 강도 {L['level']} "
               f"(영상 모서리 {L['corner_pct']:+.2f}%, 약 {L['corner_px']:.1f}px 이동)")
        msg += ", 렌즈 기울어짐(접선 왜곡) 있음" if L["tilted"] else ", 렌즈 기울어짐 거의 없음"
        self.log.emit(msg)
        if self.auto_correct_on.value:
            if not self.undistort_on.value:
                self.log.emit("자동 보정: 계산된 계수로 왜곡 제거 화면을 적용합니다. (U로 원본 비교)")
            self.set_undistort(True)

    # =================================================================
    # tick: GUI 스레드에서 자주 호출 (View 의 타이머)
    # =================================================================
    def tick(self):
        cjob = self.capture_calib_job
        if cjob is not None and cjob["done"]:   # 캡처로 보정 결과 반영 (GUI 스레드에서)
            self.capture_calib_job = None
            self._finish_capture_calib(cjob)

        job = self.probe_job
        if job is not None:
            while job["msgs"]:                  # 측정 진행 메시지
                self.log.emit(job["msgs"].pop(0))
            if job["done"]:
                self.probe_job = None
                self._finish_probe(job)

        stream = self.stream
        if stream is not None and stream.failed:
            self.log.emit("프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
            self.release()
            return
        if stream is None:
            return

        out = self.analysis_out
        if out is not None:
            self.analysis_out = None
            if out["stream"] is stream and self.analyze_on.value:
                self._post_analysis(out)
                self.last_res = out

        fid, frame = stream.latest()
        if frame is not None and fid != self.shown_id:
            self.shown_id = fid
            self.frame = frame
            self.frame_no = fid
            self._update_focus()                # 그리기 전에 선명도 갱신
            self.frame_id.value = fid           # View 가 다시 그림
            # 연 직후 첫 프레임은 검은색일 수 있으므로(ELP 5MP 카메라 실측) 30장(약 0.5초분) 넘게 받아도 검으면 알림
            if not self.black_warned and fid > 30 and frame.max() == 0:
                self.log.emit("알림: 영상이 완전히 검은색입니다. (가상 카메라/가림막/카메라 권한 확인)")
                self.black_warned = True

        elapsed = time.time() - self.started
        # 처음 써 보는 해상도: 3초 뒤 실측 FPS 를 스펙에 기록
        if not self.mode_recorded and elapsed > 3 and stream.fps > 0 and self.frame is not None:
            self.mode_recorded = True
            h, w = self.frame.shape[:2]
            prof = self.profiles.ensure(self.cam_key)
            ok, why = self._recordable(prof, w, h, stream.fps)
            if ok:
                m = record_mode(prof, self.open_spec["request_fourcc"], w, h, self.cam_fourcc, stream.fps)
                self.profiles.save()
                self.resolution_options.value = resolution_options(prof, self.request_size)
                self.log.emit(f"실측 FPS 를 스펙에 추가: {m['fourcc']} {w}x{h} {m['fps']:.0f}fps")
            else:
                self.log.emit(f"주의: {why} → 스펙에 기록하지 않았습니다. USB 케이블·포트(본체 뒷면 USB 3)를 확인하세요.")
        # 카메라 자체가 느리게 보내는 경우 (코드/분석과 무관). 일부 카메라(가상 카메라 등)는
        # 열 때마다 속도가 달라지므로 3초 뒤에도 5 FPS 미만이면 한 번 자동으로 다시 연결한다
        if not self.reopen_tried and elapsed > 3 and 0 < stream.fps < 5:
            self.log.emit(f"카메라 FPS가 {stream.fps:.1f}로 너무 낮아 자동으로 다시 연결합니다...")
            self.select_camera(self.start_camera, retry=True)
        elif not self.slow_warned and elapsed > 4 and 0 < stream.fps < 10:
            self.slow_warned = True
            self.log.emit(f"알림: 카메라가 초당 {stream.fps:.1f}장만 보내고 있습니다. 카메라 쪽 문제입니다 "
                          "(가상 카메라/조명 부족에 따른 자동 노출/USB 대역폭). [새로고침]으로 다시 열어 보세요.")

    def _recordable(self, prof, w, h, fps):
        """
        실측 FPS 를 스펙에 기록해도 되는지. 연결이 나쁠 때(검은 화면, 아주 낮은 FPS) 잰 값이 스펙에 남지 않게 한다
        (2026-11-13: 다시 꽂은 뒤 USB 연결 불량으로 1920x1080 이 1fps·검은 화면 → '1fps' 로 기록된 일이 있었음)
        """
        if self.frame.max() == 0:
            return False, f"{w}x{h} 영상이 검은색입니다 (수신 {fps:.1f}fps)"
        adv = max((a["max_fps"] for a in prof.get("advertised", []) if (a["width"], a["height"]) == (w, h)), default=0)
        limit = adv * MIN_RECORD_RATIO if adv else MIN_RECORD_FPS
        if fps < limit:
            ref = f"카메라가 알려 준 최대 {adv:.0f}fps 의 {MIN_RECORD_RATIO * 100:.0f}%" if adv else f"{MIN_RECORD_FPS}fps"
            return False, f"{w}x{h} 수신 FPS {fps:.1f} 가 너무 낮습니다 ({ref} 미만)"
        return True, ""

    # =================================================================
    # View 가 그릴 때 쓰는 값
    # =================================================================
    def display_frame(self):
        """화면에 보낼 영상 (보정 화면이면 왜곡 제거)"""
        if self.frame is None:
            return None
        return self.analyzer.undistort(self.frame) if self.undistort_on.value else self.frame

    def render_state(self):
        a = self.analyzer
        prof = self.profiles.get(self.cam_key) or {} if self.cam_key else {}
        return SimpleNamespace(
            camera_name=self.cam_name, model=prof.get("model", ""), backend=self.backend,
            fourcc=self.cam_fourcc, setting_fps=self.cam_fps,
            camera_fps=self.stream.fps if self.stream is not None else 0.0,
            analysis_fps=self.analysis_fps, analyze_on=self.analyze_on.value,
            frame_no=self.frame_no, elapsed=int(time.time() - self.started),
            brightness=float(self.frame.mean()) if self.frame is not None else 0.0,
            captured=self.capture_count.value,
            board=a.board, auto_board_on=self.auto_board_on.value, samples=len(a.samples),
            min_samples=a.MIN_SAMPLES, max_samples=a.MAX_SAMPLES, calibrating=a.calibrating, calib=a.calib,
            lens=self.lens, undistort_on=self.undistort_on.value, auto_correct_on=self.auto_correct_on.value,
            guide_on=self.guide_on.value, guide_mode=self.guide_mode.value,
            focus_on=self.focus_on.value, peaking_on=self.peaking_on.value,
            peaking_threshold=PEAKING_LEVELS[self.peaking_level.value],
            focus_score=self.focus.score, focus_peak=self.focus.peak, focus_ratio=self.focus.ratio,
            focus_trend=self.focus.trend, focus_verdict=self.focus.verdict()[0],
            res=self.last_res if self.analyze_on.value else None)
