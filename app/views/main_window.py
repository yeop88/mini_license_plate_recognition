# -*- coding: utf-8 -*-
"""
View — tkinter 메인 창

    위: 도구 줄 — [카메라 설정] [인식 설정] (누르면 메인 창 오른쪽에 별도 설정 창이 뜨고, 다시 누르면 닫힘. 둘 다 열어도 됨),
        인식 시작/멈춤, 캡처, 종료, 상태
    가운데: 영상 (카메라 화면 또는 인식 결과 화면)
    영상 아래: 재생바 (입력이 영상 파일일 때만)
    아래: 메시지 (카메라·인식 메시지 함께)

무엇을 보여 주나
    카메라 + 인식 꺼짐 : 카메라 영상 (보정 화면이면 왜곡 제거) + 카메라 정보·가이드선·초점 도우미
    카메라 + 인식 켜짐 : 인식 결과 이미지(박스·ID·값) 위에 같은 카메라 정보·가이드선·초점 도우미
    영상 파일          : 인식 결과 이미지 (재생바로 이동)
"""
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from app.views import overlays
from app.views.camera_panel import CameraPanel
from app.views.recognition_panel import RecognitionPanel
from app.views.side_panel import PANEL_WIDTH

TICK_MS = 5                  # ViewModel.tick 호출 간격
TEXT_WIDGETS = ("Entry", "TEntry", "Spinbox", "TSpinbox", "Text")   # 글자 입력 중에는 단축키를 쓰지 않음


class MainWindow:
    SEEK_THROTTLE_S = 0.12   # 재생바 드래그 중 이동 명령 간격

    def __init__(self, root, cam_vm, rec_vm):
        self.root, self.cam, self.rec = root, cam_vm, rec_vm
        self.view_fps = 0.0
        self.view_frames = 0
        self.view_since = time.time()
        self.photo = None
        self._dragging = False
        self._last_seek = 0.0

        root.title("카메라 번호판 인식")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.max_w = max(640, min(1280, sw - PANEL_WIDTH - 60))     # 영상 표시 최대 크기 (설정 창을 열어도 들어가게)
        self.max_h = max(360, sh - 330)
        self._build()
        self._bind_viewmodels()
        self._bind_keys()

        cam_vm.start()
        root.after(100, cam_vm.refresh)  # 창이 먼저 뜬 다음 카메라 검색
        root.after(30, self._tick)

    # ---------------- 위젯 ----------------
    def _build(self):
        root = self.root
        top = ttk.Frame(root, padding=(6, 6, 6, 4))
        top.pack(fill="x")
        self.btn_cam_panel = ttk.Button(top, command=lambda: self.toggle_panel("camera"))
        self.btn_cam_panel.pack(side="left", padx=2)
        self.btn_rec_panel = ttk.Button(top, command=lambda: self.toggle_panel("recognition"))
        self.btn_rec_panel.pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        self.btn_play = ttk.Button(top, width=16, command=self.rec.toggle_play)
        self.btn_play.pack(side="left", padx=2)
        ttk.Button(top, text="카메라 캡처 (Space)", command=self.cam.capture).pack(side="left", padx=2)
        ttk.Button(top, text="종료 (Q)", command=self.quit).pack(side="left", padx=2)
        self.status = ttk.Label(top, text="")
        self.status.pack(side="right")
        self.rec_status = ttk.Label(top, text="", foreground="#0050a0")
        self.rec_status.pack(side="right", padx=12)

        # 아래: 메시지 (영상보다 먼저 pack 해야 창이 작아져도 남는다)
        bottom = ttk.LabelFrame(root, text="메시지", padding=4)
        bottom.pack(side="bottom", fill="x", padx=6, pady=6)
        self.log_box = tk.Text(bottom, height=6, state="disabled", wrap="word", font=("Malgun Gothic", 9))
        scroll = ttk.Scrollbar(bottom, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_box.pack(side="left", fill="x", expand=True)

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=6)
        # 설정 창 (별도 창, 처음엔 숨김). 버튼으로 열고 닫음, 창의 X 도 닫기
        self.panels = {"camera": CameraPanel(root, self.cam, lambda: self.toggle_panel("camera")),
                       "recognition": RecognitionPanel(root, self.rec, lambda: self.toggle_panel("recognition"))}
        # 영상 + 재생바
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.video = tk.Label(left, bg="black")
        self.video.pack(anchor="n")
        self._set_photo(np.zeros((int(self.max_w * 9 / 16), self.max_w, 3), np.uint8))   # 픽셀 크기로 자리 잡기
        self.seekbar = ttk.Frame(left)
        self.btn_play2 = ttk.Button(self.seekbar, text="▶", width=3, command=self.rec.toggle_play)
        self.btn_play2.pack(side="left")
        self.seek_var = tk.DoubleVar(value=0)
        self.seek = ttk.Scale(self.seekbar, from_=0, to=1, orient="horizontal", variable=self.seek_var)
        self.seek.pack(side="left", fill="x", expand=True, padx=6)
        self.time_lbl = ttk.Label(self.seekbar, text="", font=("Consolas", 11), width=17)
        self.time_lbl.pack(side="left")
        # 마우스 위치 → 프레임 (ttk.Scale 기본 동작은 클릭 시 한 칸씩 이동이라 직접 처리)
        self.seek.bind("<Button-1>", self._on_seek_press)
        self.seek.bind("<B1-Motion>", self._on_seek_drag)
        self.seek.bind("<ButtonRelease-1>", self._on_seek_release)
        self._show_panel_buttons()

    def toggle_panel(self, name):
        """설정 창(별도 창) 열기/닫기. 열려 있으면 닫고, 닫혀 있으면 메인 창 오른쪽에 띄움. 두 창을 함께 열어도 됨"""
        p = self.panels[name]
        if p.shown:
            p.withdraw()
        else:
            other = [q for k, q in self.panels.items() if k != name and q.shown]
            p.show_beside(self.root, offset=40 if other else 0)   # 다른 설정 창이 떠 있으면 조금 비켜서
        self._show_panel_buttons()

    def _show_panel_buttons(self):
        cam_on, rec_on = self.panels["camera"].shown, self.panels["recognition"].shown
        self.btn_cam_panel.configure(text="✕ 카메라 설정 닫기" if cam_on else "⚙ 카메라 설정")
        self.btn_rec_panel.configure(text="✕ 인식 설정 닫기" if rec_on else "⚙ 인식 설정")

    # ---------------- ViewModel 구독 ----------------
    def _bind_viewmodels(self):
        cam, rec = self.cam, self.rec
        cam.log.subscribe(self._append_log)
        rec.log.subscribe(lambda m: self._append_log(f"[인식] {m}"))
        cam.capture_count.subscribe(lambda n: self.status.configure(text=f"카메라 캡처 {n}장"))
        cam.frame_id.subscribe(lambda _: self._on_camera_frame(), call_now=False)
        rec.image.subscribe(self._on_rec_image, call_now=False)
        rec.status.subscribe(lambda t: self.rec_status.configure(text=t))
        rec.playing.subscribe(lambda _: self._show_play())
        rec.source.subscribe(self._on_source)
        rec.total.subscribe(lambda n: self.seek.configure(to=max(1, n - 1)))
        rec.position.subscribe(self._on_position)

    def _bind_keys(self):
        cam, root = self.cam, self.root

        def key(seq, fn):
            def handler(e):
                if e.widget.winfo_class() in TEXT_WIDGETS:
                    return None                    # 숫자 입력 칸 등에서는 단축키 대신 글자 입력
                fn()
                return "break" if seq == "<Tab>" else None
            for w in (root, *self.panels.values()):  # 설정 창에 초점이 있어도 같은 단축키
                w.bind(seq, handler)
        key("<space>", cam.capture)
        key("s", cam.capture)
        key("q", self.quit)
        key("<Escape>", self.quit)
        key("d", cam.toggle_analyze)
        key("a", cam.toggle_auto_board)
        key("c", cam.toggle_auto_correct)
        key("u", cam.toggle_undistort)
        key("r", cam.reset_analysis)
        key("g", cam.toggle_guide)
        key("v", cam.next_guide_mode)
        key("k", cam.calibrate_from_captures)
        key("f", cam.toggle_focus_assist)
        key("p", cam.toggle_peaking)
        key("i", self.rec.toggle_play)
        for n in range(10):                        # 카메라 전환: 숫자키 = 해당 번호, Tab = 다음 카메라
            key(str(n), lambda n=n: cam.switch_to(n))
        key("<Tab>", cam.next_camera)

    # ---------------- 위젯 갱신 ----------------
    def _append_log(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}\n"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.root.update_idletasks()               # 카메라 검색처럼 오래 걸리는 동작 중에도 메시지가 보이게

    def _show_play(self):
        on, video = self.rec.playing.value, self.rec.source.value == "video"
        if video:
            self.btn_play.configure(text="⏸ 일시정지" if on else "▶ 영상 재생")
        else:
            self.btn_play.configure(text="⏸ 인식 멈춤 (I)" if on else "▶ 인식 시작 (I)")
        self.btn_play2.configure(text="⏸" if on else "▶")

    def _on_source(self, kind):
        if kind == "video":
            self.seekbar.pack(fill="x", pady=(6, 0))
        else:
            self.seekbar.pack_forget()
        self._show_play()

    def _on_position(self, fi):
        if not self._dragging and self.rec.source.value == "video":
            self.seek_var.set(fi)
            self.time_lbl.config(text=self.rec.time_text(fi))

    # ---------------- 영상 ----------------
    def _tick(self):
        self.cam.tick()
        self.rec.tick()
        if self.cam.running:
            self.root.after(TICK_MS, self._tick)

    def _count_view_fps(self):
        self.view_frames += 1
        now = time.time()
        if now - self.view_since >= 0.5:
            self.view_fps = self.view_frames / (now - self.view_since)
            self.view_frames, self.view_since = 0, now

    def _fit_width(self, frame):
        h, w = frame.shape[:2]
        return int(min(self.max_w, self.max_h * w / h))

    def _on_camera_frame(self):
        """새 카메라 프레임: 영상 파일 입력이거나 인식 결과가 들어오는 중이면 그리지 않음 (결과 화면을 보여 줌)"""
        if self.rec.source.value != "camera" or self.rec.recognition_active():
            return
        frame = self.cam.display_frame()
        if frame is not None:
            self._show_camera_view(frame)

    def _on_rec_image(self, img):
        if img is None:
            return
        if self.rec.source.value == "camera":
            self._show_camera_view(img)            # 인식 결과 위에 카메라 정보·가이드선·초점 도우미
            return
        self._count_view_fps()
        w = self._fit_width(img)
        self._set_photo(cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA))

    def _show_camera_view(self, frame):
        self._count_view_fps()
        view = overlays.compose(frame, self.cam.render_state(), self._fit_width(frame), self.view_fps)
        self._set_photo(view)

    def _set_photo(self, bgr):
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=self.photo)

    # ---------------- 재생바: 클릭 = 그 위치로 점프, 드래그 = 따라 이동, 놓으면 마지막 위치로 ----------------
    def _x_to_frame(self, x):
        pad = 8                                    # 손잡이 반 폭 정도 (양 끝에서도 0·마지막 프레임에 닿게)
        w = max(1, self.seek.winfo_width() - 2 * pad)
        return round(max(0.0, min(1.0, (x - pad) / w)) * (self.rec.total.value - 1))

    def _on_seek_press(self, e):
        self._dragging = True
        fi = self._x_to_frame(e.x)
        self.seek_var.set(fi)
        self._send_seek(fi, force=True)
        return "break"

    def _on_seek_drag(self, e):
        fi = self._x_to_frame(e.x)
        self.seek_var.set(fi)
        self._send_seek(fi)
        return "break"

    def _on_seek_release(self, e):
        fi = self._x_to_frame(e.x)
        self.seek_var.set(fi)
        self._send_seek(fi, force=True)
        self._dragging = False
        return "break"

    def _send_seek(self, fi, force=False):
        fi = int(round(float(fi)))
        self.time_lbl.config(text=self.rec.time_text(fi))
        now = time.perf_counter()
        if force or now - self._last_seek >= self.SEEK_THROTTLE_S:
            self._last_seek = now
            self.rec.seek(fi)

    def quit(self):
        self.rec.close()
        self.cam.shutdown()
        self.root.after(200, self.root.destroy)
