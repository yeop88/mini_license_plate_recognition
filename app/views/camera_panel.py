# -*- coding: utf-8 -*-
"""
카메라 설정 창 (별도 창) — CameraViewModel 의 상태를 구독해 표시하고, 버튼은 ViewModel 명령으로 넘긴다
    카메라·해상도 / 렌즈 왜곡 보정 / 가이드선 / 초점 도우미
"""
from tkinter import ttk

from app.viewmodels.camera_viewmodel import GUIDE_MODE_HELP, GUIDE_MODE_NAMES, GUIDE_MODES
from app.views.side_panel import SidePanel

WRAP = 420


class CameraPanel(SidePanel):
    def __init__(self, parent, vm, on_close):
        super().__init__(parent, "카메라 설정", on_close)
        self.vm = vm
        self._build(self.body)
        self._bind_vm()

    def _build(self, body):
        vm = self.vm
        # ---- 카메라 / 해상도 ----
        f = ttk.LabelFrame(body, text="카메라", padding=6)
        f.pack(fill="x", pady=(0, 6))
        ttk.Label(f, text="카메라").grid(row=0, column=0, sticky="w")
        self.combo = ttk.Combobox(f, state="readonly", width=42)
        self.combo.grid(row=0, column=1, sticky="we", padx=4, pady=2)
        self.combo.bind("<<ComboboxSelected>>", self._on_camera_select)
        ttk.Label(f, text="해상도").grid(row=1, column=0, sticky="w")
        self.res_combo = ttk.Combobox(f, state="readonly", width=42)
        self.res_combo.grid(row=1, column=1, sticky="we", padx=4, pady=2)
        self.res_combo.bind("<<ComboboxSelected>>", self._on_resolution_select)
        b = ttk.Frame(f)
        b.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(b, text="새로고침", command=vm.refresh).pack(side="left", padx=2)
        ttk.Button(b, text="스펙 다시 측정", command=vm.reprobe).pack(side="left", padx=2)
        ttk.Button(b, text="캡처 (Space)", command=vm.capture).pack(side="left", padx=2)
        ttk.Label(f, text="숫자키 0~9 = 그 번호 카메라, Tab = 다음 카메라. 처음 보는 카메라는 스펙을 먼저 잼 (30초~1분)",
                  foreground="gray", wraplength=WRAP).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ---- 렌즈 왜곡 보정 ----
        f = ttk.LabelFrame(body, text="렌즈 왜곡 분석·보정 (체커보드)", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.board_label = ttk.Label(f)
        self.board_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.btn_analyze = ttk.Button(f, command=vm.toggle_analyze)
        self.btn_auto_board = ttk.Button(f, command=vm.toggle_auto_board)
        self.btn_auto_correct = ttk.Button(f, command=vm.toggle_auto_correct)
        self.btn_undistort = ttk.Button(f, command=vm.toggle_undistort)
        buttons = [self.btn_analyze, self.btn_auto_board, self.btn_auto_correct, self.btn_undistort,
                   ttk.Button(f, text="초기화 (R)", command=vm.reset_analysis),
                   ttk.Button(f, text="캡처로 보정 (K)", command=vm.calibrate_from_captures),
                   ttk.Button(f, text="보정값 저장", command=vm.save_calibration),
                   ttk.Button(f, text="체커보드 이미지 저장", command=vm.save_checkerboard)]
        for i, btn in enumerate(buttons):
            btn.grid(row=1 + i // 2, column=i % 2, sticky="we", padx=2, pady=2)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="보정 화면(U)이 켜져 있으면 인식도 왜곡을 편 영상으로 합니다.",
                  foreground="gray", wraplength=WRAP).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ---- 가이드선 ----
        f = ttk.LabelFrame(body, text="가이드선", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.btn_guide = ttk.Button(f, command=vm.toggle_guide)
        self.btn_guide.grid(row=0, column=0, sticky="w", padx=2)
        ttk.Label(f, text="종류 (V)").grid(row=0, column=1, sticky="e", padx=(8, 2))
        self.guide_combo = ttk.Combobox(f, state="readonly", width=22, values=[GUIDE_MODE_NAMES[m] for m in GUIDE_MODES])
        self.guide_combo.grid(row=0, column=2, sticky="w")
        self.guide_combo.bind("<<ComboboxSelected>>",
                              lambda e: vm.set_guide_mode(GUIDE_MODES[self.guide_combo.current()]))
        self.guide_help = ttk.Label(f, text="", foreground="gray", wraplength=WRAP)
        self.guide_help.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ---- 초점 ----
        f = ttk.LabelFrame(body, text="초점 (수동 렌즈)", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.btn_focus = ttk.Button(f, command=vm.toggle_focus_assist)
        self.btn_peaking = ttk.Button(f, command=vm.toggle_peaking)
        self.btn_peak_level = ttk.Button(f, command=vm.next_peaking_level)
        buttons = [self.btn_focus, self.btn_peaking, self.btn_peak_level,
                   ttk.Button(f, text="최고값 초기화", command=lambda: vm.reset_focus_peak())]
        for i, btn in enumerate(buttons):
            btn.grid(row=i // 2, column=i % 2, sticky="we", padx=2, pady=2)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        self.focus_label = ttk.Label(f, text="", font=("Malgun Gothic", 10, "bold"))
        self.focus_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ---- 단축키 ----
        f = ttk.LabelFrame(body, text="단축키", padding=6)
        f.pack(fill="x", pady=(0, 6))
        ttk.Label(f, foreground="gray", justify="left", text=(
            "Space/s 캡처   0~9/Tab 카메라 전환   q/ESC 종료\n"
            "d 분석   a 보드 자동 인식   c 자동 보정   u 보정 화면\n"
            "r 분석 초기화   k 캡처로 보정   g 가이드선   v 가이드 종류\n"
            "f 초점 도우미   p 피킹")).pack(anchor="w")

    def _bind_vm(self):
        vm = self.vm
        on = lambda v: "ON" if v else "OFF"
        vm.cameras.subscribe(lambda cams: self.combo.configure(values=[l for _, l in cams]))
        vm.selected_camera.subscribe(self._show_selected_camera)
        vm.resolution_options.subscribe(lambda opts: (self.res_combo.configure(values=[l for _, l in opts]),
                                                      self._show_resolution(vm.resolution.value)))
        vm.resolution.subscribe(self._show_resolution)
        vm.board.subscribe(lambda _: self._show_board())
        vm.auto_board_on.subscribe(lambda v: (self._show_board(),
                                              self.btn_auto_board.configure(text=f"자동 인식 {on(v)} (A)")))
        vm.analyze_on.subscribe(lambda v: self.btn_analyze.configure(text=f"분석 {on(v)} (D)"))
        vm.auto_correct_on.subscribe(lambda v: self.btn_auto_correct.configure(text=f"자동 보정 {on(v)} (C)"))
        vm.undistort_on.subscribe(lambda v: self.btn_undistort.configure(text=f"보정 화면 {on(v)} (U)"))
        vm.guide_on.subscribe(lambda v: self.btn_guide.configure(text=f"가이드선 {on(v)} (G)"))
        vm.guide_mode.subscribe(lambda m: (self.guide_combo.current(GUIDE_MODES.index(m)),
                                           self.guide_help.configure(text=GUIDE_MODE_HELP[m])))
        vm.focus_on.subscribe(lambda v: self.btn_focus.configure(text=f"초점 도우미 {on(v)} (F)"))
        vm.peaking_on.subscribe(lambda v: self.btn_peaking.configure(text=f"피킹 {on(v)} (P)"))
        vm.peaking_level.subscribe(lambda v: self.btn_peak_level.configure(text=f"피킹 감도: {v}"))
        vm.focus_text.subscribe(lambda t: self.focus_label.configure(text=t))

    def _show_selected_camera(self, index):
        indexes = [i for i, _ in self.vm.cameras.value]
        if index in indexes:
            self.combo.current(indexes.index(index))
        else:
            self.combo.set("")

    def _show_resolution(self, key):
        keys = [k for k, _ in self.vm.resolution_options.value]
        if key in keys:
            self.res_combo.current(keys.index(key))
        else:
            self.res_combo.set("")

    def _show_board(self):
        b, auto = self.vm.board.value, self.vm.auto_board_on.value
        if b is None:
            text = "체커보드: 자동 인식 대기" if auto else "체커보드: 없음"
        else:
            text = f"체커보드: {b[0]}x{b[1]} ({'자동' if auto else '고정'})"
        self.board_label.configure(text=text)

    def _on_camera_select(self, _event=None):
        pos = self.combo.current()
        if pos >= 0:
            self.vm.select_camera(self.vm.cameras.value[pos][0])

    def _on_resolution_select(self, _event=None):
        pos = self.res_combo.current()
        if pos >= 0:
            self.vm.set_resolution(self.vm.resolution_options.value[pos][0])
