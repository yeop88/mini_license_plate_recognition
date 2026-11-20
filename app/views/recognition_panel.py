# -*- coding: utf-8 -*-
"""
인식 설정 창 (별도 창) — RecognitionViewModel 의 상태를 구독해 표시하고, 입력은 ViewModel 명령으로 넘긴다
    입력(카메라/영상) / 모델(번호판 감지기·글자 인식) / 시작·멈춤 / 추적 옵션 / 결과 캡처 / 상태 / 현재 번호판 / 최근 고정
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk

from app import paths
from app.viewmodels.recognition_viewmodel import SOURCES
from app.views.side_panel import SidePanel

WRAP = 420
STATE_KR = {"entering": "가장자리", "measuring": "읽는 중", "locked": "고정"}
NO_DETECTOR = "(등록된 감지기 없음)"


class RecognitionPanel(SidePanel):
    def __init__(self, parent, vm, on_close):
        super().__init__(parent, "인식 설정", on_close)
        self.vm = vm
        self._recent_count = 0
        self._build(self.body)
        self._bind_vm()

    # ---------------- 화면 구성 ----------------
    def _build(self, body):
        vm = self.vm
        # ---- 입력 ----
        f = ttk.LabelFrame(body, text="입력", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.src_var = tk.StringVar(value=vm.source.value)
        for c, (k, name) in enumerate(SOURCES.items()):
            ttk.Radiobutton(f, text=name, value=k, variable=self.src_var,
                            command=lambda: vm.set_source(self.src_var.get())).grid(row=0, column=c, sticky="w", padx=(0, 12))
        self.video_var = tk.StringVar()
        self.video_cb = ttk.Combobox(f, textvariable=self.video_var, state="readonly", width=40)
        self.video_cb.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))
        self.video_cb.bind("<<ComboboxSelected>>", lambda e: vm.set_source("video", self.video_var.get()))
        ttk.Button(f, text="열기…", width=6, command=self._open_video).grid(row=1, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(f, text="카메라: [카메라 설정] 에서 연 카메라 (보정 화면이 켜져 있으면 왜곡을 편 영상으로 인식)\n"
                          "영상 파일: videos\\ 의 영상 또는 [열기…] 로 고른 파일",
                  foreground="gray", wraplength=WRAP, justify="left").grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ---- 모델 ----
        f = ttk.LabelFrame(body, text="모델 (고르는 순간 불러옴)", padding=6)
        f.pack(fill="x", pady=(0, 6))
        ttk.Label(f, text="번호판 감지").grid(row=0, column=0, sticky="w")
        self.det_cb = ttk.Combobox(f, state="readonly", width=36,
                                   values=[l for _, l in vm.detector_choices] or [NO_DETECTOR])
        self.det_cb.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(f, text="글자 인식").grid(row=1, column=0, sticky="w")
        self.read_cb = ttk.Combobox(f, state="readonly", width=36, values=[l for _, l in vm.reader_choices])
        self.read_cb.grid(row=1, column=1, sticky="w", padx=4, pady=2)
        note = ("" if vm.has_detector else "아직 등록된 번호판 모델이 없습니다. ") + \
            "모델은 app\\models\\plate_models.py 에 등록합니다 (가중치는 models\\)."
        ttk.Label(f, text=note, foreground="gray" if vm.has_detector else "#a06000",
                  wraplength=WRAP).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ---- 실행 ----
        f = ttk.LabelFrame(body, text="실행", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.play_btn = ttk.Button(f, command=vm.toggle_play, width=18)
        self.play_btn.pack(side="left", padx=2)
        self.restart_btn = ttk.Button(f, text="⏮ 처음부터", command=vm.restart)
        self.restart_btn.pack(side="left", padx=2)

        # ---- 옵션 ----
        f = ttk.LabelFrame(body, text="표시·추적 옵션", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.opt_vars = {}

        def check(name, text, r):
            v = tk.BooleanVar(value=vm.options[name])
            self.opt_vars[name] = v
            ttk.Checkbutton(f, text=text, variable=v,
                            command=lambda: vm.set_option(name, v.get())).grid(row=r, column=0, columnspan=4, sticky="w")
        check("realtime", "영상: 원래 속도에 맞춤 (끄면 처리되는 대로 빠르게)", 0)
        check("show_entering", "가장자리에 걸린 번호판도 표시 (회색, 읽지 않음)", 1)

        def spin(name, label, r, c, lo, hi):
            ttk.Label(f, text=label).grid(row=r, column=c, sticky="e", padx=(0, 2))
            v = tk.IntVar(value=vm.options[name])
            self.opt_vars[name] = v
            ttk.Spinbox(f, from_=lo, to=hi, increment=1, width=5, textvariable=v).grid(row=r, column=c + 1, sticky="w")

            def push(*_):
                try:
                    vm.set_option(name, int(v.get()))
                except (tk.TclError, ValueError):
                    pass                              # 입력 중 (빈 칸 등)
            v.trace_add("write", push)
        spin("entry_margin", "가장자리 여백 px", 2, 0, 0, 300)
        spin("lock_frames", "글자 고정까지 읽기", 2, 2, 1, 30)

        # ---- 화질·추적 유지 ----
        f = ttk.LabelFrame(body, text="화질·추적 유지", padding=6)
        f.pack(fill="x", pady=(0, 6))
        for r, (name, text) in enumerate((
                ("sharp_select", "흐린 프레임 건너뛰기 + 선명도 가중 다수결 (약 0.2 ms)"),
                ("enhance_contrast", "글자 인식 전 대비 보정 (CLAHE, 어둡거나 역광일 때)"),
                ("enhance_sharpen", "글자 인식 전 선명화 (약하게, 초점 흐림에 도움)"),
                ("hold_track", "고정 후 추적 유지: 감지가 끊겨도 기다림 + 번호판 영상으로 따라가기"),
                ("low_conf", "추적 중인 번호판은 낮은 신뢰도 감지로도 이어 가기"),
                ("reid", "끊긴 뒤 같은 번호가 다시 읽히면 이전 ID·고정값 되살리기"))):
            v = tk.BooleanVar(value=vm.options[name])
            self.opt_vars[name] = v
            ttk.Checkbutton(f, text=text, variable=v,
                            command=lambda n=name, v=v: vm.set_option(n, v.get())).grid(row=r, column=0, sticky="w")

        # ---- 결과 캡처 ----
        f = ttk.LabelFrame(body, text="결과 프레임 캡처", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.cap_var = tk.BooleanVar(value=vm.capture_on.value)
        ttk.Checkbutton(f, text="캡처 (시작하면 프레임마다 저장)", variable=self.cap_var,
                        command=lambda: vm.set_capture(self.cap_var.get())).grid(row=0, column=0, sticky="w")
        self.kind_var = tk.StringVar(value=vm.capture_kind.value)
        kind = ttk.Combobox(f, textvariable=self.kind_var, state="readonly", width=9, values=vm.CAPTURE_KINDS)
        kind.grid(row=0, column=1, padx=4)
        kind.bind("<<ComboboxSelected>>", lambda e: vm.set_capture_kind(self.kind_var.get()))
        ttk.Button(f, text="폴더 열기", command=vm.open_capture_dir).grid(row=0, column=2)
        self.cap_label = ttk.Label(f, text="", wraplength=WRAP, justify="left")
        self.cap_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ---- 상태 ----
        f = ttk.LabelFrame(body, text="상태", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.info_lbl = ttk.Label(f, text="(인식 결과가 오면 표시)", font=("Consolas", 10), justify="left")
        self.info_lbl.pack(anchor="w")

        # ---- 현재 화면 번호판 ----
        f = ttk.LabelFrame(body, text="현재 화면 번호판 (위에서부터)", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.tree = ttk.Treeview(f, columns=("id", "text", "state", "conf"), show="headings", height=6)
        for c, t, w in (("id", "ID", 55), ("text", "번호", 150), ("state", "상태", 110), ("conf", "신뢰도", 80)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x")

        # ---- 최근 고정 ----
        f = ttk.LabelFrame(body, text="최근 고정된 번호판", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.recent = tk.Listbox(f, height=6, font=("Malgun Gothic", 10))
        self.recent.pack(fill="x")
        ttk.Label(f, foreground="gray", justify="left", wraplength=WRAP, text=(
            "노랑 ID n = 추적 번호 (다 보인 순서, 사라질 때까지 같음)\n"
            "흰색 = 읽는 중 (n/N, 마지막으로 읽은 글자), 파랑 = 글자 고정 (N번 읽은 것 중 다수결)\n"
            "화면 가장자리에 걸려 일부만 보이는 번호판은 다 보일 때까지 읽지 않음")).pack(anchor="w", pady=(4, 0))

    # ---------------- ViewModel 연결 ----------------
    def _bind_vm(self):
        vm = self.vm
        self.det_cb.bind("<<ComboboxSelected>>", self._on_det)
        self.read_cb.bind("<<ComboboxSelected>>", self._on_read)
        vm.det_key.subscribe(lambda k: self._show_choice(self.det_cb, vm.detector_choices, k, NO_DETECTOR))
        vm.read_key.subscribe(lambda k: self._show_choice(self.read_cb, vm.reader_choices, k, ""))
        vm.source.subscribe(self._on_source)
        vm.videos.subscribe(lambda vs: self.video_cb.configure(values=vs))
        vm.video_path.subscribe(lambda p: self.video_var.set(os.path.basename(p) if p else ""))
        vm.playing.subscribe(lambda _: self._show_play())
        vm.info.subscribe(self._on_info, call_now=False)
        vm.rows.subscribe(self._on_rows, call_now=False)
        vm.capture_status.subscribe(lambda st: self.cap_label.cget("text") != st[0]
                                    and self.cap_label.config(text=st[0], foreground=st[1]))
        vm.capture_on.subscribe(lambda on: self.cap_var.set(on))

    @staticmethod
    def _show_choice(cb, choices, key, empty):
        keys = [k for k, _ in choices]
        if key in keys:
            cb.current(keys.index(key))
        else:
            cb.set(empty)

    def _on_det(self, _e=None):
        pos = self.det_cb.current()
        if 0 <= pos < len(self.vm.detector_choices):
            self.vm.select_models(det=self.vm.detector_choices[pos][0])

    def _on_read(self, _e=None):
        pos = self.read_cb.current()
        if pos >= 0:
            self.vm.select_models(reader=self.vm.reader_choices[pos][0])

    def _open_video(self):
        start = paths.VIDEOS_DIR if os.path.isdir(paths.VIDEOS_DIR) else paths.ROOT
        path = filedialog.askopenfilename(title="영상 파일", initialdir=start,
                                          filetypes=[("영상", "*.mp4 *.avi *.mkv *.mov"), ("모든 파일", "*.*")])
        if path:
            self.vm.set_source("video", os.path.normpath(path))

    def _on_source(self, kind):
        self.src_var.set(kind)
        video = kind == "video"
        self.video_cb.configure(state="readonly" if video else "disabled")
        self.restart_btn.configure(state="normal" if video else "disabled")
        self.tree.delete(*self.tree.get_children())
        self._show_play()

    def _show_play(self):
        on, video = self.vm.playing.value, self.vm.source.value == "video"
        if video:
            self.play_btn.configure(text="⏸ 일시정지 (I)" if on else "▶ 재생 (I)")
        else:
            self.play_btn.configure(text="⏸ 인식 멈춤 (I)" if on else "▶ 인식 시작 (I)")

    def _on_info(self, d):
        s = d["stats"]
        tot = d["det"] + d["read"]
        lines = [
            f"{'영상 프레임' if d['mode'] == 'video' else '인식 프레임'}  {d['fi']:5d}"
            + (f" / {self.vm.total.value}" if d["mode"] == "video" else ""),
            f"화면 fps     {d['screen_fps']:5.1f}",
            f"감지         {d['det']:5.1f} ms",
            f"글자 인식    {d['read']:5.1f} ms (읽는 중인 번호판만)",
            f"합계         {tot:5.1f} ms ({1000 / max(tot, 1e-3):.0f} fps 가능)",
            f"고정 번호판  {s['locked']} 개 (감지 {s['detections']}회 / {s['frames']} 프레임)",
        ]
        self.info_lbl.config(text="\n".join(lines))
        plates = s["plates"]
        if s["locked"] != self._recent_count:      # 새로 고정되거나 처음부터 다시 시작했을 때만 목록 갱신
            self.recent.delete(0, "end")
            for no, text in reversed(plates):
                self.recent.insert("end", f"ID {no}   {text}")
            self._recent_count = s["locked"]

    def _on_rows(self, rows):
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            if r["state"] == "entering" and not self.vm.options["show_entering"]:
                continue
            lk = r["state"] == "locked"
            text = r["text"] if lk else (r["last"] or "-")
            state = f"고정 ({r['ratio'] * 100:.0f}%)" if lk else (
                f"읽는 중 {r['n']}/{r['need']}" if r["state"] == "measuring" else STATE_KR[r["state"]])
            conf = f"{r['conf']:.2f}" if lk else f"감지 {r['det_conf']:.2f}"
            self.tree.insert("", "end", values=(f"ID {r['no']}" if r["no"] is not None else "-", text, state, conf))
