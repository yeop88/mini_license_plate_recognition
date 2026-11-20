# -*- coding: utf-8 -*-
"""
설정 창 틀 (별도 창, tk.Toplevel): 세로 스크롤 (내용이 화면보다 길어도 볼 수 있게)
메인 창의 [카메라 설정] / [인식 설정] 버튼으로 열고 닫는다. 창의 X 를 눌러도 닫힘(숨김)이고 설정은 그대로 남는다.
"""
import tkinter as tk
from tkinter import ttk

PANEL_WIDTH = 470


class SidePanel(tk.Toplevel):
    def __init__(self, master, title, on_close):
        super().__init__(master)
        self.title(title)
        self.withdraw()                                     # 처음엔 숨김 (버튼을 누르면 뜸)
        self.protocol("WM_DELETE_WINDOW", on_close)
        frame = ttk.Frame(self, padding=6)
        frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(frame, width=PANEL_WIDTH, highlightthickness=0, borderwidth=0)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.canvas)                  # 내용은 여기에 넣는다
        win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        # 마우스 휠: 창 위에 마우스가 있을 때만
        self.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", self._on_wheel))
        self.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

    def _on_wheel(self, e):
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(int(-e.delta / 120), "units")

    def show_beside(self, main, offset=0):
        """메인 창 오른쪽에 띄움 (화면 밖으로 나가면 화면 오른쪽 끝에 맞춤)"""
        main.update_idletasks()
        sw, sh = main.winfo_screenwidth(), main.winfo_screenheight()
        w = PANEL_WIDTH + 40
        h = min(max(500, main.winfo_height()), sh - 80)
        x = min(main.winfo_rootx() + main.winfo_width() + 8 + offset, sw - w - 8)
        y = max(0, main.winfo_rooty() - 30 + offset)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.deiconify()
        self.lift()

    @property
    def shown(self):
        return self.state() != "withdrawn"
