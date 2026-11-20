# -*- coding: utf-8 -*-
"""
결과 이미지 그리기 (원본 해상도). 화면 표시와 결과 캡처가 같은 이미지를 씀

번호판 글자는 한글(예: 12가3456)이라 cv2.putText 로는 못 그림 → 글자는 PIL + 맑은 고딕으로 한 번에 그린다
    파랑 = 글자 고정, 흰색 = 읽는 중, 회색 = 가장자리에 걸림 ([잘린 번호판도 표시] 켰을 때만), 노랑 = 화면 ID
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COL_LOCKED = (255, 110, 0)       # BGR 파랑: 글자 고정
COL_MEASURE = (255, 255, 255)    # 흰색: 읽는 중
COL_ENTER = (150, 150, 150)      # 회색: 가장자리
COL_ID = (0, 215, 255)           # 노랑: 화면 ID
FONT_FILES = [r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\malgun.ttf"]
_fonts = {}


def font(size):
    size = max(10, int(size))
    if size not in _fonts:
        f = None
        for path in FONT_FILES:
            if os.path.exists(path):
                f = ImageFont.truetype(path, size)
                break
        _fonts[size] = f or ImageFont.load_default()
    return _fonts[size]


def draw_labels(bgr, labels):
    """labels: [(글자, (x, y) 왼쪽 위, 글자색 BGR, 띠색 BGR 또는 None, 글자 크기 px)] → 한 번에 그림"""
    if not labels:
        return bgr
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    rgb = lambda c: (c[2], c[1], c[0])
    for text, (x, y), col, fill, size in labels:
        f = font(size)
        x0, y0, x1, y1 = d.textbbox((x, y), text, font=f)
        if fill is not None:
            d.rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=rgb(fill))
        else:
            d.text((x, y), text, font=f, fill=(0, 0, 0), stroke_width=3, stroke_fill=(0, 0, 0))
        d.text((x, y), text, font=f, fill=rgb(col))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_tracks(frame, rows, show_entering, hud, hud_bottom_right=False):
    """rows: 추적 번호판 표시 정보 (pipeline 이 만듦)"""
    vis = frame.copy()
    h, w = vis.shape[:2]
    size = max(16, int(w / 1920 * 34))           # 해상도에 맞춘 글자 크기
    labels = []
    for r in rows:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        if r["state"] == "entering":
            if show_entering:
                cv2.rectangle(vis, (x1, y1), (x2, y2), COL_ENTER, 1)
            continue
        locked = r["state"] == "locked"
        cv2.rectangle(vis, (x1, y1), (x2, y2), COL_LOCKED if locked else COL_MEASURE, 3 if locked else 2)
        ty = y1 - int(size * 1.5) if y1 > size * 2 else y2 + 6
        if locked:
            labels.append((r["text"], (x1, ty), (255, 255, 255), COL_LOCKED, size))
        else:
            last = f" {r['last']}" if r.get("last") else ""
            labels.append((f"읽는 중 {r['n']}/{r['need']}{last}", (x1, ty), (0, 0, 0), COL_MEASURE, int(size * 0.8)))
        if r["no"] is not None:                    # 화면 ID: 박스 왼쪽 아래 바깥 (작은 번호판 글자를 가리지 않게)
            labels.append((f"ID {r['no']}", (x1, min(h - size, y2 + 4) if ty < y1 else max(0, y1 - size - 8)),
                           (0, 0, 0), COL_ID, int(size * 0.7)))
    hs = int(size * 0.9)
    if hud_bottom_right:
        tw = font(hs).getlength(hud)
        labels.append((hud, (max(0, int(w - tw - 20)), int(h * 0.92) - hs), (255, 255, 255), None, hs))
    else:
        labels.append((hud, (20, 20), (255, 255, 255), None, hs))
    return draw_labels(vis, labels)
