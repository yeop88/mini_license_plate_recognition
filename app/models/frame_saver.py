# -*- coding: utf-8 -*-
"""결과 프레임 캡처 저장 (별도 스레드, 디스크 쓰기가 인식을 늦추지 않게)

captures\\recognition\\<시각>_<cam|video>_<감지기>_<글자인식>\\
    frame_000123.jpg      결과 화면 (원본 해상도)
    raw\\frame_000123.png  원본 프레임 (저장 종류 '결과+원본' / '원본만')
    predictions.csv       프레임별 추적 번호판 (상태, 박스, 읽은 글자, 고정 글자)
"""
import collections
import csv
import os
import queue
import threading

import cv2

CAPTURE_KINDS = ("결과 화면", "결과+원본", "원본만")
CSV_HEADER = ["frame", "object_id_shown", "track_id", "state", "x1", "y1", "x2", "y2", "read_frames",
              "last_read", "plate_text", "vote_ratio", "mean_conf", "lock_reason", "lock_frame"]


def clean_name(s):
    """폴더 이름에 어색한 글자 정리"""
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in s).strip("-").replace("--", "-") or "none"


def capture_name(det, reader):
    return f"{clean_name(det or 'nodet')}_{clean_name(reader or 'noread')}"


def csv_row(fi, r):
    lk = r["state"] == "locked"
    return [fi, r["no"] if r["no"] is not None else "", r["id"], r["state"], *[int(v) for v in r["box"]], r["n"],
            r.get("last") or "", r["text"] if lk else "", round(r["ratio"], 2) if lk else "",
            round(r["conf"], 3) if lk else "", r["reason"] if lk else "", r["lock_frame"] if lk else ""]


class FrameSaver(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=120)
        self.saved = 0
        self.dropped = 0
        self.counts = collections.Counter()      # 폴더별 저장 장수

    def put(self, item):
        """item = (folder, frame index, 저장 종류, 결과 이미지, 원본 이미지, rows)"""
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1                   # 디스크가 못 따라가면 버리고 개수만 셈

    def pending(self):
        return self.q.qsize()

    def run(self):
        while True:
            folder, fi, kind, vis, raw, rows = self.q.get()
            if kind != "원본만":
                cv2.imwrite(os.path.join(folder, f"frame_{fi:06d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if kind != "결과 화면":
                os.makedirs(os.path.join(folder, "raw"), exist_ok=True)
                cv2.imwrite(os.path.join(folder, "raw", f"frame_{fi:06d}.png"), raw)
            path = os.path.join(folder, "predictions.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(CSV_HEADER)
                if not rows:
                    wr.writerow([fi] + [""] * (len(CSV_HEADER) - 1))
                for r in rows:
                    wr.writerow(csv_row(fi, r))
            self.saved += 1
            self.counts[folder] += 1
