# -*- coding: utf-8 -*-
"""
tracker (Model) — 번호판 추적 + 화면 ID + 글자 값 고정 (학습 불필요)

    예측  : 추적마다 중심 이동 속도(vx, vy, px/프레임)를 이동 평균으로 재고, 다음 위치 = 지금 박스 + 속도 (등속)
    짝짓기: 예상 박스와 새 감지 박스의 IoU 를 큰 순서로 1:1 (≥ MATCH_IOU), 남은 감지는 새 추적
    중복  : 큰 감지 박스 안에 CONTAIN_RATIO 이상 들어간 작은 감지 박스는 같은 번호판의 중복으로 보고 뺌
    상태  : entering (화면 가장자리에 걸려 일부만 보임, 읽지 않음)
            → measuring (다 보임, 화면 ID 를 붙이고 글자를 읽어 모음)
            → locked (읽은 글자가 lock_frames 개 모이면 글자 자리별 다수결로 고정, 사라질 때까지 그대로)
    끝    : 감지가 max_missed 프레임 넘게 끊기거나 박스가 화면 밖으로 나가면 추적 종료
"""
import collections

import numpy as np

ENTRY_MARGIN = 2        # 박스가 화면 가장자리에서 이만큼(px) 떨어져야 "다 보임"
LOCK_FRAMES = 5         # 글자를 이만큼 읽으면 다수결로 고정
MAX_MISSED = 10         # 감지가 이 프레임 수 넘게 끊기면 추적 종료
MATCH_IOU = 0.3
CONTAIN_RATIO = 0.8


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def text_shape(text):
    """글자 모양: 숫자 → 0, 한글 → 가, 그 밖은 그대로.  '12가3456' → '00가0000'"""
    return "".join("0" if ch.isdigit() else "가" if "가" <= ch <= "힣" else ch for ch in text)


def vote_text(preds):
    """읽은 글자들 [(글자, 신뢰도)] → (고정 글자, 일치율, 평균 신뢰도)

    1) 모양(text_shape)이 같은 읽기끼리 묶고, 가장 많은 묶음을 고름 (같으면 신뢰도 합이 큰 쪽)
       → 길이나 한글 위치가 다른 읽기는 섞지 않음 (예: 12가3456 과 123가4567)
    2) 그 묶음 안에서 글자 자리마다 가장 많이 나온 글자 (같으면 신뢰도 합이 큰 쪽)
       → 매번 다른 자리 한 글자씩 틀려도 맞는 번호를 만들 수 있음
          예: 12가3456, 12가3458, 17가3456 → 12가3456 (통째 다수결이면 셋이 모두 1표)
    일치율 = 모든 읽기에 대해 자리별로 고정 글자와 같은 비율의 평균 (모양이 다른 읽기는 0)
    """
    if not preds:
        return None, None, None
    groups = collections.defaultdict(list)
    for t, c in preds:
        groups[text_shape(t)].append((t, c))
    group = max(groups.values(), key=lambda g: (len(g), sum(c for _, c in g)))
    chars = []
    for i in range(len(group[0][0])):
        count = collections.Counter(t[i] for t, _ in group)
        score = collections.defaultdict(float)
        for t, c in group:
            score[t[i]] += c
        chars.append(max(count, key=lambda ch: (count[ch], score[ch])))
    text = "".join(chars)
    n = max(len(text), 1)
    ratio = sum(sum(a == b for a, b in zip(t, text)) / n for t, _ in group) / len(preds)
    conf = sum(c for _, c in group) / len(group)
    return text, ratio, conf


class Track:
    _next = 1

    def __init__(self, box, fi):
        self.id = Track._next          # 내부 번호 (조각마다 새로 생김)
        Track._next += 1
        self.no = None                 # 화면 ID (다 보여 읽기 시작할 때 붙음)
        self.box = np.asarray(box, float)
        self.v = np.zeros(2)           # 중심 이동 속도 (px/프레임)
        self.state = "entering"
        self.preds = []                # 읽은 글자 [(글자, 신뢰도)] (빈 글자는 넣지 않음)
        self.text, self.ratio, self.conf = None, None, None
        self.lock_reason = None
        self.missed = 0
        self.first_frame = self.last_frame = fi
        self.lock_frame = None

    def lock(self, fi, reason="다수결"):
        """모은 글자로 고정 (vote_text: 글자 자리마다 다수결)"""
        self.text, self.ratio, self.conf = vote_text(self.preds)
        self.state = "locked"
        self.lock_frame = fi
        self.lock_reason = reason


class Tracker:
    def __init__(self, width, height, entry_margin=ENTRY_MARGIN, lock_frames=LOCK_FRAMES, max_missed=MAX_MISSED,
                 next_no=1):
        self.W, self.H = width, height
        self.next_no = next_no
        self.entry_margin, self.lock_frames, self.max_missed = entry_margin, lock_frames, max_missed
        self.tracks = []

    def touches_edge(self, b):
        m = self.entry_margin
        return b[0] < m or b[1] < m or b[2] > self.W - m or b[3] > self.H - m

    def outside(self, b):
        return b[2] <= 0 or b[3] <= 0 or b[0] >= self.W or b[1] >= self.H

    @staticmethod
    def contained_indices(xyxy, ratio=CONTAIN_RATIO):
        """다른(더 큰) 감지 박스 안에 ratio 이상 들어 있는 작은 감지 박스 번호들 (같은 번호판의 중복 감지)"""
        b = np.asarray(xyxy, float).reshape(-1, 4)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        drop = set()
        for i in range(len(b)):
            for j in range(len(b)):
                if i == j or area[i] > area[j] or (area[i] == area[j] and i < j):
                    continue
                ix = max(0.0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
                iy = max(0.0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
                if ix * iy >= ratio * max(area[i], 1e-6):
                    drop.add(i)
        return drop

    def update(self, xyxy, fi):
        """감지 박스 → [(track, 감지 index)] (이번 프레임에 감지된 추적만). 상태 전환도 여기서"""
        xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        keep_idx = [i for i in range(len(xyxy)) if i not in self.contained_indices(xyxy)]
        dets = xyxy[keep_idx]
        pred = [t.box + np.array([t.v[0], t.v[1], t.v[0], t.v[1]]) for t in self.tracks]
        pairs = sorted(((iou(p, d), ti, di) for ti, p in enumerate(pred) for di, d in enumerate(dets)), reverse=True)
        used_t, used_d, out = set(), set(), []
        for v, ti, di in pairs:
            if v < MATCH_IOU or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, nb = self.tracks[ti], dets[di]
            d = np.array([(nb[0] + nb[2] - t.box[0] - t.box[2]) / 2, (nb[1] + nb[3] - t.box[1] - t.box[3]) / 2])
            t.v = d if not t.v.any() else 0.6 * t.v + 0.4 * d
            t.box, t.missed, t.last_frame = nb, 0, fi
            out.append((t, di))
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(d, fi)
                self.tracks.append(t)
                out.append((t, di))
        for ti, t in enumerate(self.tracks):
            if ti not in used_t and t.last_frame != fi:
                t.missed += 1
                t.box = t.box + np.array([t.v[0], t.v[1], t.v[0], t.v[1]])
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed and not self.outside(t.box)]
        for t, _ in out:
            if t.state == "entering" and not self.touches_edge(t.box):
                t.state = "measuring"
                t.no = self.next_no
                self.next_no += 1
        return [(t, keep_idx[di]) for t, di in out]

    def add_prediction(self, t, text, conf, fi):
        """읽은 글자 추가 → 고정되면 True. 빈 글자(못 읽음)는 세지 않음"""
        if t.state != "measuring" or not text:
            return False
        t.preds.append((str(text), float(conf)))
        if len(t.preds) >= self.lock_frames:
            t.lock(fi)
            return True
        return False
