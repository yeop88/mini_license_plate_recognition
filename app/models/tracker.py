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

고정 후 추적 유지 (hold, reid — 설정 창에서 끄고 켬)
    낮은 감지    : 이미 추적 중인 번호판은 신뢰도가 낮은 감지로도 이어 감 (update 의 conf/high, 새 추적은 만들지 않음)
    오래 기다림  : 고정된 번호판은 감지가 HOLD_MISSED 프레임까지 끊겨도 예상 위치로 유지
    영상 따라가기: 감지가 끊긴 프레임에서는 직전 번호판 영상을 예상 위치 둘레에서 찾아 박스를 이어 감 (템플릿 매칭)
    번호로 다시 찾기: 추적이 끊겼다가 REID_FRAMES 안에 같은 번호가 읽히면 이전 화면 ID 와 고정값을 그대로 되살림
"""
import collections

import cv2
import numpy as np

ENTRY_MARGIN = 2        # 박스가 화면 가장자리에서 이만큼(px) 떨어져야 "다 보임"
LOCK_FRAMES = 5         # 글자를 이만큼 읽으면 다수결로 고정
MAX_MISSED = 10         # 감지가 이 프레임 수 넘게 끊기면 추적 종료
MATCH_IOU = 0.3
CONTAIN_RATIO = 0.8
HOLD_MISSED = 30        # 고정된 번호판은 감지가 이 프레임 수까지 끊겨도 유지 (60 fps 에서 0.5 초)
BRIDGE_MAX = 90         # 감지 없이 영상으로만 따라가는 최대 연속 프레임 (그 뒤는 감지가 다시 와야 함)
BRIDGE_SCORE = 0.6      # 영상 따라가기에서 이 이상 비슷해야 같은 번호판으로 봄 (정규화 상관)
BRIDGE_SHIFT = 0.25     # 영상 따라가기로 찾은 위치가 예상 위치에서 번호판 폭 x 이 값보다 멀면 버림
REID_FRAMES = 300       # 끝난 고정 추적을 번호로 다시 찾을 수 있는 시간 (프레임, 60 fps 에서 5 초)
NEAR_RATIO = 0.6        # IoU 로 못 짝지은 감지도 예상 중심과의 거리가 박스 폭 x 이 값 안이면 같은 번호판 (가림에서 나올 때 일부만 보이는 경우)
ID_WAIT = 10            # 되살릴 후보가 있으면 화면 ID 를 바로 붙이지 않고 첫 글자를 읽을 때까지 기다리는 최대 프레임


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

    preds 에 선명도가 있으면 [(글자, 신뢰도, 선명도)] 표 수 대신 가중치(선명도 / 가장 큰 선명도)를 더함
    → 흐린 프레임에서 읽은 글자는 덜 반영
    """
    if not preds:
        return None, None, None
    sharp = [p[2] if len(p) > 2 and p[2] is not None else None for p in preds]
    top = max((s for s in sharp if s is not None), default=0.0)
    items = [(p[0], float(p[1]), (s / top if s is not None and top > 0 else 1.0)) for p, s in zip(preds, sharp)]
    groups = collections.defaultdict(list)
    for it in items:
        groups[text_shape(it[0])].append(it)
    group = max(groups.values(), key=lambda g: (sum(w for *_, w in g), sum(c for _, c, _ in g)))
    chars = []
    for i in range(len(group[0][0])):
        vote = collections.defaultdict(float)
        score = collections.defaultdict(float)
        for t, c, w in group:
            vote[t[i]] += w
            score[t[i]] += c
        chars.append(max(vote, key=lambda ch: (round(vote[ch], 6), score[ch])))
    text = "".join(chars)
    n = max(len(text), 1)
    ratio = sum(sum(a == b for a, b in zip(t, text)) / n for t, _, _ in group) / len(items)
    conf = sum(c for _, c, _ in group) / len(group)
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
        self.preds = []                # 읽은 글자 [(글자, 신뢰도, 선명도)] (빈 글자는 넣지 않음)
        self.text, self.ratio, self.conf = None, None, None
        self.lock_reason = None
        self.missed = 0
        self.first_frame = self.last_frame = fi
        self.lock_frame = None
        self.best_sharp = 0.0          # 지금까지 가장 선명했던 값 (흐린 프레임 건너뛰기 기준)
        self.tmpl = None               # 마지막으로 감지된 번호판 회색 영상 (감지가 끊길 때 영상으로 따라가기)
        self.bridged = 0               # 감지 없이 영상으로 따라간 연속 프레임 수
        self.seen = False              # 이번 프레임에 감지 또는 영상으로 위치를 찾았는지
        self.measure_frame = None      # 다 보이기 시작한 프레임

    def lock(self, fi, reason="다수결"):
        """모은 글자로 고정 (vote_text: 글자 자리마다 다수결, 선명도가 있으면 가중)"""
        self.text, self.ratio, self.conf = vote_text(self.preds)
        self.state = "locked"
        self.lock_frame = fi
        self.lock_reason = reason

    def move_to(self, nb):
        d = np.array([(nb[0] + nb[2] - self.box[0] - self.box[2]) / 2, (nb[1] + nb[3] - self.box[1] - self.box[3]) / 2])
        self.v = d if not self.v.any() else 0.6 * self.v + 0.4 * d
        self.box = np.asarray(nb, float)


class Tracker:
    def __init__(self, width, height, entry_margin=ENTRY_MARGIN, lock_frames=LOCK_FRAMES, max_missed=MAX_MISSED,
                 next_no=1):
        self.W, self.H = width, height
        self.next_no = next_no
        self.entry_margin, self.lock_frames, self.max_missed = entry_margin, lock_frames, max_missed
        # 고정 후 추적 유지 옵션 (pipeline 이 설정 창 값으로 바꿈)
        self.hold = True               # 고정된 번호판은 감지가 끊겨도 HOLD_MISSED 프레임까지 유지 + 영상으로 따라가기
        self.reid = True               # 끊긴 뒤 같은 번호가 다시 읽히면 이전 화면 ID·고정값을 되살림
        self.tracks = []
        self.lost = []                 # 끝난 고정 추적 [{no, text, ratio, conf, until}] (번호로 다시 찾기)

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

    def _max_missed(self, t):
        return max(self.max_missed, HOLD_MISSED) if (self.hold and t.state == "locked") else self.max_missed

    @staticmethod
    def _match_near(tracks, dets, used_t, used_d):
        """IoU 로 못 짝지은 것끼리 중심 거리로 짝지음 (가까운 순서, 크기가 너무 다르면 제외)"""
        cand = []
        for ti, t in enumerate(tracks):
            if id(t) in used_t:
                continue
            p = t.box + np.array([t.v[0], t.v[1], t.v[0], t.v[1]])
            pw, ph = p[2] - p[0], p[3] - p[1]
            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                dw, dh = d[2] - d[0], d[3] - d[1]
                if not (0.4 < dh / max(ph, 1e-6) < 2.5):
                    continue
                dist = np.hypot((d[0] + d[2] - p[0] - p[2]) / 2, (d[1] + d[3] - p[1] - p[3]) / 2)
                if dist < NEAR_RATIO * max(pw, dw):
                    cand.append((dist, ti, di))
        out = []
        for _, ti, di in sorted(cand):
            t = tracks[ti]
            if id(t) in used_t or di in used_d:
                continue
            used_t.add(id(t))
            used_d.add(di)
            out.append((t, di))
        return out

    @staticmethod
    def _match(tracks, dets, used_t, used_d):
        """예상 박스와 감지 박스를 IoU 큰 순서로 1:1 → [(track, dets 의 index)]"""
        pred = [t.box + np.array([t.v[0], t.v[1], t.v[0], t.v[1]]) for t in tracks]
        pairs = sorted(((iou(p, d), ti, di) for ti, p in enumerate(pred) for di, d in enumerate(dets)),
                       key=lambda p: p[0], reverse=True)
        out = []
        for v, ti, di in pairs:
            t = tracks[ti]
            if v < MATCH_IOU or id(t) in used_t or di in used_d:
                continue
            used_t.add(id(t))
            used_d.add(di)
            out.append((t, di))
        return out

    def update(self, xyxy, fi, conf=None, high=None):
        """감지 박스 → [(track, 감지 index)] (이번 프레임에 감지된 추적만). 상태 전환도 여기서

        conf, high 를 주면 두 단계로 짝지음 (ByteTrack 방식):
            1) 신뢰도 ≥ high 인 감지로 짝짓기, 남은 것은 새 추적
            2) 짝을 못 찾은 기존 추적만 신뢰도 < high 인 감지와 짝지음 (낮은 감지로는 새 추적을 만들지 않음)
        """
        xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        if conf is None or high is None:
            hi_idx, lo_idx = list(range(len(xyxy))), []
        else:
            conf = np.asarray(conf, float).reshape(-1)
            hi_idx = [i for i in range(len(xyxy)) if conf[i] >= high]
            lo_idx = [i for i in range(len(xyxy)) if conf[i] < high]
        drop = self.contained_indices(xyxy[hi_idx]) if hi_idx else set()
        hi_idx = [i for k, i in enumerate(hi_idx) if k not in drop]
        # 낮은 감지는 높은 감지와 겹치지 않는 것만 (같은 번호판의 두 번째 박스는 버림)
        lo_idx = [i for i in lo_idx if all(iou(xyxy[i], xyxy[j]) < MATCH_IOU for j in hi_idx)]
        for t in self.tracks:
            t.seen = False
        used_t, used_d, out = set(), set(), []
        for t, k in self._match(self.tracks, xyxy[hi_idx], used_t, used_d):
            out.append((t, hi_idx[k]))
        if self.hold:                                       # 가림에서 나와 일부만 보이는 번호판도 같은 추적으로
            near = [t for t in self.tracks if t.state != "entering"]
            for t, k in self._match_near(near, xyxy[hi_idx], used_t, used_d):
                out.append((t, hi_idx[k]))
        new = [hi_idx[k] for k in range(len(hi_idx)) if k not in used_d]
        if lo_idx:
            rest = [t for t in self.tracks if id(t) not in used_t]
            for t, k in self._match(rest, xyxy[lo_idx], used_t, set()):
                out.append((t, lo_idx[k]))
        for t, di in out:
            t.move_to(xyxy[di])
            t.missed, t.last_frame, t.bridged, t.seen = 0, fi, 0, True
        for di in new:
            t = Track(xyxy[di], fi)
            t.seen = True
            self.tracks.append(t)
            out.append((t, di))
        for t in self.tracks:
            if not t.seen:
                t.missed += 1
                t.box = t.box + np.array([t.v[0], t.v[1], t.v[0], t.v[1]])
        self._drop_ended(fi)
        for t, _ in out:
            if t.state == "entering" and not self.touches_edge(t.box):
                t.state = "measuring"
                t.measure_frame = fi
                if not (self.reid and self._has_reid_candidates()):
                    self._give_no(t)
            elif t.state == "measuring" and t.no is None and fi - t.measure_frame >= ID_WAIT:
                self._give_no(t)                            # 기다려도 되살릴 번호가 안 읽힘 → 새 화면 ID
        return out

    def _give_no(self, t):
        t.no = self.next_no
        self.next_no += 1

    def _has_reid_candidates(self):
        return bool(self.lost) or any(o.state == "locked" and not o.seen for o in self.tracks)

    def _drop_ended(self, fi):
        keep = []
        for t in self.tracks:
            if t.missed <= self._max_missed(t) and not self.outside(t.box):
                keep.append(t)
            elif t.state == "locked" and t.no is not None:      # 끝난 고정 추적은 번호로 다시 찾을 수 있게 보관
                self.lost.append({"no": t.no, "text": t.text, "ratio": t.ratio, "conf": t.conf,
                                  "until": fi + REID_FRAMES})
        self.tracks = keep
        self.lost = [r for r in self.lost if r["until"] >= fi]

    # ---------------- 감지가 끊길 때 영상으로 따라가기 ----------------
    def remember(self, gray, pairs):
        """감지된 번호판의 회색 영상을 저장 (다음 프레임에 감지가 끊기면 이것으로 찾음)"""
        if not self.hold:
            return
        for t, _ in pairs:
            if t.state == "entering":
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in t.box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(gray.shape[1], x2), min(gray.shape[0], y2)
            if x2 - x1 < 8 or y2 - y1 < 4:
                continue
            if t.tmpl is not None:                          # 크기가 갑자기 바뀐 박스(반쯤 가려짐 등)로는 본보기를 바꾸지 않음
                th, tw = t.tmpl.shape[:2]
                if not (0.8 < (x2 - x1) / tw < 1.25 and 0.8 < (y2 - y1) / th < 1.25):
                    continue
            t.tmpl = gray[y1:y2, x1:x2].copy()

    def bridge(self, gray):
        """감지가 끊긴 추적(읽는 중·고정)을 직전 번호판 영상으로 예상 위치 둘레에서 찾음 → 찾은 추적 목록.
        찾으면 그 위치로 옮기고 끊긴 횟수를 0 으로 (BRIDGE_MAX 프레임 연속까지만, 그 뒤는 감지가 다시 와야 함)"""
        if not self.hold:
            return []
        found = []
        H, W = gray.shape[:2]
        for t in self.tracks:
            if t.seen or t.tmpl is None or t.state == "entering" or t.bridged >= BRIDGE_MAX:
                continue
            th, tw = t.tmpl.shape[:2]
            cx, cy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2     # 이미 속도만큼 옮긴 예상 위치
            mx = tw * 0.5 + abs(t.v[0]) * 2 + 8
            my = th * 0.5 + abs(t.v[1]) * 2 + 8
            x1, y1 = int(max(0, cx - tw / 2 - mx)), int(max(0, cy - th / 2 - my))
            x2, y2 = int(min(W, cx + tw / 2 + mx)), int(min(H, cy + th / 2 + my))
            if x2 - x1 <= tw or y2 - y1 <= th:
                continue
            res = cv2.matchTemplate(gray[y1:y2, x1:x2], t.tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < BRIDGE_SCORE:
                continue
            fx, fy = x1 + loc[0] + tw / 2, y1 + loc[1] + th / 2
            if np.hypot(fx - cx, fy - cy) > BRIDGE_SHIFT * tw + 4:
                continue                                    # 예상 위치에서 많이 벗어남 (멈춘 배경 등에 붙는 것 막기)
            t.box = np.array([fx - tw / 2, fy - th / 2, fx + tw / 2, fy + th / 2], float)   # 속도는 그대로 유지
            t.missed, t.seen = 0, True
            t.bridged += 1
            found.append(t)
        return found

    # ---------------- 글자 ----------------
    def add_prediction(self, t, text, conf, fi, sharp=None):
        """읽은 글자 추가 → "lock" (이번에 고정) / "restore" (같은 번호의 이전 추적을 되살림) / None.
        빈 글자(못 읽음)는 세지 않음"""
        if t.state != "measuring":
            return None
        if not text:
            return None
        if self.reid and self._restore(t, str(text), fi):
            return "restore"
        if t.no is None:                                    # 되살릴 번호가 아님 → 새 화면 ID
            self._give_no(t)
        t.preds.append((str(text), float(conf), sharp))
        if len(t.preds) >= self.lock_frames:
            t.lock(fi)
            return "lock"
        return None

    def _restore(self, t, text, fi):
        """같은 번호로 고정됐던 추적(끝났거나 감지가 끊겨 떠도는 중)이 있으면 그 화면 ID·고정값을 이어받음"""
        rec = None
        for o in self.tracks:
            if o is not t and o.state == "locked" and o.text == text and not o.seen:
                rec = {"no": o.no, "text": o.text, "ratio": o.ratio, "conf": o.conf}
                self.tracks.remove(o)                       # 떠도는 옛 추적은 새 추적으로 바꿈
                break
        if rec is None:
            for r in reversed(self.lost):
                if r["text"] == text and r["until"] >= fi:
                    rec = r
                    self.lost.remove(r)
                    break
        if rec is None:
            return False
        if t.no is not None and t.no == self.next_no - 1:   # 방금 받은 새 화면 ID 는 돌려줌 (번호가 비지 않게)
            self.next_no -= 1
        t.no, t.text, t.ratio, t.conf = rec["no"], rec["text"], rec["ratio"], rec["conf"]
        t.state, t.lock_frame, t.lock_reason = "locked", fi, "재인식"
        return True
