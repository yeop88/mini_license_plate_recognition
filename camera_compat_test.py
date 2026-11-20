# -*- coding: utf-8 -*-
r"""
카메라 호환성 시험 (창 없이) — 지원 모드를 전부 재서 최고 화질 / 최고 프레임을 정하고 camera_profiles.json 에 저장

    python camera_compat_test.py              # 지난번에 쓴 카메라 (없으면 첫 번째)
    python camera_compat_test.py --camera 1   # 1번 카메라
    python camera_compat_test.py --no-save    # 저장하지 않고 결과만 보기

GUI 의 [카메라 설정] → [호환성 시험 (스펙 다시 측정)] 과 같은 측정 (app\models\camera_profile.probe_camera).
GUI 가 카메라를 쓰고 있으면 열 수 없으니 GUI 를 닫고 실행한다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.camera import find_cameras, plain_name  # noqa: E402
from app.models.camera_profile import ProfileRepository, probe_camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="카메라 호환성 시험 (최고 화질 / 최고 프레임)")
    ap.add_argument("--camera", type=int, default=None, help="카메라 번호 (생략하면 지난번 카메라)")
    ap.add_argument("--width", type=int, default=1920, help="'기본' 해상도 너비 (기본 1920)")
    ap.add_argument("--height", type=int, default=1080, help="'기본' 해상도 높이 (기본 1080)")
    ap.add_argument("--no-save", action="store_true", help="camera_profiles.json 에 저장하지 않음")
    a = ap.parse_args()

    repo = ProfileRepository()
    print("카메라 검색 중...", flush=True)
    cams = find_cameras()
    if not cams:
        sys.exit("카메라가 없습니다")
    names = {i: plain_name(l) for i, l in cams}
    if a.camera is None:
        index = next((i for i, n in names.items() if n == repo.last_camera), cams[0][0])
    elif a.camera in names:
        index = a.camera
    else:
        sys.exit(f"카메라 {a.camera}번이 없습니다. 있는 카메라: {names}")
    name = names[index]
    print(f"카메라 {index}번 '{name}' 시험 시작", flush=True)

    prof = probe_camera(index, a.width, a.height, log=lambda m: print(m, flush=True))
    if prof is None or "error" in prof:
        sys.exit(f"시험 실패: {(prof or {}).get('error', '중단됨')}")

    print("\n| 형식 | 요청 해상도 | 실제 | 알려 준 FPS | 실측 FPS | 검은 프레임 | 밝기 | 결과 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in prof["modes"]:
        w, h = m["request"].split()[1].split("x")
        print(f"| {m['request_fourcc']} | {w}x{h} | {m['width']}x{m['height']} | {m['advertised_fps'] or '-'} | "
              f"{m['fps']} | {m['black_ratio']} | {m['brightness']} | {'OK' if m['ok'] else m['problem']} |")
    b = prof["best"]
    fmt = lambda m: f"{m['fourcc']} {m['width']}x{m['height']} {m['fps']:.1f}fps" if m else "없음"
    print(f"\n최고 화질   : {fmt(b['max_quality'])}")
    print(f"최고 프레임 : {fmt(b['max_fps'])}")
    print(f"기본 {a.width}x{a.height}: {fmt(b['request'])}")

    if not a.no_save:
        old = repo.get(name) or {}
        for k in ("calibration_file", "model", "last_resolution", "last_used"):   # 사용자가 적어 둔 값은 유지
            if old.get(k):
                prof[k] = old[k]
        repo.put(name, prof)
        repo.save()
        print(f"\n저장: {repo.path} ('{name}')")


if __name__ == "__main__":
    main()
