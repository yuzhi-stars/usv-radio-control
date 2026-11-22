# -*- coding: utf-8 -*-
r"""
================================================================================
 板端版本（RDK X3 / X5 / 树莓派 / 任何 Linux）—— 无窗口、无 GUI 依赖
================================================================================

和桌面版 plate_digit_camera.py 的区别
    · 完全不碰 cv2.imshow / tkinter（板子上没有 X11 也不影响）
    · 结果用 JSON 一行一条打到 stdout，方便被 ROS2 节点 / 串口桥 / 上位机吞进去
    · 兼容 Python 3.8+（RDK X3 是 20.04/Python3.8，X5 是 22.04/Python3.10）
    · Pillow 可选：没有 PIL 就退回 OpenCV 内置笔画字体（但精度会掉到约 82%，见 README）
    · 支持 --bench 只跑帧率基准，不打印识别结果

依赖（二者取一）
    sudo apt install -y python3-opencv python3-numpy fonts-dejavu-core
    # 或 pip install opencv-python-headless numpy

用法
    python3 plate_digit_rdk.py                       # 默认 /dev/video0，打印 JSON
    python3 plate_digit_rdk.py --bench 100           # 只测帧率
    python3 plate_digit_rdk.py --camera 0 --width 640 --height 480 --stride 2
    python3 plate_digit_rdk.py --video 场地.mp4      # 用录像调试，不用相机
    python3 plate_digit_rdk.py --video 4号牌.jpg     # 单张图也行

输出样例（每行一个 JSON，稳定的读数为一次）
    {"ts": 1763800000.12, "digit": "6", "score": 0.86, "angle": 180, "plate_px": 185, "fps": 29.4}
================================================================================
"""
from __future__ import print_function

import sys
import os
import json
import time
import argparse
from collections import deque, Counter

import cv2
import numpy as np

# ============================== CONFIG ==============================
ALLOWED_DIGITS = "24678"   # 场地数字集合。★ 别加 9：倒置的 6 会歧义
MIN_SCORE = 0.65
SMOOTH_N = 8               # 时间平滑窗口（帧）
STABLE_RATIO = 0.6
SMOOTH_ANNOUNCE = 3        # 连续一致多少帧才输出
LOST_FRAMES = 15           # 连续多少帧没结果算丢失
PLATE_SIZE = 200
DIGIT_CANVAS = 28
ANGLE_STEP = 15
# ===================================================================

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\impact.ttf",
]

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:
    HAS_PIL = False


# ---------------------------------------------------------------- 模板
def _render_font(ch, fp):
    """渲染单个数字。Pillow 8 前后 API 不同，这里两套都兼容。"""
    c = Image.new("L", (160, 160), 0)
    d = ImageDraw.Draw(c)
    f = ImageFont.truetype(fp, 130)
    if hasattr(d, "textbbox"):                     # Pillow >= 8
        bb = d.textbbox((0, 0), ch, font=f)
        x0, y0, x1, y1 = bb[0], bb[1], bb[2], bb[3]
    else:                                          # 老版本 Pillow
        w, h = d.textsize(ch, font=f)
        x0, y0, x1, y1 = 0, 0, w, h
    d.text((80 - (x0 + x1) // 2, 80 - (y0 + y1) // 2), ch, fill=255, font=f)
    return np.array(c)


def _render_cv(ch):
    c = np.zeros((160, 160), np.uint8)
    cv2.putText(c, ch, (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, 18, cv2.LINE_AA)
    return c


def normalize_glyph(b):
    ys, xs = np.where(b > 0)
    if len(ys) < 10:
        return None
    crop = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    if h < 3 or w < 1:
        return None
    s = 20.0 / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((DIGIT_CANVAS, DIGIT_CANVAS), np.uint8)
    y0, x0 = (DIGIT_CANVAS - nh) // 2, (DIGIT_CANVAS - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = crop
    return out.astype(np.float32) / 255.0


def unitize(v):
    v = v.reshape(-1).astype(np.float32)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def rotate28(img, ang):
    M = cv2.getRotationMatrix2D((13.5, 13.5), ang, 1.0)
    return cv2.warpAffine(img, M, (DIGIT_CANVAS, DIGIT_CANVAS),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def build_templates(verbose=False):
    fonts = [f for f in FONT_CANDIDATES if os.path.isfile(f)] if HAS_PIL else []
    rows, meta = [], []
    angles = list(range(0, 360, ANGLE_STEP))
    for ch in "0123456789":
        bases = []
        for fp in fonts:
            try:
                bases.append(normalize_glyph(_render_font(ch, fp)))
            except Exception:
                pass
        if not bases:
            bases = [normalize_glyph(_render_cv(ch))]   # 兜底
        for base in bases:
            if base is None:
                continue
            for a in angles:
                u = unitize(rotate28(base, a) if a else base)
                if u is not None:
                    rows.append(u)
                    meta.append((ch, a))
    if verbose:
        if not fonts:
            print("[WARN] 没找到粗体 TTF（或没装 Pillow），已退回 OpenCV 笔画字体；"
                  "实测精度会从 ~100% 掉到 ~82%，建议 apt install fonts-dejavu-core",
                  file=sys.stderr)
        else:
            print("[INFO] 字体 %d 个，模板 %d 个" % (len(fonts), len(rows)),
                  file=sys.stderr)
    return np.vstack(rows).astype(np.float32), meta


# ---------------------------------------------------------------- 检测
def order_corners(quad):
    pts = np.array(quad, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def warp_plate(img, quad):
    dst = np.array([[0, 0], [PLATE_SIZE - 1, 0],
                    [PLATE_SIZE - 1, PLATE_SIZE - 1], [0, PLATE_SIZE - 1]], np.float32)
    M = cv2.getPerspectiveTransform(order_corners(quad), dst)
    return cv2.warpPerspective(img, M, (PLATE_SIZE, PLATE_SIZE))


def find_candidates(bgr, s_lo=130, v_lo=110):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, s_lo, v_lo]), np.array([40, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bgr.shape[:2]
    out = []
    for c in cnts:
        if not (4e-4 <= cv2.contourArea(c) / float(H * W) <= 0.08):
            continue
        peri = cv2.arcLength(c, True)
        quad = None
        for eps in np.arange(0.02, 0.12, 0.01):
            ap = cv2.approxPolyDP(c, eps * peri, True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                quad = ap.reshape(4, 2).astype(np.float32)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(c))
        side = min(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[1] - quad[2]))
        if side < 10:
            continue
        if max(np.linalg.norm(quad[0] - quad[1]),
               np.linalg.norm(quad[1] - quad[2])) / max(1.0, side) > 2.0:
            continue
        out.append((quad, float(side)))
    return out


def glyph_of(plate):
    v = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)[:, :, 2], (5, 5), 0)
    _, b = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = int(PLATE_SIZE * 0.08)
    b[:m, :] = 0
    b[-m:, :] = 0
    b[:, :m] = 0
    b[:, -m:] = 0
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    ink = cv2.contourArea(c) / float(PLATE_SIZE * PLATE_SIZE)
    if h < PLATE_SIZE * 0.3 or w < PLATE_SIZE * 0.06 or w / max(h, 1) > 1.6:
        return None, ink
    g = np.zeros_like(b)
    cv2.drawContours(g, [c], -1, 255, -1)
    return g, ink


def classify(plate, T, meta):
    g, ink = glyph_of(plate)
    if g is None or not (0.05 <= ink <= 0.55):
        return None, None, 0.0
    norm = normalize_glyph(g)
    if norm is None:
        return None, None, 0.0
    u = unitize(norm)
    if u is None:
        return None, None, 0.0
    scores = T @ u
    best = (None, None, -9.0)
    for i, (ch, a) in enumerate(meta):
        if ch in ALLOWED_DIGITS and scores[i] > best[2]:
            best = (ch, a, float(scores[i]))
    return best


def detect(img, T, meta):
    out = []
    for quad, side in find_candidates(img):
        ch, ang, sc = classify(warp_plate(img, quad), T, meta)
        if ch is None:
            continue
        out.append({"digit": ch, "angle": ang, "score": sc, "plate_px": side})
    return out


# ---------------------------------------------------------------- 取帧
def open_source(args):
    """返回 (读取函数, 关闭函数, 描述)"""
    if args.video:
        if os.path.splitext(args.video)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            img = cv2.imdecode(np.fromfile(args.video, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit("读不了图片：%s" % args.video)
            # 单张图也要喂够帧数，否则时间平滑永远攒不满、出不了结果
            state = {"left": max(args.bench, SMOOTH_N * 2 + 2)}

            def read_img():
                if state["left"] <= 0:
                    return False, None
                state["left"] -= 1
                return True, img
            return read_img, (lambda: None), "图片 %s" % args.video

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit("打不开视频：%s" % args.video)
        return (cap.read, cap.release, "视频 %s" % args.video)

    # 相机：Linux 用 V4L2，Windows 用 DSHOW
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    elif sys.platform.startswith("win"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("打不开 camera %d（MIPI 摄像头通常不能直接用 cv2 打开，"
                         "需要走 RDK 的 mipi_cam/v4l2 节点或先用 USB 摄像头）" % args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(10):                       # 丢掉开流前几帧（可能全黑）
        cap.read()
    return (cap.read, cap.release,
            "camera %d (%dx%d @%d)" % (args.camera, int(cap.get(3)), int(cap.get(4)),
                                       int(cap.get(cv2.CAP_PROP_FPS))))


def main():
    ap = argparse.ArgumentParser(description="黄底黑字数字牌识别（板端/无 GUI 版）")
    ap.add_argument("--camera", type=int, default=0, help="摄像头序号，默认 0")
    ap.add_argument("--video", default=None, help="改用视频或图片文件（调试用）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧处理一次（1=每帧）")
    ap.add_argument("--bench", type=int, default=0, help="只跑 N 帧测速，不输出识别结果")
    ap.add_argument("--quiet", action="store_true", help="不输出 INFO 日志")
    args = ap.parse_args()

    def log(*a):
        if not args.quiet:
            print(*a, file=sys.stderr)

    t0 = time.perf_counter()
    T, meta = build_templates(verbose=not args.quiet)
    log("模板就绪 %.0f ms，允许数字 %s，门限 %.2f"
        % ((time.perf_counter() - t0) * 1000, ALLOWED_DIGITS, MIN_SCORE))

    read, close, desc = open_source(args)
    log("输入：%s" % desc)

    hist = deque(maxlen=SMOOTH_N)
    pending, pending_n = None, 0
    announced = None
    lost = 0
    n = seen = 0
    limit = args.bench if args.bench else 0          # --bench N：处理 N 帧后退出
    emit = not args.bench                            # 测速模式不打印识别结果
    t_start = time.perf_counter()
    proc_time = 0.0
    try:
        while True:
            ok, frame = read()
            if not ok or frame is None:
                break
            seen += 1
            if args.stride > 1 and seen % args.stride:
                continue
            n += 1
            tp = time.perf_counter()
            try:
                results = detect(frame, T, meta)
            except Exception as e:                       # noqa: BLE001
                results = []
                log("[WARN] 本帧出错已跳过：%r" % e)
            proc_time += time.perf_counter() - tp

            credible = [r for r in results if r["score"] >= MIN_SCORE]
            hist.append(credible[0]["digit"] if credible else None)
            vals = [d for d in hist if d]
            stable = None
            if vals and len(vals) >= SMOOTH_N * STABLE_RATIO:
                stable = Counter(vals).most_common(1)[0][0]

            if stable:
                lost = 0
                pending, pending_n = (stable, pending_n + 1) if stable == pending \
                    else (stable, 1)
                if pending_n >= SMOOTH_ANNOUNCE and stable != announced:
                    best = next((r for r in credible if r["digit"] == stable), None)
                    out = {"ts": round(time.time(), 3), "digit": stable,
                           "fps": round(n / max(1e-6, time.perf_counter() - t_start), 1)}
                    if best:
                        out.update({"score": round(best["score"], 3),
                                    "angle": best["angle"],
                                    "plate_px": int(best["plate_px"])})
                    if emit:
                        print(json.dumps(out, ensure_ascii=False), flush=True)
                    announced = stable
            else:
                pending, pending_n = None, 0
                lost += 1
                if announced is not None and lost >= LOST_FRAMES:
                    if emit:
                        print(json.dumps({"ts": round(time.time(), 3), "digit": None,
                                          "event": "lost"}), flush=True)
                    announced = None

            if limit and n >= limit:
                break
    except KeyboardInterrupt:
        pass
    finally:
        close()

    total = time.perf_counter() - t_start
    log("")
    log("处理 %d 帧（读到 %d 帧），总耗时 %.2f s -> %.1f fps；纯处理 %.1f ms/帧"
        % (n, seen, total, n / max(1e-6, total),
           (proc_time / max(1, n)) * 1000 if n else 0.0))


if __name__ == "__main__":
    main()
