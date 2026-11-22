# -*- coding: utf-8 -*-
r"""
================================================================================
 摄像头实时版：对着黄底黑字数字牌，实时显示识别结果
================================================================================

用法
    直接运行（VSCode 里按 F5 也行）：
        python plate_digit_camera.py
    先用无窗口模式自检 60 帧（不开窗口，适合确认相机通不通）：
        python plate_digit_camera.py --test 60

窗口里的按键
    q 或 ESC   退出
    s          存一张带标注的快照到 output\snap_<时间>.jpg
    p          显示/隐藏「矫正后的牌面」小图（用来验收分类器看到了什么）

本机实测（2026-11-22，DSHOW 后端）
    1280×720、FPS 设 30  ->  实测 29 fps
    1280×720 不设 FPS    ->  只有 5.7 fps（所以脚本里一定设了 FPS 和 MJPG）

依赖：pip install opencv-python numpy pillow（本机已装）
================================================================================
"""

import os
import sys
import time
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================== CONFIG ==============================
CAMERA_INDEX = 0          # 就用 camera 0
FRAME_W, FRAME_H = 1280, 720
TARGET_FPS = 30
ALLOWED_DIGITS = "24678"  # 场地数字（别加 9，否则倒置的 6 会歧义）
MIN_SCORE = 0.65          # 置信度门限，正常读数 0.80~0.92
SMOOTH_N = 8              # 时间平滑窗口（帧）：最近 8 帧里同一数字占多数才认
STABLE_RATIO = 0.6        # 占比超过这个值才算「稳定」
SHOW_PLATE = True         # 是否显示矫正后的牌面小图
# ===================================================================

DIGIT_CANVAS = 28
PLATE_SIZE = 200
ANGLE_STEP = 15

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ariblk.ttf",
    r"C:\Windows\Fonts\arialn.ttf",
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


# ---------------------------------------------------------------- 模板
def _render_font(ch, fp):
    c = Image.new("L", (160, 160), 0)
    d = ImageDraw.Draw(c)
    f = ImageFont.truetype(fp, 130)
    bb = d.textbbox((0, 0), ch, font=f)
    d.text((80 - (bb[0] + bb[2]) // 2, 80 - (bb[1] + bb[3]) // 2), ch, fill=255, font=f)
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


def build_templates():
    fonts = [f for f in FONT_CANDIDATES if os.path.isfile(f)]
    rows, meta = [], []
    angles = list(range(0, 360, ANGLE_STEP))
    for ch in "0123456789":
        bases = ([normalize_glyph(_render_font(ch, f)) for f in fonts] if fonts
                 else [normalize_glyph(_render_cv(ch))])
        for base in bases:
            if base is None:
                continue
            for a in angles:
                u = unitize(rotate28(base, a) if a else base)
                if u is not None:
                    rows.append(u)
                    meta.append((ch, a))
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


def classify(plate, T, meta, allowed):
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
        if ch in allowed and scores[i] > best[2]:
            best = (ch, a, float(scores[i]))
    return best


def detect(img, T, meta):
    """返回所有候选（不按阈值过滤，让 HUD 能看到「差一点的」目标）"""
    out = []
    for quad, side in find_candidates(img):
        plate = warp_plate(img, quad)
        d, a, sc = classify(plate, T, meta, ALLOWED_DIGITS)
        if d is None:
            continue
        out.append({"digit": d, "angle": a, "score": sc, "side": side,
                    "quad": quad, "plate": plate,
                    "center": (float(quad[:, 0].mean()), float(quad[:, 1].mean()))})
    out.sort(key=lambda r: (round(r["center"][1] / 60.0), r["center"][0]))
    return out


# ---------------------------------------------------------------- 绘制
def draw_result(frame, results, stable, show_plate):
    vis = frame.copy()
    for r in results:
        ok = r["score"] >= MIN_SCORE
        color = (0, 255, 0) if ok else (0, 165, 255)     # 绿=可信，橙=低于门限
        pts = order_corners(r["quad"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2)
        cx, cy = int(r["center"][0]), int(r["center"][1])
        label = "%s %.2f %ddeg" % (r["digit"], r["score"], r["angle"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (cx - 3, cy - th - 10), (cx + tw + 5, cy + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (cx, cy - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)

    # 左上角大号字：稳定结果
    if stable:
        txt = "STABLE  %s" % stable
        col = (0, 255, 0)
    else:
        txt = "..."
        col = (200, 200, 200)
    cv2.putText(vis, txt, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(vis, txt, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3, cv2.LINE_AA)

    # 右上角显示矫正后的牌面小图（验收分类器输入）
    if show_plate and results:
        best = max(results, key=lambda r: r["score"])
        thumb = cv2.resize(best["plate"], (120, 120), interpolation=cv2.INTER_AREA)
        x0, y0 = vis.shape[1] - 132, 12
        vis[y0:y0 + 120, x0:x0 + 120] = thumb
        cv2.rectangle(vis, (x0 - 2, y0 - 2), (x0 + 121, y0 + 121), (255, 255, 255), 1)
    return vis


def _backends():
    """Windows 上用 DSHOW / MSMF；其它平台用默认后端。
       （Windows 上再额外试 CAP_ANY 会走到 FFMPEG，缺 libavdevice 只会刷一条无用 WARN）"""
    if sys.platform.startswith("win"):
        return [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
    return [(cv2.CAP_ANY, "默认")]


def open_camera(index=None, width=None, height=None, fps=None):
    """按给定序号打开摄像头：依次尝试 DSHOW / MSMF / 默认后端。
       参数设成实测能跑满帧的那组（不设 FPS 某些相机会掉到 5.7 fps）。"""
    index = CAMERA_INDEX if index is None else index
    width = FRAME_W if width is None else width
    height = FRAME_H if height is None else height
    fps = TARGET_FPS if fps is None else fps
    for flag, name in _backends():
        try:
            cap = cv2.VideoCapture(index, flag) if flag != cv2.CAP_ANY \
                else cv2.VideoCapture(index)
        except Exception:
            continue
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(10):                       # 丢掉开流前几帧（可能是全黑）
            cap.read()
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 3:
            print("相机就绪：序号 %d，后端 %s，实际 %dx%d，报告 %d fps"
                  % (index, name, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(cap.get(cv2.CAP_PROP_FPS))))
            return cap
        cap.release()
    return None


def _silence_cv_log(enable=True):
    """开关 OpenCV 自己的日志。扫描摄像头序号时，不存在的序号会刷
       'backend ... can't be used to capture by index' 之类的 WARN，很吵。"""
    try:
        lvl = (cv2.utils.logging.LOG_LEVEL_SILENT if enable
               else cv2.utils.logging.LOG_LEVEL_ERROR)
        cv2.utils.logging.setLogLevel(lvl)
        return True
    except Exception:
        return False


def list_cameras(max_index=6, width=None, height=None):
    """逐个试 0..max_index-1，列出真正能出图的摄像头。
       USB 摄像头在 Windows 上常常是 1 或 2，且拔插后会变，所以调试前先跑这个。"""
    width = FRAME_W if width is None else width
    height = FRAME_H if height is None else height
    _silence_cv_log(True)
    print("扫描摄像头序号 0~%d（请求 %dx%d @%d fps）..."
          % (max_index - 1, width, height, TARGET_FPS))
    print("%-6s %-8s %-12s %-8s %s" % ("序号", "后端", "实际分辨率", "报告fps", "画面亮度"))
    found = []
    for idx in range(max_index):
        got = None
        for flag, name in _backends():
            try:
                cap = cv2.VideoCapture(idx, flag) if flag != cv2.CAP_ANY \
                    else cv2.VideoCapture(idx)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            for _ in range(6):
                cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                got = (idx, name,
                       "%dx%d" % (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                       int(cap.get(cv2.CAP_PROP_FPS)),
                       float(frame.mean()))
            cap.release()
            if got:
                break
        if got:
            found.append(got)
            print("%-6d %-8s %-12s %-8d %.0f" % got)
    _silence_cv_log(False)
    print("-" * 46)
    if found:
        print("可用的：%s" % "、".join("--camera %d" % g[0] for g in found))
        print("用法示例：python plate_digit_camera.py --camera %d" % found[0][0])
    else:
        print("一个都没扫到。检查：① 摄像头是否插好；② 是否被微信/QQ/腾讯会议占用；"
              "③ 设备管理器里有没有识别出来。")
    return found


# ---------------------------------------------------------------- 显示层
# 本机装的是 opencv-python-headless（cv2.getBuildInformation() 里 GUI: NONE），
# cv2.imshow 会直接抛 "The function is not implemented"。
# 所以这里做两套显示：能开 cv2 窗口就用，不能就退回 tkinter（Python 自带 + PIL）。
import contextlib


@contextlib.contextmanager
def _mute_stderr():
    """临时把 C 层的 stderr 指到 null。
       用途：Tk 建窗/贴图时会让 libpng 吐 'iCCP: known incorrect sRGB profile' 警告，
       30 帧/秒就是 30 行噪音，把终端刷满。只包住贴图那几行，其余错误照常显示。"""
    saved = None
    try:
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        saved = None
    try:
        yield
    finally:
        if saved is not None:
            try:
                os.dup2(saved, 2)
                os.close(saved)
            except Exception:
                pass


class _CvViewer:
    """原生 OpenCV 窗口"""

    def __init__(self, title):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(self, frame):
        cv2.imshow(self.title, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyAllWindows()


class _TkViewer:
    """tkinter 窗口（不需要重装 OpenCV）"""

    def __init__(self, title):
        import tkinter as tk
        from PIL import ImageTk
        self.ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.lbl = tk.Label(self.root, bd=0)
        self.lbl.pack()
        self._key = None
        self._closed = False
        self._photo = None
        self.root.bind("<KeyPress>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 把窗口提到最前并抢焦点（否则键盘事件可能落在 VSCode 的终端上）
        self.root.update_idletasks()
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(1000, lambda: self.root.attributes("-topmost", False))
        try:
            self.root.focus_force()
        except Exception:
            pass

    def _on_key(self, e):
        if e.keysym == "Escape":
            self._key = 27
        elif len(e.keysym) == 1:
            self._key = ord(e.keysym.lower())
        else:
            self._key = None

    def _on_close(self):
        self._closed = True

    def show(self, frame):
        if self._closed:
            return ord("q")
        h, w = frame.shape[:2]
        max_w = 1100                                    # 显示端缩一点，省时间
        if w > max_w:
            frame = cv2.resize(frame, (max_w, int(h * max_w / w)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with _mute_stderr():
            self._photo = self.ImageTk.PhotoImage(Image.fromarray(rgb))
            self.lbl.configure(image=self._photo)
            self.root.update()                          # 处理事件（含键盘/关窗）
        key, self._key = self._key, None
        return key

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def make_viewer(title):
    """能开 cv2 窗口就开，否则退 tkinter。返回 (viewer, 后端名)"""
    try:
        cv2.namedWindow("_gui_probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_gui_probe")
        return _CvViewer(title), "OpenCV"
    except cv2.error:
        # Tk 首次建窗/首次贴图会从 libpng 吐一串 iCCP 警告，这里一并静音
        with _mute_stderr():
            v = _TkViewer(title)
        return v, "tkinter"


# ---------------------------------------------------------------- 主流程
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import argparse
    ap = argparse.ArgumentParser(
        description="黄底黑字数字牌实时识别（摄像头序号可选）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX,
                    help="摄像头序号。USB 摄像头常是 1 或 2，先用 --list 查")
    ap.add_argument("--width", type=int, default=FRAME_W, help="请求宽度")
    ap.add_argument("--height", type=int, default=FRAME_H, help="请求高度")
    ap.add_argument("--fps", type=int, default=TARGET_FPS, help="请求帧率")
    ap.add_argument("--list", action="store_true", help="扫描并列出可用摄像头后退出")
    ap.add_argument("--test", type=int, default=0, metavar="N",
                    help="无窗口跑 N 帧自检后退出（不开窗口，适合远程/无桌面环境）")
    args = ap.parse_args()

    if args.list:
        list_cameras(width=args.width, height=args.height)
        return

    test_n = args.test
    # 调试用：环境变量 PD_AUTOQUIT=N -> 显示 N 帧后自动退出并存一张 HUD 预览图
    autoquit = int(os.environ.get("PD_AUTOQUIT", "0") or 0)

    t0 = time.perf_counter()
    T, meta = build_templates()
    print("模板 %d 个（%.0f ms）  允许数字 %s  门限 %.2f"
          % (T.shape[0], (time.perf_counter() - t0) * 1000, ALLOWED_DIGITS, MIN_SCORE))

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        print("打不开 camera %d。排查顺序：① 先跑 `python plate_digit_camera.py --list` "
              "看 USB 摄像头是哪个序号；② 关掉微信/QQ/腾讯会议等占用摄像头的程序；"
              "③ 设备管理器里确认摄像头已识别。" % args.camera)
        return

    hist = deque(maxlen=SMOOTH_N)
    prev = time.perf_counter()
    fps_smooth = 0.0
    dark_reported = False
    n = 0
    miss = 0
    show_plate = SHOW_PLATE          # 用局部变量，避免 global 污染
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    # 读数状态机（修掉「stable 来自历史、但当前帧已无候选」导致的 IndexError）
    announced = None        # 上一次公告出去的数字
    announced_info = None   # 上一次公告对应的详细信息（置信度/角度/像素）
    pending, pending_n = None, 0      # 去抖：连续 3 帧一致才公告
    lost_n = 0                        # 连续多少帧没有可信结果
    err_n = 0                         # 单帧异常计数（只打印前几次，别刷屏）

    viewer = None
    if not test_n:
        viewer, backend = make_viewer("plate digit cam%d  (q=quit, s=snap, p=plate)"
                                      % args.camera)
        print("显示后端：%s%s" % (backend, "（OpenCV 是 headless 版，改用 tkinter，不影响识别）"
                                  if backend == "tkinter" else ""))
        print("窗口已开：点一下窗口再按键 —— q/ESC 退出，s 存快照，p 切换牌面小图")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读帧失败，退出")
                break
            n += 1

            # 单帧异常不让整个窗口崩掉
            try:
                results = detect(frame, T, meta)
            except Exception as e:                       # noqa: BLE001
                results = []
                err_n += 1
                if err_n <= 3:
                    print("⚠ 本帧检测出错（已跳过，不影响继续运行）：%r" % e)

            credible = [r for r in results if r["score"] >= MIN_SCORE]
            hist.append(credible[0]["digit"] if credible else None)
            vals = [d for d in hist if d]
            stable = None
            if vals and len(vals) >= SMOOTH_N * STABLE_RATIO:
                stable = Counter(vals).most_common(1)[0][0]

            # ---- 状态推进：去抖 + 丢失判定
            if stable:
                lost_n = 0
                pending, pending_n = (stable, pending_n + 1) if stable == pending \
                    else (stable, 1)
                if pending_n >= 3 and stable != announced:
                    info = next((r for r in credible if r["digit"] == stable), None) \
                        or announced_info
                    if info:
                        print("→ 数字 %s  (置信度 %.2f, 角度 %d°, 牌面 %d px)"
                              % (stable, info["score"], info["angle"], info["side"]))
                    else:
                        print("→ 数字 %s" % stable)
                    announced, announced_info = stable, info
            else:
                pending, pending_n = None, 0
                lost_n += 1
                if announced is not None and lost_n >= 15:   # 约 0.5 s 才判丢失
                    print("→ 目标丢失")
                    announced, announced_info = None, None

            # HUD 上显示谁：有稳定值显示稳定值；刚丢失则保留 12 帧（避免一闪一闪）
            shown = stable if stable else (announced if lost_n <= 12 else None)

            # 全黑检测（镜头被挡 / 没灯）
            if frame.mean() < 5 and not dark_reported:
                print("⚠ 画面几乎是黑的：检查镜头盖/光线，或相机刚开流还没出图")
                dark_reported = True
            elif frame.mean() >= 5:
                dark_reported = False

            now = time.perf_counter()
            fps_smooth = (0.9 * fps_smooth + 0.1 * (1.0 / max(1e-6, now - prev))
                          if fps_smooth else 1.0 / max(1e-6, now - prev))
            prev = now

            # ---- 无窗口自检模式
            if test_n:
                if n == 1:
                    print("自检模式：跑 %d 帧不开窗口" % test_n)
                if n % 20 == 0 or n == test_n:
                    print("  第 %d 帧：fps %.1f  候选 %d 个  稳定 %s"
                          % (n, fps_smooth, len(results), shown or "-"))
                if n >= test_n:
                    print("自检完成：平均 %.1f fps" % fps_smooth)
                    break
                continue

            vis = draw_result(frame, results, shown, show_plate)
            cv2.putText(vis, "%.0f fps" % fps_smooth, (14, vis.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            # 连续没结果时给个提示，免得对着画面干等
            miss = 0 if credible else miss + 1
            if miss > 30:
                hint = "no plate: point at the yellow sign / check light / get closer"
                cv2.putText(vis, hint, (14, vis.shape[0] - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            key = viewer.show(vis)
            if autoquit and n >= autoquit:
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, "hud_preview.jpg")
                cv2.imencode(".jpg", vis)[1].tofile(path)
                print("自动退出（PD_AUTOQUIT=%d），HUD 预览已存 %s" % (autoquit, path))
                break
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, "snap_%s.jpg" % time.strftime("%Y%m%d_%H%M%S"))
                cv2.imencode(".jpg", vis)[1].tofile(path)
                print("快照已存 %s" % path)
            if key == ord("p"):
                show_plate = not show_plate
    finally:
        cap.release()
        if viewer is not None:
            viewer.close()
    print("已退出（共 %d 帧）" % n)


if __name__ == "__main__":
    main()
