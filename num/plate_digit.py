# -*- coding: utf-8 -*-
r"""
================================================================================
 场地数字牌识别（黄底黑字 / 支持倒置与任意旋转）—— 单文件验证脚本
================================================================================
用途
    读一张图或一个文件夹里的图，找出地面上的黄底黑字数字牌，输出：数字、置信度、
    牌面在画面里的旋转角、牌面像素大小。带旋转不变性（整块倒过来也能读）。

依赖（三件，本机已装）
    pip install opencv-python numpy pillow

怎么用
    ① VSCode 里直接按 F5 / 右上角三角运行 —— 默认扫描本脚本同目录下的
       samples 文件夹（没有则扫 images ，再没有则扫脚本同目录的所有图片）
    ② 指定路径（终端里跑）：
         python plate_digit.py "D:\\照片\\场地"
         python plate_digit.py "D:\\照片\\4号牌.jpg"

输出
    终端打印识别结果表格；
    每张图生成一张标注图，存到 <脚本目录>\output\ 下。

怎么调
    下面 CONFIG 区四个参数，按现场情况改。
================================================================================
"""

import os
import sys
import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================== CONFIG ==============================
# 允许的数字集合。本场地只有 2/4/6/7/8；想放开就改成 "0123456789"。
# ★ 注意：允许集合里同时出现 6 和 9 时，牌子整块倒置将无法区分（6 转 180° 就是 9）。
ALLOWED_DIGITS = "24678"

MIN_SCORE = 0.65        # 置信度低于此值直接丢弃（0~1，正常读数在 0.75~0.95）
                        # 调低=更容易接受弱目标（也更容易误检），调高=更严格
PLATE_SIZE = 200        # 透视矫正后的正方形边长（内部用，不用改）
DRAW_OUTPUT = True      # 是否保存标注图
# ===================================================================

DIGIT_CANVAS = 28
ANGLE_STEP = 15                                    # 旋转搜索步长（度），24 个朝向

# 字体候选：Windows 优先，其次 Linux/macOS（找不到就退回 OpenCV 内置笔画字体）
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
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


# ---------------------------------------------------------------- 模板构建
def _render_with_font(ch, font_path):
    """用一个 TrueType 字体把单个数字渲染成 160×160 的白字黑底图"""
    canvas = Image.new("L", (160, 160), 0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, 130)
    bb = draw.textbbox((0, 0), ch, font=font)
    draw.text((80 - (bb[0] + bb[2]) // 2, 80 - (bb[1] + bb[3]) // 2),
              ch, fill=255, font=font)
    return np.array(canvas)


def _render_with_opencv(ch):
    """兜底：没有可用字体时，用 OpenCV 内置笔画字体渲染"""
    canvas = np.zeros((160, 160), np.uint8)
    cv2.putText(canvas, ch, (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, 18, cv2.LINE_AA)
    return canvas


def normalize_glyph(bin_img):
    """裁到笔画外接框 -> 等比缩放到 20×20 -> 贴到 28×28 中心（MNIST 风格归一化）"""
    ys, xs = np.where(bin_img > 0)
    if len(ys) < 10:
        return None
    crop = bin_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    if h < 3 or w < 1:
        return None
    s = 20.0 / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((DIGIT_CANVAS, DIGIT_CANVAS), np.uint8)
    y0, x0 = (DIGIT_CANVAS - nh) // 2, (DIGIT_CANVAS - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = crop
    return canvas.astype(np.float32) / 255.0


def _unitize(vec):
    """拉平 + 去均值 + 单位化，之后两个向量的点积就等于归一化互相关"""
    v = vec.reshape(-1).astype(np.float32)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _rotate28(img28, angle):
    M = cv2.getRotationMatrix2D((13.5, 13.5), angle, 1.0)
    return cv2.warpAffine(img28, M, (DIGIT_CANVAS, DIGIT_CANVAS),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def build_templates():
    """生成模板矩阵 T (N×784) 和标签表 [(数字, 角度), ...]
       多字体 × 多数字 × 多角度 —— 角度这一维就是「倒置/斜放」的免疫力来源"""
    fonts = [f for f in FONT_CANDIDATES if os.path.isfile(f)]
    rows, meta = [], []
    angles = list(range(0, 360, ANGLE_STEP))
    for ch in "0123456789":
        if fonts:
            bases = [normalize_glyph(_render_with_font(ch, fp)) for fp in fonts]
        else:
            bases = [normalize_glyph(_render_with_opencv(ch))]
        for base in bases:
            if base is None:
                continue
            for ang in angles:
                rot = _rotate28(base, ang) if ang else base
                u = _unitize(rot)
                if u is not None:
                    rows.append(u)
                    meta.append((ch, ang))
    if not rows:
        raise RuntimeError("模板生成失败：一个可用字体都没找到，且内置字体渲染也失败")
    return np.vstack(rows).astype(np.float32), meta, len(fonts)


# ---------------------------------------------------------------- 检测
def order_corners_polar(quad):
    """四点排序：绕质心按极角排序，再旋转到「最靠左上」的点开头。
       —— 用 sum/diff 那套经典写法在牌子正好转成 45° 菱形时会乱序，这里避开。"""
    pts = np.array(quad, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def warp_plate(img, quad):
    """四点透视矫正成 PLATE_SIZE 正方形（正方形里牌子的朝向还是自由旋转的）"""
    dst = np.array([[0, 0], [PLATE_SIZE - 1, 0],
                    [PLATE_SIZE - 1, PLATE_SIZE - 1], [0, PLATE_SIZE - 1]], np.float32)
    M = cv2.getPerspectiveTransform(order_corners_polar(quad), dst)
    return cv2.warpPerspective(img, M, (PLATE_SIZE, PLATE_SIZE))


def find_candidates(bgr, s_lo=130, v_lo=110):
    """HSV 找黄牌候选。返回 [(quad, side_px), ...]
       s_lo 是饱和度下限：调低会把米色地面/木地板也算进来（本项目踩过这个坑）"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, s_lo, v_lo]), np.array([40, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bgr.shape[:2]
    out = []
    for c in cnts:
        ratio = cv2.contourArea(c) / float(H * W)
        if not (4e-4 <= ratio <= 0.08):        # 太小（噪点）或太大（地面）都丢
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
        side = min(np.linalg.norm(quad[0] - quad[1]),
                   np.linalg.norm(quad[1] - quad[2]))
        if side < 10:
            continue
        if max(np.linalg.norm(quad[0] - quad[1]),
               np.linalg.norm(quad[1] - quad[2])) / max(1.0, side) > 2.0:
            continue                            # 牌子接近正方形
        out.append((quad, float(side)))
    return out


def glyph_of(plate):
    """从矫正后的牌面里取出黑色数字笔画（二值图），顺便算墨迹占比"""
    v = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)[:, :, 2], (5, 5), 0)
    _, b = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = int(PLATE_SIZE * 0.08)                  # 去掉贴边的牌框/反光
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
    """返回 (数字, 角度, 置信度)；在 allowed 集合内做「数字×朝向」联合搜索"""
    g, ink = glyph_of(plate)
    if g is None or not (0.05 <= ink <= 0.55):     # 墨迹太少=没字；太多=整块黑
        return None, None, 0.0
    norm = normalize_glyph(g)
    if norm is None:
        return None, None, 0.0
    u = _unitize(norm)
    if u is None:
        return None, None, 0.0
    scores = T @ u                                  # 一次算完 N 个模板
    best = (None, None, -9.0)
    for i, (ch, ang) in enumerate(meta):
        if ch not in allowed:
            continue
        if scores[i] > best[2]:
            best = (ch, ang, float(scores[i]))
    return best


# ---------------------------------------------------------------- 主流程
def read_image(path):
    """用 imdecode 读，路径里有中文也不会挂"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def save_image(path, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        buf.tofile(path)


def detect(img, T, meta, allowed):
    """返回该图里所有牌子：[{digit, angle, score, side, quad, center}, ...]"""
    found = []
    for quad, side in find_candidates(img):
        digit, ang, score = classify(warp_plate(img, quad), T, meta, allowed)
        if digit is None or score < MIN_SCORE:
            continue
        found.append({"digit": digit, "angle": ang, "score": score,
                      "side": side, "quad": quad,
                      "center": (float(quad[:, 0].mean()), float(quad[:, 1].mean()))})
    # 按位置排序：先上后下、再左后右（方便和导航逻辑对应）
    found.sort(key=lambda d: (round(d["center"][1] / 50.0), d["center"][0]))
    return found


def annotate(img, results):
    """在图上画出牌框和结果，供人眼验收"""
    out = img.copy()
    for r in results:
        pts = order_corners_polar(r["quad"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 0, 255), 3)
        cx, cy = int(r["center"][0]), int(r["center"][1])
        label = "%s  %.2f  %ddeg  %dpx" % (r["digit"], r["score"], r["angle"], r["side"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (cx - 4, cy - th - 14), (cx + tw + 6, cy + 6),
                      (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    # 左上角写总览
    if results:
        summary = " + ".join(r["digit"] for r in results)
        cv2.putText(out, "detected: " + summary, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def collect_files(arg):
    """决定要处理哪些图：优先命令行参数，其次 samples/、images/、脚本同目录"""
    here = os.path.dirname(os.path.abspath(__file__))
    if arg:
        if os.path.isdir(arg):
            folder = arg
        elif os.path.isfile(arg):
            return [arg]
        else:
            raise SystemExit("路径不存在：%s" % arg)
    else:
        folder = None
        for name in ("samples", "images", "pics"):
            cand = os.path.join(here, name)
            if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*")):
                folder = cand
                break
        if folder is None:
            folder = here
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                   if p.lower().endswith(exts))
    return files


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    files = collect_files(arg)
    if not files:
        print("没找到图片。用法：python plate_digit.py <图片或文件夹路径>")
        print("或把图片放进脚本同目录的 samples\\ 文件夹里再运行。")
        return

    T, meta, nfont = build_templates()
    print("=" * 78)
    print("模板：%d 个（%d 种字体 × 10 数字 × %d 个朝向）  允许数字：%s  阈值：%.2f"
          % (T.shape[0], nfont if nfont else 1, 360 // ANGLE_STEP,
             ALLOWED_DIGITS, MIN_SCORE))
    print("=" * 78)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if DRAW_OUTPUT:
        os.makedirs(out_dir, exist_ok=True)

    ok, fail = 0, 0
    for i, path in enumerate(files, 1):
        img = read_image(path)
        name = os.path.basename(path)
        if img is None:
            print("[%2d/%2d] %-28s 读不了（格式或路径问题）" % (i, len(files), name))
            fail += 1
            continue
        results = detect(img, T, meta, ALLOWED_DIGITS)
        if results:
            ok += 1
            txt = "  ".join("%s(%.2f,%d°,%dpx)" % (r["digit"], r["score"],
                                                   r["angle"], r["side"])
                            for r in results)
            print("[%2d/%2d] %-28s %s" % (i, len(files), name, txt))
        else:
            fail += 1
            print("[%2d/%2d] %-28s 未检出牌子" % (i, len(files), name))
        if DRAW_OUTPUT:
            save_image(os.path.join(out_dir, os.path.splitext(name)[0] + "_result.jpg"),
                       annotate(img, results))

    print("-" * 78)
    print("共 %d 张：检出 %d 张，未检出 %d 张" % (len(files), ok, fail))
    if DRAW_OUTPUT:
        print("标注图已存到：%s" % out_dir)
        print("（打开 output\\ 里的 *_result.jpg 看框和读数对不对）")


if __name__ == "__main__":
    main()
