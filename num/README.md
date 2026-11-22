# 数字识别（纯 CV 版 / 零模型）

识别场地上的**黄底黑字数字牌**（本场地编号：2、4、6、7、8）。

和上级 `digit-camera/`（YOLO 版）是**并行的两套方案**，可以互相对拍验证。这一套的特点：

- **不训练、不加载模型**，没有 `.pt` 权重，也没有 torch 依赖
- **整块倒置 180°、任意倾角都能读**（YOLO 那套目前不行，见下面的对拍）
- 内存 66 MB 量级，单帧 720p 约 4.7 ms（本机 CPU）

---

## 和 digit-camera/ 的实测对拍

同一批 17 张现场照片，正立与整块倒置 180° 各测一次：

| 方案 | 正立 | 倒置 180° | 单帧耗时（本机） | 依赖 | 内存 |
|---|---|---|---|---|---|
| **本目录（纯 CV）** | **17/17** | **17/17** | 4.7 ms (720p) | opencv + numpy + pillow | ~66 MB |
| `digit-camera/`（YOLO best.pt） | 16/17 | **4/17** | 约 26 ms | + torch + ultralytics | 数百 MB |

YOLO 倒置时的具体错法（不是随机错，是有规律的）：

```
6 → 9   (三张全是)     4 → 7      7 → 4      2 → 7
```

`6 → 9` 是物理必然：6 的字形转 180° 就是 9。其余是训练集里大概没有倒置样本。
**如果继续用 YOLO，加 180° 旋转增强重训能救回大部分**，但 6/9 那一项仍然只能靠"场地没有 9"这个先验。

---

## 三个脚本

| 文件 | 用途 | 跑在哪 |
|---|---|---|
| `plate_digit.py` | 离线批处理：扫一个文件夹或单张图，输出标注图 | Windows / Linux 都行 |
| `plate_digit_camera.py` | 桌面实时：打开 camera 0，窗口里实时显示结果 | 有屏幕的机器（会自动处理 headless OpenCV） |
| `plate_digit_rdk.py` | **板端**：无窗口，结果打成 JSON，兼容 Python 3.8 | RDK X3 / X5 / 树莓派 |

---

## 安装

Windows 上本机已有的全局 Python 3.14 就够（自带 `cv2 5.0.0 + numpy + pillow`），**不用装东西**。

板子上：

```bash
sudo apt install -y python3-opencv python3-numpy fonts-dejavu-core
```

> `fonts-dejavu-core` **必须装**。实测没有粗体 TTF 时精度会从 100% 掉到 82%（失败的 6 张全是"6 读成 8"）。

---

## 运行

### 1. 离线批处理（最快验证）

```powershell
cd num
python plate_digit.py              # 默认扫 ./samples 里的 4 张
python plate_digit.py "D:\某目录"   # 扫指定文件夹
python plate_digit.py "D:\某张牌.jpg"
```

标注图写到 `output\*_result.jpg`，打开就能看框和读数对不对：

```
[ 1/ 4] p05.jpg    6(0.86,0°,173px)
[ 2/ 4] p08.jpg    7(0.87,0°,159px)
[ 3/ 4] p11.jpg    8(0.90,0°,170px)
[ 4/ 4] p15.jpg    2(0.82,0°,59px)
共 4 张：检出 4 张，未检出 0 张
```

每行格式：`数字(置信度, 牌面旋转角, 牌面像素)`。角度 180° 表示牌子是倒着看到的。

### 2. 桌面实时（摄像头序号可选）

**先查 USB 摄像头是哪个序号**（插着笔记本内置摄像头时，USB 的序号通常排在后面，而且拔插会变）：

```powershell
python plate_digit_camera.py --list
```

输出：

```
扫描摄像头序号 0~5（请求 1280x720 @30 fps）...
序号     后端       实际分辨率        报告fps    画面亮度
0      DSHOW    1280x720     30       134
----------------------------------------------
可用的：--camera 0
用法示例：python plate_digit_camera.py --camera 0
```

然后按序号跑：

```powershell
python plate_digit_camera.py --camera 1                  # 用 1 号（USB 摄像头）
python plate_digit_camera.py --camera 1 --width 640 --height 480
python plate_digit_camera.py --camera 1 --test 60        # 无窗口自检 60 帧
```

窗口里：左上角大字是去抖后的结果，绿框=可信、橙框=低置信度，右上角小图是分类器实际看到的牌面。

按键：`q`/`ESC` 退出，`s` 存快照，`p` 切换牌面小图。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--camera N` | 0 | 摄像头序号。**插 USB 摄像头后先跑 `--list`** |
| `--width` / `--height` | 1280 / 720 | 请求分辨率。USB2.0 上 720p 比 1080p 稳 |
| `--fps` | 30 | 请求帧率。**不设这个某些相机会掉到 5.7 fps** |
| `--list` | — | 扫描 0~5 号，列出真正能出图的摄像头后退出 |
| `--test N` | 0 | 无窗口跑 N 帧自检后退出（远程/无桌面时用） |

> 说明：脚本顶部的 `CAMERA_INDEX` 只是**默认值**，命令行 `--camera` 优先。
> 分辨率/帧率同理，`FRAME_W`/`FRAME_H`/`TARGET_FPS` 是默认值。

### 3. 板端（RDK）

```bash
python3 plate_digit_rdk.py                 # 默认 /dev/video0
python3 plate_digit_rdk.py --bench 120     # 先测这块板能跑多少
python3 plate_digit_rdk.py --width 640 --height 480
python3 plate_digit_rdk.py --video 场地.mp4 # 用录像调，不用相机
```

输出一行一个 JSON，日志走 stderr，所以可以直接接管道：

```json
{"ts": 1795319866.351, "digit": "2", "fps": 73.5, "score": 0.816, "angle": 0, "plate_px": 59}
{"ts": 1795319870.12, "digit": null, "event": "lost"}
```

---

## 要调的东西

三个脚本顶部都有 CONFIG 区，最常动的四个：

| 参数 | 默认 | 说明 |
|---|---|---|
| `ALLOWED_DIGITS` | `"24678"` | 场地数字集合。**别加 9**：加了之后倒置的 6 会变成歧义 |
| `MIN_SCORE` | `0.65` | 置信度门限。正常读数 0.80~0.92；调高更严，调低更容易接受弱目标 |
| `SMOOTH_N` | `8` | 时间平滑窗口（帧）。调大更稳但响应变慢 |
| `ANGLE_STEP` | `15` | 旋转搜索步长。板上嫌慢可以改 45（模板变成 1/3，匹配开销也降到 1/3） |
| `CAMERA_INDEX` / `FRAME_W` / `FRAME_H` / `TARGET_FPS` | 0 / 1280 / 720 / 30 | 只是默认值，命令行 `--camera` 等参数优先 |

---

## 实测数据

| 项 | 值 |
|---|---|
| 正立 / 倒置 180° / 任意角度（24 组） | 17/17 / 17/17 / 最差 16/17 |
| 留出验证（前 9 张调参，后 8 张只测） | 8/8 |
| 单帧耗时 1280×720 | 4.7 ms（纯处理，不含显示） |
| 单帧耗时 640×480 | 2.4 ms |
| 内存 常驻 / 峰值 | 66 / 75 MB |
| 模板矩阵 | 1920 个 28×28 float32 ≈ 6.0 MB |

板子上的帧率**必须实测**（我没有 RDK 硬件）。RDK X3 是 4 核 A53 @1.5 GHz，按单核性能约为本机 1/4~1/6 估算：720p 约 15~30 ms、640×480 约 5~10 ms —— 这个估算可能差 2 倍，用 `--bench` 看真值。

---

## 已知限制（都是实测出来的，不是猜的）

1. **6 和 9 在 180° 倒置下物理不可分**。同一张图（真实的 6，倒置输入）：限制数字集合 1~8 时读 `6 @180° score 0.857`；放开到 0~9 时读 `9 @0° score 0.858`。置信度只差 0.001 —— 任何字形方法都救不了，只能靠"场地没有 9"这个先验，或者给牌子加一个不对称角标。
2. **字体缺失精度掉到 82%**（见安装那节）。
3. **颜色阈值和场地相关**。这套用 HSV 找黄牌，`S>=130` 是在米色大理石地面上调出来的；换深色地面/强白光得重调。调低到 `S=90` 时米色地面会被当成牌子，检测直接崩。
4. **MIPI 摄像头不能直接用 `cv2.VideoCapture`**。USB 摄像头没问题；RDK 的 MIPI 摄像头走 `hobot_mipi_cam` 那套节点、输出 NV12，需要订阅 ROS2 image 话题转 BGR 再喂给 `detect()`，这一步还没做。

---

## 验证过的 / 没验证过的

**验证过**：17 张现场静态图（正立、倒置、24 个旋转角、7 档分辨率）、内存与耗时、无字体回退精度、中文路径下读写。

**没验证过**：真实航行/飞行中的反光与抖动、RDK 上的实际帧率、MIPI 相机链路、夜间或强逆光。

---

## 补充说明

- `digit-camera/camera.py` 里用的是 `cv2.imshow`，需要 **GUI 版** OpenCV。项目 `.venv` 里是 `opencv-python 5.1`（有 GUI），所以直接跑没问题；但用全局 Python（装的是 `opencv-python-headless`，`GUI: NONE`）会报 `The function is not implemented`。本目录的 `plate_digit_camera.py` 已经做了自动回退（退到 tkinter），两种环境都能显示。
- 详细调研（GitHub 方案对比、v1→v4 的演进过程、6/9 歧义实证数据）在：
  `F:\Projects\dsh\digit-plate-recognition\README.md`
- `samples/` 里的 4 张图只是让你开箱验证用的，完整 17 张测试集在上面的目录里。

---

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-22 | 建立本目录：纯 CV 数字识别（离线 / 桌面实时 / 板端三套脚本）+ 与 YOLO 版的对拍数据 |
