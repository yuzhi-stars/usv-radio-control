# 数字识别

用摄像头识别 0 到 9。权重是 `weights/best.pt`，输入 640。

## 安装

需要 Python 3.10 及以上。

```powershell
cd digit-camera
python -m pip install -r requirements.txt
```

## 运行

```powershell
python camera.py
```

默认打开 0 号摄像头，分辨率设为 640×480。换摄像头：

```powershell
python camera.py --camera 1
```

终端在识别结果变化时打印数字和置信度，例如 `6 0.82`。没有目标时打印 `无`。预览窗口点一下，按 Q 退出。
