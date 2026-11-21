"""打开摄像头，加载数字模型，终端打印识别到的数字。

  python camera.py
  python camera.py --camera 0

预览窗点一下后按 Q 退出。
"""
import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"


def open_camera(index):
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if not WEIGHTS.is_file():
        raise SystemExit(f"找不到权重 {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"打不开相机 {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Q 退出")
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
        digits = []
        if result.boxes is not None:
            for cls, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                digits.append((int(cls), float(score)))
        digits.sort(key=lambda item: -item[1])
        text = "  ".join(f"{digit} {score:.2f}" for digit, score in digits)
        if text != last:
            print(text if text else "无")
            last = text
        cv2.imshow("digit", result.plot())
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
