"""使用本次训练的最佳权重进行本机摄像头推理。"""
import argparse
from pathlib import Path
from time import perf_counter
from threading import Condition, Event, Thread

import cv2
from ultralytics import YOLO


class LatestFrame:
    """持续采集，只保留最新一帧，避免推理时积压旧画面。"""
    def __init__(self, camera):
        self.camera = camera
        self.condition = Condition()
        self.stop = Event()
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.capture_fps = 0.0
        self.error = None
        self.thread = Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        previous = perf_counter()
        try:
            while not self.stop.is_set():
                ok, frame = self.camera.read()
                now = perf_counter()
                if not ok or frame is None:
                    raise RuntimeError("读取摄像头画面失败")
                instantaneous = 1 / max(now - previous, 1e-6)
                previous = now
                with self.condition:
                    self.capture_fps = (instantaneous if self.sequence == 0 else
                                        0.9 * self.capture_fps + 0.1 * instantaneous)
                    self.frame = frame
                    self.timestamp = now
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def latest(self, after):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > after or self.error is not None, timeout=5)
            if self.error:
                raise self.error
            if not ready:
                raise RuntimeError("摄像头连续 5 秒没有新画面")
            return self.sequence, self.frame, self.timestamp, self.capture_fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", choices=("MJPG", "YUYV"), default="MJPG")
    parser.add_argument("--preview-only", action="store_true", help="只测采集与显示，不运行模型")
    args = parser.parse_args()

    weights = Path(__file__).resolve().parents[1] / "runs/yolo26x_1280_b2_quality_v2/weights/best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"找不到训练权重：{weights}")
    model = None if args.preview_only else YOLO(str(weights))
    window = "Volleyball YOLO26x"
    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    reader = None
    try:
        if not camera.isOpened():
            raise RuntimeError(f"无法打开 {args.camera}，请检查设备映射、权限及摄像头占用。")
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FPS, args.fps)
        # 驱动保留多个采集缓冲；LatestFrame 仍只向主循环提供最新帧。
        buffer_ok = camera.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        print(
            f"采集缓冲：请求 4，设置成功={buffer_ok}，"
            f"读回值={camera.get(cv2.CAP_PROP_BUFFERSIZE):.0f}",
            flush=True,
        )
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 720)
        code = int(camera.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((code >> (8 * i)) & 255) for i in range(4))
        print(f"协商采集格式：{codec}；驱动报告 FPS：{camera.get(cv2.CAP_PROP_FPS):.1f}", flush=True)
        if model is not None:
            print(f"模型：{weights}\n类别：{model.names}", flush=True)
        else:
            print("纯摄像头预览模式：不运行模型。", flush=True)
        print("点击画面窗口，按 Q 或 Esc 退出。FPS 为采集、推理和显示循环的处理速度，不代表摄像头端到端延迟。", flush=True)
        fps = 0.0
        previous = perf_counter()
        last_report = previous
        reader = LatestFrame(camera)
        sequence = 0
        while True:
            sequence, frame, captured_at, capture_fps = reader.latest(sequence)
            if fps == 0:
                height, width = frame.shape[:2]
                print(f"请求采集：{args.width}×{args.height}；实际采集：{width}×{height}", flush=True)
                if (width, height) != (args.width, args.height):
                    print("摄像头未采用请求分辨率，当前按实际分辨率继续。", flush=True)
            started = perf_counter()
            if model is not None:
                result = model.predict(
                    source=frame, imgsz=args.imgsz, conf=args.conf,
                    device=0, quantize=16, save=False, verbose=False,
                )[0]
                predict_ms = (perf_counter() - started) * 1000
                annotated = result.plot(line_width=2)
            else:
                predict_ms = 0.0
                annotated = frame.copy()
            # 只缩小显示画面，不影响模型接收到的原始分辨率。
            height, width = annotated.shape[:2]
            annotated = cv2.resize(annotated, (960, round(height * 960 / width)))
            age_ms = (perf_counter() - captured_at) * 1000
            cv2.putText(annotated, f"Camera: {capture_fps:.1f} | Loop: {fps:.1f} | Predict: {predict_ms:.0f} ms",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            now = perf_counter()
            current_fps = 1.0 / max(now - previous, 1e-6)
            fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
            previous = now
            if now - last_report >= 3:
                print(f"采集 {capture_fps:.1f} FPS | 处理 {fps:.1f} FPS | 推理调用 {predict_ms:.1f} ms | 读帧后至显示提交前 {age_ms:.1f} ms", flush=True)
                last_report = now
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error as exc:
                # 窗口已销毁时正常退出，其他异常继续报告。
                if "NULL guiReceiver" in str(exc):
                    break
                raise
    except KeyboardInterrupt:
        print("收到终端中断，正在退出。")
    finally:
        if reader is not None:
            reader.stop.set()
            reader.thread.join(timeout=2)
        camera.release()
        cv2.destroyAllWindows()
        print("摄像头已释放，检测已结束。")


if __name__ == "__main__":
    main()
