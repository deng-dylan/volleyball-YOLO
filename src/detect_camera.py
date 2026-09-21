import cv2
from ultralytics import YOLO

# 加载本次训练得到的最佳模型，只加载一次。
model = YOLO(
    "/workspace/runs/volleyball_v8n_baseline_v1/weights/best.pt"
)

window_name = "Volleyball detection"
camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

try:
    if not camera.isOpened():
        raise RuntimeError("无法打开摄像头")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print("排球检测已启动：点击画面窗口，按 Q 或 Esc 退出。")

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("读取摄像头画面失败")

        # 用 GPU 对当前这一帧执行检测。
        result = model.predict(
            source=frame,
            imgsz=960,
            conf=0.25,
            device=0,
            save=False,
            verbose=False,
        )[0]

        # 将检测框、类别和置信度画到画面上。
        annotated_frame = result.plot(line_width=2)
        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("收到终端中断，正在退出。")

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("摄像头已释放，检测已结束。")
