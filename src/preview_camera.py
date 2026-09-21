import cv2

window_name = "Camera preview"
camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

try:
    if not camera.isOpened():
        raise RuntimeError("无法打开摄像头")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    print("预览已启动：点击画面窗口，按 Q 或 Esc 退出。")

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("读取摄像头画面失败")

        cv2.imshow(window_name, frame)

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
    print("摄像头已释放，预览窗口已关闭。")
