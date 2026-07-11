#图像旋转
def rotate_demo():
    src = cv.imread("D:/images/test1.png")
    cv.imshow("input", src)
    h, w, c= src.shape
    angle = 45   # 可改成任意角度

    # 1. 计算旋转后外接矩形的尺寸
    rad = np.radians(angle)
    cos_abs = abs(np.cos(rad))
    sin_abs = abs(np.sin(rad))
    new_w = int(w * cos_abs + h * sin_abs)
    new_h = int(w * sin_abs + h * cos_abs)

    # 2. 获取旋转矩阵（绕图像中心，角度制）
    M = cv.getRotationMatrix2D((w/2, h/2), angle, 1.0)

    # 3. 调整平移量，使旋转后的图像位于新画布中心
    M[0, 2] += (new_w / 2 - w / 2)
    M[1, 2] += (new_h / 2 - h / 2)

    # 4. 使用新尺寸执行仿射变换
    dst = cv.warpAffine(src, M, (new_w, new_h))

    cv.imshow("rotate-center-demo", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

#视频文件/摄像头使用
def video_demo():
    # cap = cv.VideoCapture(0)
    cap = cv.VideoCapture("D:/TV sucai/01.mp4")
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    print(w,h,fps)
    while True:
        ret, frame = cap.read()
        # frame = cv.flip(frame,1) #读视频文件时不需要这行代码
        if ret is not True:
            break
        cv.imshow("frame", frame)
        c = cv.waitKey(10)
        if c == 27:
            break
    cv.destroyAllWindows()