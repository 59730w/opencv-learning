import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('TkAgg')
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#视频处理与保存
def video_save_demo():
    # cap = cv.VideoCapture(0)
    cap = cv.VideoCapture("D:/TV sucai/01.mp4")
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter.fourcc(*'mp4v')
    out = cv.VideoWriter("D:/images/test6.mp4", fourcc, fps, (int(w), int(h)), True)
    print(w,h,fps)
    while True:
        ret, frame = cap.read()
        # frame = cv.flip(frame,1) #读视频文件时不需要这行代码
        if ret is not True:
            break
        cv.imshow("frame", frame)
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        cv.imshow("result",hsv)
        out.write(hsv)
        c = cv.waitKey(10)
        if c == 27:
            break
    out.release()
    cap.release()
    cv.destroyAllWindows()

#图像直方图
def image_hist():
    image = cv.imread("D:/images/test2.png")
    cv.imshow("input",image)
    color = ('blue','green','red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image],[i],None,[256],[0,256])
        print(hist)
        plt.plot(hist,color=color)
        plt.xlim([0,256])
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()