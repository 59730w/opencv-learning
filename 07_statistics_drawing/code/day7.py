#图像像素值统计
def pixel_stat_demo():
    b1 = cv.imread("D:/images/test4.png")
    print(b1.shape)
    cv.imshow("input", b1)
    print(np.max(b1[:, :, 0])) #统计像素最大值
    # print(np.min(b1[:,:,2])) #统计像素最小值
    means, dev = cv.meanStdDev(b1) #统计图片像素均值和方差
    print(means, dev)
    cv.waitKey(0)
    cv.destroyAllWindows()

#图像几何形状绘制
def drawing_demo():
    # b1 = np.zeros((512,512,3), dtype=np.uint8)
    b1 = cv.imread("D:/images/test2.png")
    cv.rectangle(b1,(280,100),(480,320),(0,0,255),2,8,0) #绘制矩形
    # cv.circle(b1,(200,200),100,(255,0,0),2,8,0) #绘制圆形
    # cv.line(b1,(100,100),(300,300),(0,255,0),2,8,0) #绘制直线
    cv.putText(b1,"99% face",(280,100),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,8) #输入文本
    cv.imshow("input", b1)
    cv.waitKey(0)
    cv.destroyAllWindows()