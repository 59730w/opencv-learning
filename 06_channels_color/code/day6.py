#通道分离与合并
def channels_split_merge_demo():
    b1 = cv.imread("D:/images/test1.png")
    print(b1.shape)
    cv.imshow("input", b1)
    cv.imshow("b1", b1[:,:,2])
    mv = cv.split(b1)
    mv[0][:,:] = 255
    result = cv.merge(mv)
    cv.imshow("result", result)
    dst = np.zeros(b1.shape, dtype=np.uint8)
    cv.mixChannels([b1],[dst],fromTo=[2,0,1,1,0,2])
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

#图像色彩空间转换
def color_space_demo():
    b1 = cv.imread("D:/images/test3.png")
    print(b1.shape)
    cv.imshow("input", b1)
    hsv = cv.cvtColor(b1, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    mask = cv.inRange(hsv,(0,0,221),(180,30,255))
    cv.bitwise_not(mask,mask)
    result = cv.bitwise_and(b1,b1,mask=mask)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()