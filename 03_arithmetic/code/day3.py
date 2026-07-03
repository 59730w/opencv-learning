#图像像素的算术操作
def math_demo():
    image = cv.imread("D:/images/test.jpg")
    cv.imshow("input", image)
    # h,w,c =image.shape #高h代表有多少行像素，宽w代表有多少列像素
    blank = np.zeros_like(image)
    blank[:, :] = (2, 2, 2) #图像像素的赋值
    # result = cv.add(image, blank) #图像像素加法操作
    # result = cv.subtract(image, blank) #图像像素减法操作
    # result = cv.multiply(image, blank) #图像像素乘法操作
    result = cv.divide(image, blank) #图像像素除法操作
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

#TrackBar-动态调整图像亮度
def nothing(x):
    print(x)

def adjust_lightness_demo():
    image = cv.imread("D:/images/test.jpg")
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input", 0, 100, nothing)
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        pos = cv.getTrackbarPos("lightness", "input")
        blank[:, :] = (pos, pos, pos)
        result = cv.add(image, blank)
        cv.imshow("result", result)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()