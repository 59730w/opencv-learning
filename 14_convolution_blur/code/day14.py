#图像卷积操作
def blur_demo():
    image = cv.imread("D:/images/test5.png")
    cv.imshow("input", image)
    result = cv.blur(image,(15,15))
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

#高斯模糊
def conv_demo():
    image = cv.imread("D:/images/test5.png")
    cv.imshow("input", image)
    result = cv.GaussianBlur(image,(0,0),15)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()