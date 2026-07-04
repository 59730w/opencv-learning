#TrackBar-参数传递与调整亮度与对比度
def adjust_contrast_demo():
    image = cv.imread("D:/images/test.jpg")
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input", 0, 100, nothing)
    cv.createTrackbar("contrast", "input", 100, 200, nothing)
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        lightness = cv.getTrackbarPos("lightness", "input")       #亮度
        contrast = cv.getTrackbarPos("contrast", "input") / 100   #对比度
        print("lightness:", lightness, "contrast:", contrast)
        result = cv.addWeighted(image, contrast, blank, 0.5, lightness)
        cv.imshow("result", result)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()

#键盘响应操作
def keys_demo():
    image = cv.imread("D:/images/test.jpg")
    cv.imshow("input", image)
    while True:
        c = cv.waitKey(1)
        #ASCII中，数字0~9的ASCII值为48~57，大写字母A~Z的ASCII值为65~90，小写字母a~z的ASCII值为97~122
        if c == 49: #1,键盘按1
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cv.imshow("result", gray)
        if c == 50: #2，键盘按2
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            cv.imshow("result", hsv)
        if c == 51: #3，键盘按3
            opposite = cv.bitwise_not(image)
            cv.imshow("result", opposite)
        if c == 27: #ESC，键盘按ESC，表示退出
            break
    cv.destroyAllWindows()