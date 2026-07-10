#图像放缩与插值
def resize_demo():
    image = cv.imread("D:/images/test5.png")
    h,w,c = image.shape
    cv.namedWindow("resize", cv.WINDOW_AUTOSIZE)
    dst = cv.resize(image, (0,0), fx=0.75, fy=0.75, interpolation=cv.INTER_NEAREST)
    # dst = cv.resize(image, (w//2, h//2), interpolation=cv.INTER_NEAREST)
    cv.imshow("resize", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

#图像翻转
def flip_demo():
    image = cv.imread("D:/images/test5.png")
    cv.imshow("input", image)
    cv.namedWindow("flip", cv.WINDOW_AUTOSIZE)
    dst = cv.flip(image,1) #flipCode=0,表示上下翻转，=1表示左右翻转，=-1表示对角线翻转
    cv.imshow("flip", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()