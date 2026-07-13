#二维直方图
def hist2d_demo():
    image = cv.imread("D:/images/test1.png")
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0,1], None, [48,48], [0,180,0,256])
    dst = cv.resize(hist,(400,400))
    cv.normalize(dst,dst,0,255,cv.NORM_MINMAX)
    cv.imshow("image",image)
    dst = cv.applyColorMap(np.uint8(dst),cv.COLORMAP_JET)
    cv.imshow("hist",dst)
    plt.imshow(hist,interpolation='nearest')
    plt.title("2D Histogram")
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

#直方图均衡化
def eqhist_demo():
    image = cv.imread("D:/images/test4.png",cv.IMREAD_GRAYSCALE)
    cv.imshow("input", image)
    result = cv.equalizeHist(image)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()