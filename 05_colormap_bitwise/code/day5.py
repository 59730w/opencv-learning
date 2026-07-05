#OpenCV自带颜色表操作
def color_table_demo():
    colormap = [
        cv.COLORMAP_AUTUMN,
        cv.COLORMAP_BONE,
        cv.COLORMAP_JET,
        cv.COLORMAP_WINTER,
        cv.COLORMAP_RAINBOW,
        cv.COLORMAP_OCEAN,
        cv.COLORMAP_SUMMER,
        cv.COLORMAP_SPRING,
        cv.COLORMAP_COOL,
        cv.COLORMAP_PINK,
        cv.COLORMAP_HOT,
        cv.COLORMAP_PARULA,
        cv.COLORMAP_MAGMA,
        cv.COLORMAP_INFERNO,
        cv.COLORMAP_PLASMA,
        cv.COLORMAP_VIRIDIS,
        cv.COLORMAP_CIVIDIS,
        cv.COLORMAP_TWILIGHT,
        cv.COLORMAP_TWILIGHT_SHIFTED,
    ]
    image = cv.imread("D:/images/test1.png")
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", image)
    index = 0
    while True:
        dst = cv.applyColorMap(image, colormap[index%19])
        index += 1
        cv.imshow("color style", dst)
        c = cv.waitKey(500)
        if c == 27: #ESC，键盘按ESC，表示退出
            break
    cv.destroyAllWindows()

#图像像素的逻辑操作
def bitwise_demo():
    b1 = np.zeros((400,400,3), dtype=np.uint8)
    b1[:, :] =(255, 0, 255)
    b2 = np.zeros((400, 400, 3), dtype=np.uint8)
    b2[:, :] = (0, 255, 255)
    cv.imshow("b1", b1)
    cv.imshow("b2", b2)
    dst1 = cv.bitwise_and(b1, b2)
    dst2 = cv.bitwise_or(b1, b2)
    cv.imshow("bitwise_and", dst1)
    cv.imshow("bitwise_or", dst2)
    cv.waitKey(0)
    cv.destroyAllWindows()