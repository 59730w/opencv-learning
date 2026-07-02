#图像对象的创建与赋值
def mat_demo():
    image = cv.imread("D:/images/test.jpg") #opencv读取时默认是BGR，BGR的通道为3，即0~255
    # image = cv.imread("D:/images/test.jpg", cv.IMREAD_GRAYSCALE) #读取图像时并转为gray，gray的通道为1可以不显示
    print(image.shape) #输出图像的尺寸：高h、宽w、通道c
    print(image)
    roi = image[60:500, 60:500, :] #若image是gray，则最后的 ：需要删除
    blank = np.zeros_like(image) #图像对象的创建
    blank[60:500, 60:500, :] = image[60:500, 60:500, :] #图像对象的赋值
    # blank = np.copy(image) #image拷贝给blank
    cv.imshow("blank", blank)
    cv.imshow("roi", roi)
    cv.waitKey(0)
    cv.destroyAllWindows()

#图像像素的读写操作
def pixel_demo():
    image = cv.imread("D:/images/test.jpg")
    cv.imshow("input", image)
    h,w,c =image.shape #高h代表有多少行像素，宽w代表有多少列像素
    for row in range(h):
        for col in range(w):
            b,g,r = image[row, col] #图像像素的读操作
            image[row, col] = (255-b, 255-g, 255-r) #图像像素的写操作，这里以取反为例
    cv.imshow("result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()