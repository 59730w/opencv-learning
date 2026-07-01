import cv2 as cv

#图像读取与显示
def read_demo():
    image = cv.imread("D:/images/test.jpg")
    cv.imshow("input",image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#图像色彩空间的转换
def color_space_demo():
    image = cv.imread("D:/images/test.jpg") #opencv读取时默认是BGR
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #BGR转成gray
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV) #BGR转成hsv
    cv.imshow("gray", gray)
    cv.imshow("hsv", hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    color_space_demo()