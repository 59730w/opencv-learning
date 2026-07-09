#鼠标操作与响应
# b1 = np.zeros((512,512,3), dtype=np.uint8)
b1 = cv.imread("D:/images/test1.png")
img = np.copy(b1)
x1 = -1
x2 = -1
y1 = -1
y2 = -1
def mouse_drawing(event,x,y,flags,param):
    global x1,x2,y1,y2
    if event == cv.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
    if event == cv.EVENT_MOUSEMOVE:
        if x1 < 0 or y1 < 0:
            return
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        if dx > 0 and dy > 0:
            b1[:,:,:] = img[:,:,:]
            cv.rectangle(b1,(x1,y1),(x2,y2),(0,0,255),2,8,0)
    if event == cv.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        if dx > 0 and dy > 0:
            b1[:, :, :] = img[:,:,:]
            cv.rectangle(b1, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)
        x1 = -1
        x2 = -1
        y1 = -1
        y2 = -1
def mouse_demo():
    cv.namedWindow("mouse_demo", cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("mouse_demo", mouse_drawing)
    while True:
        cv.imshow("mouse_demo", b1)
        c = cv.waitKey(10)
        if c == 27:
            break
    cv.destroyAllWindows()

#图像像素类型转换与归一化
def norm_demo():
    image = cv.imread("D:/images/test5.png")
    cv.namedWindow("norm_demo", cv.WINDOW_AUTOSIZE)
    result = np.zeros_like(np.float32(image))
    cv.normalize(np.float32(image),result,0,1,cv.NORM_MINMAX,dtype=cv.CV_32F)
    cv.imshow("norm_demo", np.uint8(result*255))
    cv.waitKey(0)
    cv.destroyAllWindows()