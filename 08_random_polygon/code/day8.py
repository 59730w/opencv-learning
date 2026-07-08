#随机数与随机颜色
def random_color_demo():
    b1 = np.zeros((512,512,3), dtype=np.uint8)
    while True:
        xx = np.random.randint(0,512,2,dtype=np.int32)
        yy = np.random.randint(0, 512, 2, dtype=np.int32)
        bgr = np.random.randint(0, 255, 3, dtype=np.uint8)
        cv.line(b1,(xx[0],yy[0]),(xx[1],yy[1]),(int(bgr[0]),int(bgr[1]),int(bgr[2])),1,8,0) #绘制直线
        cv.imshow("input", b1)
        c = cv.waitKey(10)
        if c == 27:
            break
    cv.destroyAllWindows()

#多边形填充与绘制
def polyline_drawing_demo():
    canvas = np.zeros((512,512,3), dtype=np.uint8)
    pts = np.array([[100,100],[350,100],[450,280],[320,450],[80,400]],dtype=np.int32)
    # cv.fillPoly(canvas,[pts],(255,0,255),8,0)
    # cv.polylines(canvas,[pts],True,(0,0,255),2,8,0)
    cv.drawContours(canvas,[pts],-1,(255,0,0),-1) #集合了上面两个api的优点，最后的-1表示填充；最后的1表示绘制
    cv.imshow("polyline", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()