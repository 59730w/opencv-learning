# 多尺度目标检测
import torch
import matplotlib
from d2l import torch as d2l
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
# 设置中文字体（Windows系统）
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)


if __name__ == "__main__":
    img = d2l.plt.imread(r"D:\DL_code\d2l-zh\pytorch\img\catdog.jpg")
    h, w = img.shape[:2]
    print(h, w)

    display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
    plt.show()

    display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
    plt.show()

    display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
    plt.show()