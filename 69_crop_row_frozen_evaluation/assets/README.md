# Day69 本地产物

冻结视频、逐帧记录、外部数据与联系表保存在：

`D:/DL_code/data/crop_row_perception/day69_frozen_evaluation/`

该目录包含 6 段叠加视频、896 条视频帧记录、429 条 CRDLD 记录、1,760 条
RowDetr 记录和五张视觉审计联系表。由于原始数据许可边界尚未解除，这些大文件不进入 Git；
仓库只保存可复现代码、冻结协议、结果摘要和笔记。

第二版新增 `day69_v2_ssr_external_audit.jpg`：12 个冻结 miss 与 8 个高方向误差 match，紫色为
SSR 中央行参考、绿色为匹配预测、青色为其他预测。SSR v7 `expansion_2` 的 98 对原始图像/标签、
逐图 JSONL、哈希清单与 COMPLETE 访问台账保存在
`D:/DL_code/data/crop_row_perception/day69_v2_ssr_external/`，原始第三方数据不进入 Git。

`heji_web_ui.png` 是本地“禾迹”可视化网页的实际浏览器渲染截图，用于验证上传、设备选择、
结果预览和下载区域的最终布局；截图不包含第三方视频帧。
