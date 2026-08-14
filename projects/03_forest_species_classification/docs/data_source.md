# BarkVN-50 数据来源记录

## 基本信息

- 数据集名称：BarkVN-50
- 版本：Version 1
- 发布日期：2020-02-16
- 作者：Vinh Truong Hoang
- 发布机构：Ho Chi Minh City Open University
- 任务类型：树皮图像树种分类
- 类别数量：50
- 图片数量：5,578
- 图片分辨率：303 × 404
- 许可协议：CC BY 4.0

## 官方来源

- 官方页面：https://data.mendeley.com/datasets/gbt4tdmttn/1
- DOI：https://doi.org/10.17632/gbt4tdmttn.1

## 建议引用

Truong Hoang, Vinh (2020), “BarkVN-50”,<br>
Mendeley Data, Version 1.<br>
DOI: 10.17632/gbt4tdmttn.1

## 本地存放位置

原始数据下载并解压到：

`datasets/raw/BarkVN-50`

原始图片不得修改、重命名或直接删除。清洗、划分和生成的文件统一放入：

`datasets/processed`

## 数据使用范围

本项目将数据用于非商业学习、机器视觉实验和项目展示。公开项目必须注明作者、数据集名称、版本、DOI和CC BY 4.0许可。

数据集本体不上传至GitHub。

## 数据划分风险

官方页面没有明确提供每张图片对应的树木个体、采集地点或拍摄批次编号，因此不能直接证明随机图片划分不存在同源泄漏。

正式划分前必须：

1. 检查无法读取的图片；
2. 检查完全重复图片；
3. 检查近似重复及疑似连拍图片；
4. 尽可能将相似图片放入同一个数据子集；
5. 在最终报告中保留数据泄漏风险说明。

## 下载记录

- 下载日期：2026-08-13
- 下载方式：Mendeley Data 官方页面 `Download All`
- 外层下载文件名：`gbt4tdmttn-1.zip`（验证内部 RAR 后已删除）
- 外层 ZIP 大小：184,042,761 字节
- 外层 ZIP SHA-256：`AB45C51926ACFCB1DFD485E1F404AF7063F7673BD2505CB066CB5742219EA9A3`
- 原始归档：`datasets/raw/BarkVN-50/v1/BarkVN-50.rar`
- RAR 大小：184,044,839 字节
- RAR SHA-256：`607F3E53EE474FFCEB3FFA25A23FB5ED08C955C971D5D89153164B242AA67018`
- RAR 完整性：7-Zip 测试通过，`Everything is Ok`
- 解压目录：`datasets/raw/BarkVN-50/v1/images/BarkVN-50_mendeley`
- 解压后类别数量：50
- 解压后图片数量：5,578 张 JPG
- 解压后图片总大小：183,888,470 字节
- OpenCV 解码检查：5,578 张全部成功，均为 3 通道
- 图片尺寸：4,414 张为 303×404，1,159 张为 245×327，另有 5 张特殊尺寸待数据审计
- 每类图片数量范围：80–239 张
