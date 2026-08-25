# Day53 示例资产

本目录只保留 ONNX 推理一致性检查需要的最小资产：

| 文件 | 原始类别 | 原始路径 |
| --- | --- | --- |
| `acacia_IMG_6348.JPG` | Acacia | `BarkVN-50_mendeley/Acacia/IMG_6348.JPG` |
| `adenanthera_microsperma_IMG_5777.JPG` | Adenanthera microsperma | `BarkVN-50_mendeley/Adenanthera microsperma/IMG_5777.JPG` |
| `cananga_odorata_IMG_5483.JPG` | Cananga odorata | `BarkVN-50_mendeley/Cananga odorata/IMG_5483.JPG` |
| `class_to_idx.json` | 50 类索引映射 | 森林树种项目 `datasets/processed/class_to_idx.json` |

图片来自 BarkVN-50 v1 数据集，每张尺寸均为 303×404。这里只把它们当作同源回归样本，用于确认 PyTorch 和 ONNX Runtime 对相同输入产生一致输出，不把结果当成外部泛化证据。
