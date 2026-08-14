# ResNet18测试集错误分析

分析日期：2026-08-14

## 测试集总体结果

- 测试图片：823张
- 正确预测：807张
- 错误预测：16张
- 测试准确率：98.06%
- 测试Macro-F1：0.9790
- 使用检查点：第13轮`best.pt`

该测试集没有参与训练、超参数调整或最佳轮次选择。

## 错误数量分布

真实类别中错误较多的类别：

- `Tectona grandis`：3张
- `Eucalyptus`：2张
- `Lagerstroemia speciosa`：2张
- 其余涉及类别：各1张

被错误预测较多的目标类别：

- `Dalbergia oliveri`：接收4张其他类别图片
- `Erythrina fusca`：接收2张
- `Senna siamea`：接收2张

这解释了`Dalbergia oliveri`召回率为1.000，但精确率只有0.765：它自身13张全部识别正确，同时吸收了4个假阳性。

## 预测置信度

16张错误图片按预测置信度划分：

- 高置信度错误，置信度不低于80%：5张
- 中等置信度错误，60%至80%：7张
- 较低置信度错误，低于60%：4张

最高的三个错误为：

1. `Hopea → Wrightia`：99.32%
2. `Senna siamea → Tamarindus indica`：99.17%
3. `Ficus racemosa → Artocarpus heterophyllus`：93.36%

高置信度错误说明模型不仅存在不确定预测，也可能对某些相似纹理形成了稳定但错误的判断。

## 图片人工检查

检查`misclassified_contact_sheet.jpg`后发现：

1. 16张图片均为有效树皮或茎干近景，没有非目标图片。
2. 没有发现文件损坏、纯黑图片或完全无法辨认的样本。
3. 多组图片具有相似的纵向裂纹、鳞片状纹理、灰褐色外观或苔藓覆盖。
4. 部分图片存在强光、阴影、低对比度、拍摄距离差异或局部模糊。
5. `Lagerstroemia speciosa → Erythrina fusca`样本纹理较模糊，细节不足。
6. `Terminalia catappa → Eucalyptus`样本较平滑、对比度偏低。
7. 多个被预测成`Dalbergia oliveri`的样本均具有细密纵向纹理，说明模型可能过度依赖这一局部特征。
8. 仅凭这些树皮图片无法可靠确认植物学标签是否错误，因此不修改原始标签。

## 混淆模式

16次错误分别属于16个不同类别对，没有一个类别对重复出现。

这说明当前模型没有表现出单一、持续重复的类别对混淆，但存在少量分散的细粒度错误。后续应重点检查错误图片本身，而不是仅针对一个类别对重新训练。

## 当前结论

测试集结果表明ResNet18在BarkVN-50内部划分上表现较好，但仍需保留以下限制：

- BarkVN-50缺少树木个体、地点和拍摄批次元数据；
- 感知哈希只能减少高度相似图片跨子集泄漏，不能完全排除同源图片；
- 数据集内部高分不能证明对其他地区、设备和拍摄条件的泛化能力；
- 后续必须使用独立来源的外部图片测试；
- 不根据测试集错误重新选择模型或调整参数，避免对测试集过拟合。

## 相关文件

- `outputs/resnet18_baseline/test_metrics.json`
- `outputs/resnet18_baseline/test_predictions.csv`
- `outputs/resnet18_baseline/per_class_metrics.csv`
- `outputs/resnet18_baseline/confusion_matrix_normalized.png`
- `outputs/resnet18_baseline/confusion_pairs.csv`
- `outputs/resnet18_baseline/misclassified_contact_sheet.jpg`