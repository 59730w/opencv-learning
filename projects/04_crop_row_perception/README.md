# Practical 04：农业机器人作物行视觉感知

这是一个 Day59-Day70 的短周期学习项目，目标是把已有 OpenCV、深度学习、C++和实验验证能力迁移到农业机器人前视机器视觉问题。

当前只研究离线视觉感知：从单目RGB图像或视频帧检测所有满足可见性规则的作物行，选择
相机中心左右最近的可靠作物行作为当前走廊边界，再估计图像平面走廊中心、方向、消失点、
置信度和拒绝状态。作物行本身不是可通行中心；项目不包含真实机器人控制，也不把图像坐标
误写成物理距离或真实机器人车体边界。

CRDLD 与 RowDetr 已完成下载和实测：CRDLD 可作受限 ID 开发候选，RowDetr 高粱子源中 1,760 张干净图像已冻结为外部正样本。由于 CRDLD 许可/最高分组仍未知，且前视目标域拒识负样本尚缺，完整数据门禁仍为 `BLOCKED`。

## 门禁状态

| 门禁 | 状态 | 日期 | 证据 |
|---|---|---|---|
| 目标契约 | PASS | 2026-08-31 | `target_contract.yaml` |
| 数据可行性 | SCOPED PASS / FULL BLOCKED | 2026-09-02 | 正样本学习可继续；许可、最高分组与拒识负样本仍阻断完整声明 |
| 环境 | NOT_AVAILABLE | — | 前两门通过后执行 |
| 管线试运行 | NOT_AVAILABLE | — | 后续执行 |
| 内部有效性 | NOT_AVAILABLE | — | 后续执行 |
| 基线/OOD开发 | NOT_AVAILABLE | — | 后续执行 |
| 冻结外部测试 | NOT_AVAILABLE | — | 后续执行 |
| 交付 | NOT_AVAILABLE | — | Day70 执行 |

只有显式 `PASS` 才能进入下一门。开源项目展示和论文指标均不算本项目效果证据。

Day60 的审查工作已经完成，但数据门禁为 `BLOCKED`。这表示当天学习已完成，不表示项目可以继续进入环境或管线开发。

## Day59 产物

- `target_contract.yaml`：任务、坐标、输出、条件、指标和声明边界；
- `evidence_registry.yaml`：开发、OOD与冻结外部证据角色；
- `docs/open_source_baseline_review.md`：七个开源候选的适用性与许可证边界；
- `../../59_crop_row_perception_contract/code/day59_geometry.py`：坐标与误差定义的最小可执行参考；
- `../../59_crop_row_perception_contract/code/day59_notes.md`：完整中文学习笔记。
- 2026-09-05原位修订合同为schema v2：输出从单中央参考行扩展为可变数量作物行、左右相邻
  边界、走廊中心、多线消失点和图像行距；保留原始范围变更记录，新增多行Precision/Recall、
  边界成对正确率和走廊中心误差，不把图像结果称为真实机器人安全边界。

## Day60 产物

- `data_viability_report.md`：候选数据、实下载检查、阻断项与解除条件；
- `data/source_registry.json`：五个候选的机器可读来源登记；
- `data/audit_result.json`：49.7 MB 试审包的配对、解码、尺寸、标签值和重复检查；
- `data/downloaded_sources_audit.json`：CRDLD 与 RowDetr 下载后的完整实审；
- `data/frozen_external_sorghum_manifest.json`：1,760 张只读外部正样本冻结清单；
- `data/build_frozen_sorghum_manifest.py`：清单生成与坏标签排除；
- `data/sample_remote_zip_camera.py`：无需下载完整大包的远程 ZIP 相机抽样工具；
- `../../60_crop_row_data_viability/code/day60_data_audit.py`：可重跑的数据审查器；
- `../../60_crop_row_data_viability/code/day60_notes.md`：完整中文学习笔记。
- 2026-09-05补充多行标签审计：CRDLD全部1,930张标签可解码且均含多行信号，但所有中心线
  合并在二值JPEG掩膜中，没有实例ID；允许固定水平带交点与顺序匹配学习，正式实例身份、
  漏行农业语义及完整数据门禁仍未通过。

## Day61 产物

- `day61_scope_decision.yaml`：完整门禁继续阻断、受限正样本学习可继续的双层范围决定；
- `data/scoped_crdld/`：排除3张精确重复后的训练、验证和同源内部基准清单；
- `../../61_crop_row_color_illumination/code/day61_color_illumination.py`：有界Gray-World、HSV/Lab/固定ExG、互斥开发分区、清单验证与鲁棒性评分；
- `../../61_crop_row_color_illumination/tests/test_day61_color_illumination.py`：颜色处理与代理指标测试；
- `../../61_crop_row_color_illumination/code/day61_notes.md`：详细中文学习笔记；
- 本地第三次优化结果：`D:/DL_code/data/crop_row_perception/day61_color_illumination/*_v4.*`，因CRDLD许可未明确而不纳入Git。

## Day62 产物

- `../../62_crop_row_morphology_regions/code/day62_morphology_regions.py`：开闭运算、连通域/轮廓统计、面积过滤、训练五折选择和已复用验证确认；
- `../../62_crop_row_morphology_regions/tests/test_day62_morphology_regions.py`：二值掩码、形态学、区域统计、门槛与真实文件评估测试；
- `../../62_crop_row_morphology_regions/code/day62_notes.md`：完整中文学习笔记；
- 第一版方案：3×3 opening、5×5 closing、删除小于图像面积0.02%的连通域；
- 第二版冻结方案：3×3 opening、5×7纵向closing、顶部阈值为底部40%的透视感知面积过滤；训练开发集五折中5/5通过，已复用validation-development确认门槛通过；
- 本地第一版、失败候选和第二版结果：`D:/DL_code/data/crop_row_perception/day62_morphology_regions/`，因CRDLD许可未明确而不纳入Git。

## Day63 产物

- `../../63_crop_row_geometry_extraction/code/day63_crop_row_geometry.py`：保留Hough、整线搜索和单行Extra Trees基线，新增多带传统检测、多行顺序匹配、走廊选择、Tiny U-Net与ResNet18中心线网络；
- `../../63_crop_row_geometry_extraction/tests/test_day63_crop_row_geometry.py`：34项单行、多行、走廊、中心线张量、网络形状和端到端边界测试；
- `../../63_crop_row_geometry_extraction/code/day63_notes.md`：保留旧版本历史，并记录多行重学的失败轮次、交叉拟合协议、最终指标和限制；
- 最终方法以RGB和冻结Day62掩码为四通道输入，按训练开发与复用验证开发分别进行三折标签排除的ResNet18中心线交叉拟合；
- 训练开发OOF的多行P/R为0.9552/0.9585，位置MAE 0.0101、方向MAE 1.652°；复用验证开发OOF为0.9507/0.8178、0.0120和2.873°，两个分区均通过Day63多行几何门；
- 合计走廊边界成对正确率0.9160、走廊中心MAE 0.0146，但不安全false-valid率0.1314仍未达到0.05，可靠拒识留给Day65～Day68；
- 本地结果：`D:/DL_code/data/crop_row_perception/day63_crop_row_geometry/`，因CRDLD许可未明确而不纳入Git；
- CRDLD同源内部基准、RowDetr冻结外部集和目标域拒识负样本均未访问或不可用；所有结果仍是已参与选择的同源开发证据，不是外部泛化。

## Day64 产物

- `../../64_crop_row_camera_coordinates_measurement/code/day64_camera_measurement.py`：归一化/像素坐标、图像偏移、方向代理、多线消失点、相机射线、地面变换和明确阻塞状态；
- `../../64_crop_row_camera_coordinates_measurement/tests/test_day64_camera_measurement.py`：15项坐标符号、分辨率、状态传播、消失点、合成标定与端到端测试；
- `../../64_crop_row_camera_coordinates_measurement/code/day64_notes.md`：完整中文公式、测试过程、真实结果、可视化与物理测量边界；
- 冻结Day63 `extra_depth12_leaf2` 后在248张已复用开发图上复算，valid fraction为0.9476，方向定义与Day63最大差异为0°；
- 一条中央行不能提供真实消失点；没有相机内参和地面变换，真实相机射线与米制测量保持 `BLOCKED`；
- 本地结果：`D:/DL_code/data/crop_row_perception/day64_camera_measurement/`，数据图片和运行结果不纳入Git；
- Day65只对图像偏移、方向代理、置信度、不确定性和状态做视频时序稳定。

## 修订后的 Day59～Day70 路线

| Day | 学习任务 | 应有成果 |
|---|---|---|
| 59 | 多行与行间走廊目标合同 | 明确所有可见行、左右边界、走廊中心、消失点、拒识和禁止声明 |
| 60 | 数据与多行标签审计 | 证明CRDLD可做顺序派生多行学习，同时标明实例、走廊、负样本和许可缺口 |
| 61 | 颜色与光照 | 冻结对整幅植被有效的Gray-World+HSV输入，不绑定单中央行 |
| 62 | 形态学与区域 | 冻结保留远近多行结构的透视感知掩码，不按中央目标裁剪 |
| 63 | 多作物行几何重学 | OOF检测全部可评价行，输出有序行、左右边界和图像走廊候选；核心几何门通过 |
| 64 | 坐标与测量边界 | 保留既有坐标公式；Day65接入多行走廊中心和多线消失点，不伪造米制测量 |
| 65 | 多行视频时序与安全状态 | 跟踪行身份、平滑走廊/消失点、处理漏检，并重点降低false-valid |
| 66 | 完整离线Pilot | RGB视频端到端输出多行、边界、走廊中心、方向、置信度和valid/degraded/reject |
| 67 | 失败案例分组 | 定量分类杂草、遮挡、缺行、中央作物行、地头、转弯、光照和时序切换失败 |
| 68 | 一轮受控改进 | 只针对Day67最主要且可由现有开发证据解决的失败原因改进并重新过门 |
| 69 | 冻结测试 | 冻结模型与阈值后一次访问CRDLD内部基准和RowDetr外部正样本；缺负样本则安全门保持BLOCKED |
| 70 | 交付与导师汇报 | Demo、可复现代码、指标/失败报告、证据登记、限制和下一步采集/标定清单 |

Day70的目标是“证据清楚的离线农业机器人视觉Pilot”，不是已经能安全控制真实机器人。
真实车体边界、米制走廊、闭环控制和安全认证仍需要相机/车体标定、目标域负样本与实车测试。
