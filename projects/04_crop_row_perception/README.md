# Practical 04：农业机器人作物行视觉感知

这是一个 Day59-Day71 的短周期学习项目，目标是把已有 OpenCV、深度学习、C++和实验验证能力迁移到农业机器人前视机器视觉问题。

当前只研究离线视觉感知：从单目RGB图像或视频帧检测所有满足可见性规则的作物行，选择
相机中心左右最近的可靠作物行作为当前走廊边界，再估计图像平面走廊中心、方向、消失点、
置信度和拒绝状态。作物行本身不是可通行中心；项目不包含真实机器人控制，也不把图像坐标
误写成物理距离或真实机器人车体边界。

CRDLD、RowDetr 与 SSR 已完成受控实测：CRDLD 只作受限同源开发/内部证据；RowDetr 第一版因标签支持域失配而 INVALID，第二版只作协议开发；SSR v7 `expansion_2` 的 98 张后续现场图像给出了有效外部中央行结果，但 Recall 0.7143 未达 0.80。由于 CRDLD 许可/最高分组仍未知，且前视目标域拒识负样本尚缺，完整数据门禁仍为 `BLOCKED`。

## 门禁状态

| 门禁 | 状态 | 日期 | 证据 |
|---|---|---|---|
| 目标契约 | PASS | 2026-08-31 | `target_contract.yaml` |
| 数据可行性 | SCOPED PASS / FULL BLOCKED | 2026-09-02 | 正样本学习可继续；许可、最高分组与拒识负样本仍阻断完整声明 |
| 环境 | SCOPED PASS | 2026-09-02 | 受限同源正样本学习环境可执行；不解除完整数据门阻断 |
| 管线试运行 | ENGINEERING PASS / DEVICE POLICY DEVELOPMENT PASS | 2026-09-12 | Day61–69管线通过；Day71在一段89帧开发视频上定位cuDNN TF32差异并验证关闭TF32的稳定CUDA入口 |
| 内部有效性 | SCOPED PASS / SAFETY BLOCKED | 2026-09-05 | 多行几何与测量门通过；unsafe false-valid及真实视频安全未通过 |
| 基线/OOD开发 | ENGINEERING PASS / ACCURACY BLOCKED | 2026-09-09 | 开发视频Pilot、失败审计与严重遮挡受控改进完成；无逐帧真值，不声明真实视频准确率 |
| 冻结外部测试 | VALID RESULT / RECALL GATE FAILED | 2026-09-10 | SSR expansion_2 98/98参考有效，中央行Recall 0.7143未达0.80；匹配位置MAE 0.0234、方向MAE 4.374°通过 |
| 交付 | DAY71 COMPLETE / USER APPROVED FOR GITHUB / SAFETY BLOCKED | 2026-09-12 | Day70交付保持冻结；Day71分层诊断和稳定CUDA CLI已由用户确认，进入GitHub交付 |

只有显式 `PASS` 才能进入下一门。开源项目展示和论文指标均不算本项目效果证据。

Day60 后用户明确将执行范围收窄为“同源正样本几何与工程开发”，因此允许沿受限路径完成
Day61–68；这不等于完整数据门通过，也不允许把管线工程结果提升为真实视频安全、独立外部
泛化或实车部署证据。冻结外部测试仍需等参数、评价协议和证据边界全部锁定后再执行。

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

- `../../64_crop_row_camera_coordinates_measurement/code/day64_camera_measurement.py`：保留单行历史基线，当前入口把Day63多行OOF输出转换为左右边界、图像走廊中心、偏移、航向、行间距和多线IRLS消失点；
- `../../64_crop_row_camera_coordinates_measurement/tests/test_day64_camera_measurement.py`：20项坐标、状态传播、中央作物行防误导、缺边界、多线消失点、合成标定与端到端测试；
- `../../64_crop_row_camera_coordinates_measurement/code/day64_notes.md`：已原位改为多行重学最终版，记录公式、RED/GREEN过程、真实结果、失败优化与物理边界；
- 1,498帧Day63标签排除OOF审计得到左右边界成对正确率0.9185、走廊中心MAE 0.0146、走廊航向MAE 1.186°，多线消失点可用率0.9987、中位误差0.0195，Day64测量门全部通过；
- 中央作物行位于相机中心时不生成行驶中心；两侧相邻边界中线才是图像走廊。安全false-valid仍为0.1314，未达到0.05，留给Day65时序与拒识；
- 没有相机内参和地面变换，真实相机射线、米制行距和米制偏移保持 `BLOCKED`；
- 本地结果：`D:/DL_code/data/crop_row_perception/day64_camera_measurement/`，数据图片和运行结果不纳入Git；
- Day65随后跟踪多行身份，并对走廊中心、消失点、偏移、方向、置信度和状态做视频时序稳定与失效判断。

## Day65 产物

- `../../65_crop_row_video_temporal_stability/code/day65_video_temporal.py`：接入冻结Day63模型，完成最长2帧光流补测、有序多行身份关联、置信度加权Kalman跟踪、走廊切换迟滞和失效状态机；
- `../../65_crop_row_video_temporal_stability/tests/test_day65_video_temporal.py`：24项冻结组排除、身份保持、遮挡恢复、中央行/缺边拒绝、光流、指标边界、结果写出和CLI测试；
- `../../65_crop_row_video_temporal_stability/code/day65_notes.md`：记录第一轮过度拒绝、最终优化方法、正式全量复跑、人工抽样和证据边界；
- LeCropFollow CROW开发范围共25段、10,995帧，冻结同源测试为另6段且未访问；全部开发视频解码完成；
- 相对原始逐帧输出，走廊中心抖动降低84.00%、状态切换降低88.89%、走廊切换降低50.37%，时序有效帧保留率为33.91%；
- 7项工程验收全部通过；合成缺边、中央作物行与不受支持场景均未错误发布有效导航；
- 真实视频没有逐帧走廊有效性真值，所以工程门为`PASS`，真实视频安全门仍为`BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH`；
- 本地结果：`D:/DL_code/data/crop_row_perception/day65_video_temporal_verified/`，视频、逐帧结果和抽样图因许可未核验而不纳入Git；
- Day66已冻结当前Day65配置并完成完整离线视频Pilot；未在冻结测试视频上为了改善Demo继续调参。

## Day66 产物

- `../../66_crop_row_offline_video_pilot/code/day66_offline_video_pilot.py`：冻结配置校验、完整视频Pilot、逐帧安全合同、连续叠加、CPU基准和分层视觉审计；
- `../../66_crop_row_offline_video_pilot/code/day66_frozen_config.json`：与Day65正式结果逐字段一致的冻结时序配置；
- `../../66_crop_row_offline_video_pilot/tests/test_day66_offline_video_pilot.py`：覆盖配置漂移、四态合同、非valid导航隔离、显示可读性、视频对齐、运行时边界和视觉审计；
- `../../66_crop_row_offline_video_pilot/code/day66_notes.md`：记录第一轮性能失败、输出等价优化、显示第二版、正式全量重跑和负证据；
- 25段开发视频、10,995帧全部输出帧对齐JSONL和连续MP4；状态为490 valid、47 candidate、10,436 degraded、22 reject，非valid导航泄漏为0；
- 第二版画面显示所有活跃作物行及稳定ID、明确的LEFT/RIGHT边界、valid专属绿色中心、独立黄色方向箭头/角度、置信度、四态和原因；
- 9项工程检查与Day61–66联合138项测试全部通过；640x360 CPU结果为中位41.14 ms、P95 49.33 ms，计时不含解码/缩放和叠加编码；
- 6段冻结同源视频仍未访问；疑似草地/围栏valid留作Day67失败组，真实视频安全门继续为`BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH`；
- 本地结果：`D:/DL_code/data/crop_row_perception/day66_offline_video_pilot/`，视频、JSONL和审计图因许可未核验而不纳入Git；可复核失败分组协议已在Day67完成。

## Day67 产物

- `../../67_crop_row_failure_taxonomy/code/day67_failure_taxonomy.py`：严格读取Day66开发输出，生成连续事件、固定种子分层复核包、上下文媒体、标注聚合和Day68选择门；
- `../../67_crop_row_failure_taxonomy/code/day67_taxonomy.json`：冻结系统触发器、视觉标签、证据级别和Day68跨episode门；
- `../../67_crop_row_failure_taxonomy/tests/test_day67_failure_taxonomy.py`：覆盖冻结隔离、事件边界、导航合同、抽样、渲染、标注与决策；
- 25段/10,995帧形成4,809个事件；第三版复核536个事件，包含全部296个valid事件以及按12层进行SRSWOR抽取的240个非valid事件；
- 两轮全联系表和重点上下文复核发现9个疑似不安全valid事件/13帧/2个episode；草地/围栏valid共6事件/10帧但集中于单一episode，不能声称跨场景错误率；
- 每个抽样事件记录层总体、样本量、真实纳入概率和设计权重；严重遮挡复核35个事件、覆盖14个episode，加权估计占可行动失败事件43.14%，成为Day68唯一目标；第一版未加权和第二版权重错配结论均不再用于决策；
- 本地结果：`D:/DL_code/data/crop_row_perception/day67_failure_taxonomy/`；视觉标注为 `MODEL_ASSISTED_REVIEW_DEVELOPMENT_ONLY`，六段冻结同源视频未访问，真实视频安全仍为 `BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH`。

## Day68 产物

- `../../68_crop_row_severe_occlusion/code/day68_severe_occlusion.py`：因果恢复候选、走廊边缘可观测性/边界越界守卫、valid专属导航清空、全量运行和叠加；
- `../../68_crop_row_severe_occlusion/code/day68_frozen_config.json`：冻结阈值、输入哈希、第一版失败记录与第二版敏感性；
- `../../68_crop_row_severe_occlusion/tests/test_day68_severe_occlusion.py`：12项状态合同、可观测性、冻结隔离与逐帧视频对齐测试；
- 第一版拦截3/3目标但误拦36个已复核正常帧，第二版将正常误拦降到14帧，同时保留473/490=96.53%的Day66 valid帧；
- 最终在25段开发视频/10,995帧上把3个已复核严重遮挡疑似不安全valid事件降为0，非valid导航泄漏为0；
- 25个帧对齐JSONL和25个叠加MP4均在本地生成并完整解码核对，Day61～68联合175项测试通过；
- 本地结果：`D:/DL_code/data/crop_row_perception/day68_severe_occlusion/`；六段冻结同源视频零访问，模型辅助复核不是真值，真实视频安全仍为 `BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH`。

## Day69 产物

- `../../69_crop_row_frozen_evaluation/code/run_crop_row_pilot.py`：用户可运行的单视频/目录CLI，输出2倍分辨率叠加MP4、逐帧JSONL、CSV和汇总报告；
- `../../69_crop_row_frozen_evaluation/启动禾迹视觉台.bat`：Windows双击入口，自动打开只监听本机的“禾迹”视频上传与结果预览网页；
- `../../69_crop_row_frozen_evaluation/code/day69_web_ui.py`：复用冻结CLI的Gradio界面，提供设备选择、隔离输出、进度、叠加视频预览与五类结果下载；
- `../../69_crop_row_frozen_evaluation/code/day69_frozen_evaluation.py`：冻结协议校验、RowDetr折线转换、静态几何汇总和Day63→65→66→68完整视频组合；
- `../../69_crop_row_frozen_evaluation/code/day69_frozen_protocol.json`：第一次读取冻结媒体前锁定的12项代码/模型/清单哈希、指标和反调参规则；
- `../../69_crop_row_frozen_evaluation/code/day69_frozen_result_summary.json`：不包含原始数据的精简冻结结果；
- `../../69_crop_row_frozen_evaluation/code/day69_external_adjudication.json`：RowDetr参考转换无可评估行的访问后裁决；
- `../../69_crop_row_frozen_evaluation/code/day69_v2_evaluation.py`：局部折线/YOLO多边形解析、最大匹配数优先的单调动态规划、零分母fail-closed汇总和SSR静态评估；
- `../../69_crop_row_frozen_evaluation/code/day69_v2_frozen_protocol.json`：第二版读取SSR expansion_2前锁定的评价、阈值、门槛、哈希与反调参规则；
- `../../69_crop_row_frozen_evaluation/code/day69_v2_result_summary.json`：第二版外部结果、失败门和不可声明范围；
- `../../69_crop_row_frozen_evaluation/assets/day69_v2_ssr_external_audit.jpg`：12个miss与8个高方向误差match的视觉复核；
- `../../69_crop_row_frozen_evaluation/assets/heji_web_ui.png`：“禾迹”本地视觉台实际浏览器渲染截图；
- `../../69_crop_row_frozen_evaluation/code/day69_notes.md`：方法、结果、负证据和程序使用说明；
- `../../69_crop_row_frozen_evaluation/tests/test_day69_frozen_evaluation.py`：协议漂移、角色隔离、折线语义、正样本安全边界、静态哈希和端到端输出测试；
- 6段同源冻结视频共896帧，输出18 valid、12 candidate、841 degraded、25 reject，所有源哈希和叠加视频完整性通过，导航契约违规为0；
- CRDLD同源内部429张的Precision/Recall为0.9658/0.9702，绝对几何门通过，但边界配对相对开发下降0.1056，略超预注册0.10上限；
- RowDetr 1,760张/3,929条折线中0条跨越预注册`y=0.40`，因此原始0值不可解释为准确率，外部正样本评估裁决为`INVALID_REFERENCE_TRANSFORMATION_NO_EVALUABLE_ROWS`；
- 第二版在RowDetr开发证据上证明局部可见指标可评价3,322条折线；随后拒绝边界框标签的CRIS-Cotton，改用从未触碰的SSR v7 expansion_2；
- SSR 98/98标签可评价、70张匹配，中央行Recall 0.7143未达0.80；已匹配位置MAE 0.0234和方向MAE 4.374°通过。该有效负结果不允许事后调参重跑；
- 本地完整结果位于`D:/DL_code/data/crop_row_perception/day69_frozen_evaluation/`，许可未核验的原始视频、图像和衍生产物不进入Git；
- 第二版SSR完整结果位于`D:/DL_code/data/crop_row_perception/day69_v2_ssr_external/`；可运行程序已经完成，但完整多行外部泛化、负样本拒识和真实视频安全均未建立。
- 网页在真实89帧开发视频上完成浏览器上传与按钮端到端检查，得到39 valid、1 candidate、49 degraded、0 reject，导航契约违规为0；网页是本地诊断包装，不改变冻结成绩或安全边界。

## Day70 产物

- `../../70_crop_row_delivery_report/code/day70_delivery.py`：检查冻结哈希、来源指标、声明状态、本地网页约束、文档合同和Git候选文件卫生，并生成机器可读报告与SVG证据板；
- `../../70_crop_row_delivery_report/code/day70_delivery_manifest.json`：固定Day69基线提交、10个核心文件SHA-256、16项来源指标和六个最终声明状态；
- `../../70_crop_row_delivery_report/code/day70_delivery_report.json`：Day70实际交付检查结果，不重新计算或覆盖Day69成绩；
- `../../70_crop_row_delivery_report/code/day70_device_parity_result.json`：同一89帧开发视频CPU/CUDA复现审计；37帧状态与导航可用性不一致，两端导航合同违规均为0；
- `../../70_crop_row_delivery_report/assets/day70_evidence_board.svg`：明确区分 `PASS`、`FAILED`、`NOT_ESTABLISHED` 和 `BLOCKED` 的证据状态总览；
- `../../70_crop_row_delivery_report/docs/day70_advisor_brief.md`：面向导师的项目目标、方法演进、运行接口、核心结果、失败案例和下一步；
- `../../70_crop_row_delivery_report/docs/acquisition_and_calibration_checklist.md`：把目标域负样本、逐帧真值、多行实例、相机内外参、地面变换和影子模式逐项映射到阻断门；
- `../../70_crop_row_delivery_report/code/day70_notes.md`：Day70完整中文学习笔记、运行方法、证据表和最终结论；
- Day70只打包证据，不修改Day69模型、阈值、协议或冻结结果。最终表述是“证据清楚的离线农业机器人视觉Pilot”：工程交付通过，SSR中央行Recall 0.7143未达0.80；同一89帧输入的CPU/CUDA状态轨迹有37帧不一致，精确复现门为 `FAILED`，演示复现应显式选择CPU；完整多行外部泛化、目标域拒识、真实视频安全和米制控制仍为 `BLOCKED`。

## Day71 产物

- `../../71_crop_row_device_parity_diagnosis/code/day71_device_parity.py`：六臂CPU/CUDA、后端、batch和TF32受控诊断；
- `../../71_crop_row_device_parity_diagnosis/code/day71_device_parity_result.json`：根因、逐层差异、运行时间和机器可读验收结果；
- `../../71_crop_row_device_parity_diagnosis/code/day71_frame_differences.csv`：89帧设备状态、解码行数和概率差异；
- `../../71_crop_row_device_parity_diagnosis/code/run_crop_row_pilot_device_stable.py`：关闭cuDNN TF32、强制CUDA batch32并把实际策略写入报告的稳定执行入口；
- `../../71_crop_row_device_parity_diagnosis/code/day71_notes.md`：完整中文学习笔记和证据边界；
- 生产CPU与默认CUDA batch32的37帧状态/导航差异被精确复现；关闭TF32后，真实稳定CUDA与生产CPU CLI达到89/89帧状态和导航严格一致，连续几何最大绝对差4.823e-06（容差1e-05）；输出39 valid、1 candidate、49 degraded、0 reject且导航违规为0；
- 该结果只覆盖当前环境的一段开发视频，不证明全部视频、其他GPU、外部泛化、拒识安全或真实机器人可靠性。

## 修订后的 Day59～Day71 路线

| Day | 学习任务 | 应有成果 |
|---|---|---|
| 59 | 多行与行间走廊目标合同 | 明确所有可见行、左右边界、走廊中心、消失点、拒识和禁止声明 |
| 60 | 数据与多行标签审计 | 证明CRDLD可做顺序派生多行学习，同时标明实例、走廊、负样本和许可缺口 |
| 61 | 颜色与光照 | 冻结对整幅植被有效的Gray-World+HSV输入，不绑定单中央行 |
| 62 | 形态学与区域 | 冻结保留远近多行结构的透视感知掩码，不按中央目标裁剪 |
| 63 | 多作物行几何重学 | OOF检测全部可评价行，输出有序行、左右边界和图像走廊候选；核心几何门通过 |
| 64 | 多行坐标与测量边界 | 用左右相邻行得到走廊中心、偏移、方向、行距和多线消失点，不伪造米制测量 |
| 65 | 多行视频时序与安全状态 | 已完成：保持多行身份、平滑走廊、短遮挡恢复和显式拒绝；真实视频安全率因缺真值仍阻塞 |
| 66 | 完整离线Pilot | 已完成：输出多行/ID、边界、valid专属中心、独立方向代理、置信度、四态和原因；真实视频安全仍阻塞 |
| 67 | 失败案例分组 | 已完成：事件级系统触发器与视觉标签分离；全valid复核并将严重遮挡预注册为Day68目标 |
| 68 | 一轮受控改进 | 已完成：只改严重遮挡下的可观测性/拒绝，拦截3/3开发期目标并保留96.53% valid，非valid导航泄漏为0 |
| 69 | 冻结测试 | 第二版已完成：修复RowDetr暴露的零参考评价缺陷；SSR 98张有效外部中央行评估中位置/方向门通过、Recall 0.7143未达0.80；安全门保持BLOCKED |
| 70 | 交付与导师汇报 | 已完成：Demo、可复现检查器、指标/失败报告、证据登记、限制和下一步采集/标定清单 |
| 71 | 设备一致性诊断与稳定执行 | 已完成：分层隔离后定位cuDNN TF32路径；稳定CUDA batch32在89帧开发视频上与生产CPU最终一致，完整跨设备结论仍待扩大验证 |

Day70～71的目标是“证据清楚且运行策略可复核的离线农业机器人视觉Pilot”，不是已经能安全控制真实机器人。
真实车体边界、米制走廊、闭环控制和安全认证仍需要相机/车体标定、目标域负样本与实车测试。
