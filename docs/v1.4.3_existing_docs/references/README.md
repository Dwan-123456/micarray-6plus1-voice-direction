# 重要研究参考资料

> **历史研究资料索引**：本目录保存不同开发阶段收集的外部报告和硬件资料，用于解释设计空间、提出实验假设和规划后续优化。报告内容不自动成为v1.4.3已实现功能或性能结论。

当前v1.4.3实现和约束的权威顺序为：

1. 项目代码与`config/config.yaml`；
2. 根`README.md`和`docs/project_document/`现行技术文档；
3. `CHANGELOG.md`；
4. 本目录研究报告。

## 报告索引

### Sipeed MicArray麦克风阵列官方资料

文件：[`sipeed_micarray.md`](sipeed_micarray.md)

官方页面：<https://wiki.sipeed.com/hardware/zh/modules/micarray.html>

官方源文档：<https://github.com/sipeed/sipeed_wiki/blob/main/docs/hardware/zh/modules/micarray.md>

下载快照：2026-08-26，官方文件最新提交`232411a35675f410233a4a808aaa3308c5b4427f`。资料说明7个`MSM261S4030H0`麦克风、I²S引脚、12颗`SK9822`灯、机械尺寸及K210参考用法。本文是硬件模块通用资料，不代表MA-USB8通过USB暴露7路物理麦。

### Sipeed MA-USB8（BL616）官方使用指南

文件：[`sipeed_ma_usb8_bl616.md`](sipeed_ma_usb8_bl616.md)

官方页面：<https://wiki.sipeed.com/hardware/zh/modules/micarray_usbboard_bl616.html>

官方源文档：<https://github.com/sipeed/sipeed_wiki/blob/main/docs/hardware/zh/modules/micarray_usbboard_bl616.md>

下载快照：2026-08-26，官方文件最新提交`0f5fca9043bd467d1eee216f82bc811b5529225a`。指南明确MA-USB8 UAC2.0采集为8通道、PCM S16_LE、48 kHz：CH0～CH5是六个外圈原始麦，CH6是六麦延时求和后的全频段波束输出，CH7是保留PEC通道而非中心麦；另含灯控命令`e/E`、12档波束、USB/串口热力图与故障排查。该通道定义应作为后续实机通道映射复核的重要依据，尚未因资料入库自动修改当前代码配置。

### 6+1阵列MUSIC伪峰抑制工程报告（中文版PDF）

文件：[`music_false_peak_suppression_6plus1_cn.pdf`](music_false_peak_suppression_6plus1_cn.pdf)

原文件：`MUSIC_false_peak_suppression_report_CN.pdf`。

重点覆盖：

- 按当前48 kHz、NFFT 1024、20 ms更新和半径4 cm的6+1阵列，计算不同频率及50°～180°夹角的最坏阵列流形相关度；
- 建议将2.0～3.8 kHz作为低阈值候选带，将2.7～3.6 kHz作为一源/两源白化残差核验带，并对3.8～4.0 kHz降权；
- 将完整噪声协方差白化、频点SPP/Eigen-SNR/几何权重、跨频支持和一源/两源残差下降组合为第二候选的证据；
- 建议对仅使用六个环麦与使用完整6+1阵列进行A/B测试，验证中心麦对方向流形相关度和实测谱的影响；
- 给出静音、单源、双源和功率不平衡场景的错误第二源率、False births/min、召回率、确认延迟与轨迹碎片率标定方法。

当前v1.4.3采用边界：Runtime根据GCC-PHAT计数使用1/2阶MUSIC，正常公开观测最多2个方向，内部最多维护4条方向轨迹，并保持50°硬NMS；Gate连续OPEN约200 ms后才允许新ID出生。报告中的噪声白化、30～35° NMS、严格第二峰跨频门禁和320～480 ms窗口尚未成为当前实现要求。

### 低源数安静声场的MUSIC伪峰与旁瓣双层抑制

文件：[`music_pseudo_peak_sidelobe_suppression_6plus1.md`](music_pseudo_peak_sidelobe_suppression_6plus1.md)

原题：《面向低源数安静声场的 MUSIC 伪峰与旁瓣双层抑制研究报告》。

重点覆盖：

- 针对当前半径4 cm、六环形麦加中心麦的7麦平面阵列，分析模型阶数、有限快拍、有色噪声、阵列流形失配和混响导致的MUSIC伪峰；
- 论证将合法声源数严格限制为0～2，并停止使用原始特征值95%累计能量直接估计声源数；
- 建议利用静音段噪声协方差实施预白化或GEVD，并以1～4 kHz为宽证据频带、2.5～4 kHz为高空间分辨权重频带；
- 提出第一声源高召回、第二声源接受跨频/跨时间/子空间联合验证的非对称判决，以及必要时使用候选区SBL或残差模型复核；
- 给出仿真、真实阵列、不同SNR/快拍/混响/通道误差和双源功率差的测试矩阵与验收指标。

适配边界：报告是外部研究参考，不是当前实现契约。当前v1.4.3使用48 kHz输入、2～4 kHz Rolling NormMUSIC、GCC-PHAT 0/1/2计数映射MUSIC 1/2阶，以及Circular IMM-JPDA；报告提出的特征值模型阶数、1～4 kHz融合、第二峰模型残差复核和SBL尚未实现。

### DOA短时消失与ID连续性工程方案

文件：[`doa_tracking_hungarian_kalman_short_dropout.pdf`](doa_tracking_hungarian_kalman_short_dropout.pdf)

原题：《基于匈牙利数据关联与卡尔曼滤波的 DOA 声源追踪：解决两秒以内短时消失导致 ID 不连续的工程方案》。

重点覆盖：

- 使用匈牙利算法完成多声源观测与既有轨迹的数据关联；
- 使用卡尔曼预测跨越两秒以内的短时观测缺失；
- 通过轨迹生命周期、门控和恢复规则维持方向ID连续性；
- 面向当前L2内部跟踪与L3按ID连续收音的工程实现建议。

### DOA与说话人识别联合的短时失联追踪

文件：[`doa_speaker_id_hungarian_kalman_tracking.pdf`](doa_speaker_id_hungarian_kalman_tracking.pdf)

原题：《基于 DOA 与说话人识别的短时失联鲁棒目标追踪：匈牙利数据关联、卡尔曼预测与 ID 连续性方案》。

重点覆盖：

- 将DOA运动信息与说话人身份信息共同用于目标关联；
- 在遮挡、漏检和短时失联期间维持目标ID；
- 讨论匈牙利关联代价、卡尔曼预测和轨迹恢复策略；
- 为后续引入说话人嵌入提供研究依据，不代表当前项目已经实现说话人识别。

### 6+1阵列空间可分离度图

文件：[`rho_map_6plus1_100hz_100_6000hz_1deg.png`](rho_map_6plus1_100hz_100_6000hz_1deg.png)

原文件：`rho_map_100Hz_100_6000Hz_angle1deg.png`。

该图按100 Hz频率分辨率和1°角度分辨率展示6+1阵列在100～6000 Hz、0～180°夹角下的空间相关度`rho`。绿色低`rho`区域表示较易分离，黄色为一般，红色高`rho`区域表示较难分离。它用于候选角距、频带选择和L3波束形成研究，不直接替代真实房间与实机测试。

### Python双声源波束形成与实时优化

文件：[`python_two_source_beamforming_realtime_optimization.pdf`](python_two_source_beamforming_realtime_optimization.pdf)

重点覆盖：

- 继续以Python为主，优先批量`solve/eigh/einsum`、缓存、预分配和减少CPU/GPU往返；
- 以DOA、公共`track_id`和轻量Mask网络估计目标/干扰SCM，再交给MVDR/LCMV；
- Oracle-mask MVDR、WPE和大型神经分离模型的消融与上界测试；
- 角度、SIR、SNR、混响、运动、重叠讲话、音质和实时性能的实验矩阵。

### 4 cm 6+1阵列双固定声源L3优化

文件：[`l3_4cm_6plus1_dual_fixed_source_separation.pdf`](l3_4cm_6plus1_dual_fixed_source_separation.pdf)

重点覆盖：

- 4 cm 6+1阵列在不同频率和90°～180°夹角下的相关度与WNG分析；
- 从自由场DOA steering升级到逐`track_id`真实RTF；
- 分开维护背景噪声和两个说话人的SCM，使竞争说话人进入干扰协方差；
- 用WNG约束连续soft-null替代固定`rho`硬分支，以及低频Wiener/MWF后滤波；
- 自由场、麦克风误差、DOA误差、早晚混响和真实IMCRA协方差的分层消融。

## 当前项目适配说明

报告中的部分架构描述基于旧研究阶段的320 ms窗口。当前v1.4.3只实现L1/L2：单个`DecisionWindow`为160 ms、48 kHz、7680 samples并每20 ms发布；MUSIC跨连续窗口维护200 ms滚动协方差。当前没有L3/L4运行链，未来RTF/SCM和波束形成方案见`docs/project_document/07-future-beamforming-and-two-speaker-reconstruction.md`。

报告中的加速倍数、SI-SDR、PESQ、WER、WNG门限和分频验收值来自文献、理论计算或工程估计，必须在当前硬件、真实6+1阵列和诊室录音上重新验证，不能直接写成项目已达到的指标。
