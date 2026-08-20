# 重要研究参考资料

本目录保存与当前6+1麦克风阵列项目直接相关的外部深度研究报告。它们用于解释设计空间、提出实验假设和规划后续L3优化，不是当前版本的规范性接口或已经完成的性能结论。

当前实现和约束的权威顺序仍为：

1. 项目代码与`config/config.yaml`；
2. `ARCHITECTURE_V1.1_TARGET.md`及各层公共契约；
3. 根目录`README.md`和`CHANGELOG.md`；
4. 本目录研究报告。

## 报告索引

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

报告中的部分架构描述基于研究时的320 ms窗口。当前主线已经将L3和L4直接音频窗口统一为160 ms、48 kHz、7680 samples，并继续每20 ms调度一次。报告提出的长时间RTF/SCM统计应理解为跨多个160 ms窗口维护的递归状态，不应据此把直接音频窗口恢复为320 ms。

报告中的加速倍数、SI-SDR、PESQ、WER、WNG门限和分频验收值来自文献、理论计算或工程估计，必须在当前硬件、真实6+1阵列和诊室录音上重新验证，不能直接写成项目已达到的指标。
