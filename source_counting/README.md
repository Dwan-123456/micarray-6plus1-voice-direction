# 突出声源数估计

本目录实现轻量实时评估器，只估计当前可分辨的突出方向声源数`0/1/2`。讲话、扬声器、敲击和其他方向性噪声采用相同规则；不判断声音类别，不输出方向、分数或置信度，也不写入磁盘。它在单一L2 worker中位于P Gate与MUSIC之间，只有当前20 ms窗口的P Gate为OPEN且手动开关开启时才执行。

## 算法

`IncrementalGccPhatSourceCounter`使用7个物理麦的21个麦克风对：

1. 在2～4 kHz计算PHAT归一化互谱；
2. 通过预计算的360°远场TDOA表形成GCC-PHAT空间图；
3. 用能量、空间图最大峰和robust-z区分`0`与`≥1`；
4. 在每个麦克风对的GCC中对第一峰预测时延施加软notch，重新形成残差图；
5. 只接受离主峰至少50°的真实局部极大值，并以残差绝对峰、robust-z和残差/主峰比筛选第二候选；
6. 直接复用已缓存的逐帧PHAT互谱，要求至少3个重叠帧同时支持两个候选，避免把160 ms内先后出现的两个方向误算为同时两个；此校验不执行额外FFT；
7. 最近3次判断中至少2次一致后发布计数。

第一次收到新session/epoch、P Gate重新OPEN或超过160 ms的数据断点时建立15个50%重叠STFT帧。正常每20 ms只计算新增的2帧并淘汰最旧2帧；若L2队列跳过少量窗口，则只补算尚未处理的新帧，不重复变换整个160 ms上下文。Gate非OPEN或手动关闭时立即清空累计与稳定状态，不执行FFT、空间图或计数。

与旧`project_config_v1.4`配置兼容：配置缺少整个`source_counting`段时估计默认关闭，MUSIC固定2阶；本仓库配置显式启用估计、默认不让其驱动MUSIC。这样升级不会让既有外置配置加载失败。

## 运行边界与MUSIC阶数

- 每个权威`DecisionWindow`先评估P Gate，再在同一worker、同一时间轴身份下估计声源数，最后决定本窗MUSIC行为；异步旧结果不会驱动新窗口。
- “MUSIC阶数使用估计结果”关闭时，Gate OPEN后的MUSIC固定2阶；开启后，计数`1/2`对应MUSIC 1/2阶，计数`0`跳过本窗特征分解与空间谱，预热`None`或计数故障也显式跳过，绝不把0传入MUSIC噪声子空间切片。
- Gate关闭、计数为0、预热或故障时，已有方向ID只按原跟踪规则coast/expire。计数异常只记录计数故障，不升级为主L2故障。
- Test UI右下角独立控制框显示两个开关、`突出声源数：0/1/2`和当前MUSIC阶数；底部性能栏另行显示计数平均耗时、执行帧率和故障率。预热、Gate关闭、故障、过期或快速重启后的旧结果显示`—`。

## 实现来源

实现参考了成熟项目的公开结构，但没有复制或引入其代码和依赖：

- [pyroomacoustics SRP-PHAT](https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/srp.py)：MIT，参考PHAT互谱和steered-response计算；完整包需要额外编译链，因此未作为运行依赖。
- [ODAS](https://github.com/introlab/odas)：MIT，参考实时多麦克风对、固定数量potential以及已检测TDOA抑制的旁路结构；完整C运行时的FFTW/ALSA等依赖未引入。
- [ReSpeaker mic_array](https://github.com/respeaker/mic_array)：Apache-2.0，核对紧凑6+1阵列的GCC-PHAT与插值实践；旧采集层未引入。
- Brutti、Omologo与Svaizer，[Multiple Source Localization Based on Acoustic Map De-Emphasis](https://link.springer.com/article/10.1155/2010/147495)：主峰去强调后以残差峰判断第二源的论文依据。

这里的`2`表示两个可由当前8 cm直径阵列分辨的突出方向。两个物理声源若方向相同或过近，仍会被计为`1`；阈值属于评估初值，必须通过真实阵列录音继续标定。
