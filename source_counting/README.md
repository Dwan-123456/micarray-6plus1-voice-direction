# 突出声源数估计

本目录实现轻量实时评估器，只估计当前可分辨的突出方向声源数`0/1/2`。讲话、扬声器、敲击和其他方向性噪声采用相同规则；不判断声音类别，不输出方向、分数或置信度，也不写入磁盘。估计默认开启，在单一L2 worker中对每个20 ms窗口持续增量执行，与P Gate状态无关；只有手动关闭估计时才停止。

## 算法

`IncrementalGccPhatSourceCounter`使用7个物理麦的21个麦克风对：

1. 在2～4 kHz计算PHAT归一化互谱；
2. 通过预计算的360°远场TDOA表形成GCC-PHAT空间图；
3. 用能量、空间图最大峰和robust-z区分`0`与`≥1`；
4. 在每个麦克风对的GCC中对第一峰预测时延施加软notch，重新形成残差图；
5. 只接受离主峰至少50°的真实局部极大值，并以残差绝对峰、robust-z和残差/主峰比筛选第二候选；
6. 直接复用已缓存的逐帧PHAT互谱，要求至少3个重叠帧同时支持两个候选，避免把160 ms内先后出现的两个方向误算为同时两个；此校验不执行额外FFT；
7. 最近3次判断中至少2次一致后发布计数。

第一次收到新session/epoch、重新启用估计或超过160 ms的数据断点时建立15个50%重叠STFT帧。正常每20 ms只计算新增的2帧并淘汰最旧2帧；若L2队列跳过少量窗口，则只补算尚未处理的新帧，不重复变换整个160 ms上下文。Gate关闭不会中断或重置计数；手动关闭时立即清空累计与稳定状态，不执行FFT、空间图或计数。

配置缺少整个`source_counting`段时，估计和“MUSIC阶数使用估计结果”均默认开启；本仓库配置显式采用相同默认值。旧外置配置仍可加载，但升级后若不需要持续估计，应通过Test UI或配置手动关闭。

## 运行边界与MUSIC阶数

- 每个权威`DecisionWindow`在同一worker、同一时间轴身份下持续估计声源数；P Gate只控制随后是否执行MUSIC，异步旧结果不会驱动新窗口。
- “MUSIC阶数使用估计结果”关闭时，Gate OPEN后的MUSIC固定2阶；开启后，计数`0/1`和预热`None`映射为1阶，计数`2`及以上映射为2阶。计数故障安全回退到1阶且不升级为主L2故障。
- Gate关闭时计数仍更新和显示，MUSIC不执行，已有方向ID按原跟踪规则coast/expire。
- Test UI右下角独立控制框显示两个开关、`突出声源数：0/1/2`和当前MUSIC阶数；底部性能栏另行显示计数平均耗时、持续执行帧率和故障率。预热、故障、关闭估计、过期或快速重启后的旧结果显示`—`。

## 实现来源

实现参考了成熟项目的公开结构，但没有复制或引入其代码和依赖：

- [pyroomacoustics SRP-PHAT](https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/srp.py)：MIT，参考PHAT互谱和steered-response计算；完整包需要额外编译链，因此未作为运行依赖。
- [ODAS](https://github.com/introlab/odas)：MIT，参考实时多麦克风对、固定数量potential以及已检测TDOA抑制的旁路结构；完整C运行时的FFTW/ALSA等依赖未引入。
- [ReSpeaker mic_array](https://github.com/respeaker/mic_array)：Apache-2.0，核对紧凑6+1阵列的GCC-PHAT与插值实践；旧采集层未引入。
- Brutti、Omologo与Svaizer，[Multiple Source Localization Based on Acoustic Map De-Emphasis](https://link.springer.com/article/10.1155/2010/147495)：主峰去强调后以残差峰判断第二源的论文依据。

这里的`2`表示两个可由当前8 cm直径阵列分辨的突出方向。两个物理声源若方向相同或过近，仍会被计为`1`；阈值属于评估初值，必须通过真实阵列录音继续标定。
