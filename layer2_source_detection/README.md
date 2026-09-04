# Layer 2 v1.4.4

L2读取当前20 ms的P2控制Gate，默认阈值为0.80。默认开启的突出声源数估计对每个窗口持续增量运行；Gate OPEN时再以固定2阶或同窗映射阶数执行7麦、2～4 kHz加权frequency-normalized MUSIC。跟随模式把计数`0/1`、预热或计数故障映射为1阶，把`2`及以上映射为2阶。Gate关闭时继续计数但不执行MUSIC，已有轨迹只coast/expire。Circular IMM-JPDA维护方向ID、角度和短时预测。v1.4.4不包含DPD、噪声白化或7×7噪声协方差链路。

当前`DecisionWindow`固定携带160 ms、7680 samples，扫描器首次重建得到15个960/480 STFT帧；连续窗口每次增加2帧，两个后继窗口后扩展到配置的200 ms、19帧滚动协方差，再持续移入2帧、移出2帧。`layer2.context_ms=200`同时用于Gate连续开启的ID出生预热门槛：必须连续OPEN 10个20 ms hop后才允许新建方向ID。单窗音频长度和跨窗滚动状态是两个不同概念。

实时公开结果仅包含时间轴、MUSIC诊断、候选、track_id、角度与轨迹状态。没有L3或声纹反馈依赖。
