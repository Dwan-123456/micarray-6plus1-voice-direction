# Layer 2 v1.4

L2读取当前20 ms的P2控制Gate，默认阈值为0.80。默认开启的突出声源数估计对每个窗口持续增量运行；Gate OPEN时再以固定2阶或同窗映射阶数执行7麦200 ms、2～4 kHz加权frequency-normalized MUSIC。跟随模式把计数`0/1`、预热或计数故障映射为1阶，把`2`及以上映射为2阶。Gate关闭时继续计数但不执行MUSIC，已有轨迹只coast/expire。Circular IMM-JPDA维护方向ID、角度和短时预测。v1.4不再包含DPD、噪声白化或7×7噪声协方差链路。

实时公开结果仅包含时间轴、MUSIC诊断、候选、track_id、角度与轨迹状态。没有L3或声纹反馈依赖。
