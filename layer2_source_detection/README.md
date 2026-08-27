# Layer 2 v1.4

L2读取当前20 ms的P2控制Gate，默认阈值为0.80。Gate开启并满足连续上下文后，在7麦200 ms滚动统计上执行2～4 kHz加权frequency-normalized MUSIC。Circular IMM-JPDA维护方向ID、角度和短时预测。v1.4不再包含DPD、噪声白化或7×7噪声协方差链路。

实时公开结果仅包含时间轴、MUSIC诊断、候选、track_id、角度与轨迹状态。没有L3或声纹反馈依赖。
