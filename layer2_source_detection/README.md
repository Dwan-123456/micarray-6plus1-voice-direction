# Layer 2 v1.4

L2读取当前20 ms的P2控制Gate，默认阈值为0.80。每个窗口先得到新鲜Gate结果；仅在Gate OPEN时运行可选的突出声源数估计，然后以固定2阶或同窗估计的1/2阶在7麦200 ms滚动统计上执行2～4 kHz加权frequency-normalized MUSIC。估计为0、尚在预热或发生计数故障时显式跳过本窗MUSIC，已有轨迹只coast/expire。Circular IMM-JPDA维护方向ID、角度和短时预测。v1.4不再包含DPD、噪声白化或7×7噪声协方差链路。

实时公开结果仅包含时间轴、MUSIC诊断、候选、track_id、角度与轨迹状态。没有L3或声纹反馈依赖。
