# v1.4 Runtime

`ApplicationRuntime`调度真实输入、L1 IMCRA、可选预降噪、WindowAssembler以及L2。每个`DecisionWindow`在唯一L2 worker中依次执行当前P Gate、可选突出声源数估计、MUSIC和方向ID；因此计数驱动的MUSIC阶数与Gate严格属于同一窗口，不存在异步旧结果。Gate非OPEN或估计开关关闭时不执行计数；估计异常单独记录并与主L2故障隔离。L2队列、10秒音频环、UI latest-only邮箱和1秒性能事件均有硬上限；停止或重新采集会清空内存状态，不写音频或计数历史。
