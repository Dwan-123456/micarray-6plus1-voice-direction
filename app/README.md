# v1.4 Runtime

`ApplicationRuntime`调度真实输入、L1 IMCRA、可选预降噪、WindowAssembler以及L2。声源数估计默认开启，在唯一L2 worker中对每个`DecisionWindow`持续增量运行，与P Gate是否OPEN无关；手动关闭估计时才停止并清空状态。Gate OPEN且阶数跟随开启时，同窗计数`0/1`或预热映射为MUSIC 1阶，`2`及以上映射为2阶；跟随关闭时固定2阶。Gate关闭时继续计数但不执行MUSIC。估计异常单独记录并与主L2故障隔离。L2队列、1秒完整块音频环、1秒底层原始音频环、UI latest-only邮箱和1秒性能事件均有硬上限；停止或重新采集会清空内存状态，不写音频或计数历史。
