# v1.4 Runtime

`ApplicationRuntime`只调度真实输入、L1 IMCRA、可选预降噪、WindowAssembler和L2 MUSIC/ID。L2 latest-wins队列、10秒音频环、UI邮箱和1秒性能事件均有硬上限；停止或重新采集会清空内存状态，不写音频。
