# Pipeline Log UI

该目录实现项目 1.1.0 的独立只读 Pipeline Log UI。它是观察平面，不是 Layer 5，也不是 Development Test UI 的面板。

入口为 `launch_log_ui(provider)` 或 `PipelineLogWindow(provider)`。`provider` 必须由正式宿主显式注入，并提供已经存在的项目公开查询方法；Log UI 不接受 data root，不构造 `DataManagerService`，不打开 Catalog/SQLite/WAL，不消费 Runtime latest-only 邮箱，也没有启动、停止、参数修改、标注、导出、删除、恢复或重建控件。

首版提供记录列表、会话总览、Pipeline 时间线、单窗详情、ID 与异常五页。v3/v4 通过 capability 与 schema 双重探测；缺失字段分别保留为 `N/A / 接口未提供 / 未记录 / 尚未封存 / 校验失败`。未知 schema fail-closed，不进入统计。完成频率只以明确 `COMPLETED` 窗口为分子，以各 epoch 完整公开 sample 区间之和为分母；compute、queue wait、end-to-end 分位数分别显示样本数和缺失数。

音频默认不读取。用户选择方向轨并点击播放后，界面才调用公开 `track_audio_assets`，只播放该接口已经完成路径范围和 hash 校验后返回的资产，且界面不展示绝对路径。session 模型使用有界内存 LRU，时间线按 500 窗分页；后台加载可取消，关闭 UI 不影响 Runtime、录音或其他界面。

当前项目没有跨进程只读逐窗事件流。未注入公开查询端口时必须显示 `Unavailable`，不得通过私有字段、内部队列或 Catalog 文件补救。
