# 已知问题

当前没有已确认且仍未解决的问题。

## 已解决：ADM-001 连续录音无法正常结束

- 状态：已于2026-08-14修复。
- 修复：手动模式的“结束”执行暂停封段；连续/事件模式直接切换到关闭状态，不再调用仅限手动模式的暂停接口。
- 回归测试：`test_capture_stop_uses_mode_specific_control_without_continuous_pause_bug`。
