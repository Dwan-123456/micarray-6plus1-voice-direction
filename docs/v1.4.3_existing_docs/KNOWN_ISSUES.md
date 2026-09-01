# 已知问题与验收边界

> **历史归档快照**：本文记录旧完整架构时期的已知问题，其中Audio Data Manager、L1 HTTP服务、Production UI和L3–L6不属于当前v1.4.3精简运行范围。当前边界见根`README.md`和`docs/project_document/06-limitations-and-open-issues.md`。

当前静态审查未发现已确认的软件阻断项。以下内容仍需真实设备或目标环境验收，不能由静态检查代替：

- Sipeed MA-USB8 的真实 8 通道顺序、CDC 控制、指示灯和热力图输入。
- L2–L4 在真实声场中的方向、多声源、人声概率和 CUDA 性能目标。
- 长时间录音、磁盘容量边界、恢复和真实录音回放。
- 正式面向最终用户的人声方向界面与统一 `app.main` 入口尚未定稿；当前可用入口是 Development Test UI、Audio Data Manager 和 L1 设备服务。

Pipeline Log UI 由本次审查明确排除，不在上述结论范围内。

## 已解决：ADM-001 连续录音无法正常结束

- 状态：已于2026-08-14修复。
- 修复：手动模式的“结束”执行暂停封段；连续/事件模式直接切换到关闭状态，不再调用仅限手动模式的暂停接口。
- 回归测试：`test_capture_stop_uses_mode_specific_control_without_continuous_pause_bug`。
