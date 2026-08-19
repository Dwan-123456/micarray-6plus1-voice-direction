# 麦克风阵列录音与数据管理

权威目标见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../../ARCHITECTURE_V0.3_TARGET.md)。**当前程序仍按v0.2资产格式运行，v0.3界面与存储迁移尚未实施。**

目标界面继续管理Runtime Sessions、Test Corpus、录制向导、质量与标注、系统维护及实验快照。新增内容包括：native/logical 8ch通道说明、HardwareMix标识、IMCRA/Gate时间线、原始SRP空间响应、L2平滑候选、逐方向L3音频和L4结果。内部追踪ID不显示、不检索也不写入资产。

“用所选样本进行模拟测试”目标输入为已登记的logical 8ch 48 kHz音频，并通过同一个ApplicationRuntime实时注入。旧7ch语料必须通过显式兼容导入或迁移生成缺失通道说明，不能静默冒充新8ch资产。

录制、查询、QA、hash、恢复和索引任务继续在后台执行，不得阻塞实时采集。音频只在本地保存，不自动上传；scratch录音与正式Catalog保持隔离。

当前启动方式保持不变：

```powershell
python scripts/run_audio_data_manager.py --data-root data
```

在v0.3代码、manifest schema和迁移测试完成前，该命令仍启动当前v0.2实现。
