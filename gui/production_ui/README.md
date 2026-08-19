# 麦克风阵列录音与数据管理

权威目标见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../../ARCHITECTURE_V0.3_TARGET.md)。专用测试录音使用`raw_microphone_recording_v1`：录制时直接流式保存主机收到的原始8通道PCM16与每一帧CDC热力图，不在内存累计整段录音，也不为新录音生成7通道派生音频。

目标界面继续管理Runtime Sessions、Test Corpus、录制向导、质量与标注、系统维护及实验快照。新增内容包括：native/logical 8ch通道说明、HardwareMix标识、IMCRA/Gate时间线、原始SRP空间响应、L2平滑候选、逐方向L3音频和L4结果。内部追踪ID不显示、不检索也不写入资产。

操作者在开始录制前填写环境、数字形式的声源数量、每个声源各自的类型与移动方式，以及噪音来源；只录环境噪音时声源数量可以为0。系统使用环境、声源数量和录制时间自动生成列表显示名称，并把上述结构化信息随录音写入labels和manifest。录音列表只提供原始8通道任选通道试听、用所选录音进行模拟测试、移到可恢复回收站三类操作。“模拟测试”必须同时存在`native_8ch`与`cdc_hotmaps`资产，Test UI按原始相对时序注入两路输入；只有普通WAV的旧样本不能冒充完整麦克风阵列。

录制、查询、QA、hash、恢复和索引任务继续在后台执行，不得阻塞实时采集。音频只在本地保存，不自动上传；scratch录音与正式Catalog保持隔离。

当前启动方式保持不变：

```powershell
python scripts/run_audio_data_manager.py --data-root data
```

结束录音后，音频、热力图、环境、逐声源类型与移动方式、噪音来源、录制区间与资产哈希共同写入独立Recording目录并登记到Catalog。
