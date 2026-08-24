# 麦克风阵列录音与数据管理

本目录已纳入项目1.3.2。运行录音详情使用公共`(session_id, stream_epoch, track_id)`，显示轨迹时间线摘要并提供逐ID连续补偿音频试听；该音频与Test UI及L5 CNN输入一致，重叠L3原始窗不再重复保存。旧v3和L1-only录音明确不含算法ID。

权威目标见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../../ARCHITECTURE_V0.3_TARGET.md)。专用测试录音使用`raw_microphone_recording_v1`：录制时直接流式保存主机收到的原始8通道PCM16与每一帧CDC热力图，不在内存累计整段录音，也不为新录音生成7通道派生音频。

界面继续管理Runtime Sessions、Test Corpus、录制向导、质量与标注、系统维护及实验快照。运行录音详情显示方向ID、首末sample、持续时间、首末角与角度变化、状态和最新L5概率；可试听逐ID增强时间线、Center参考，以及native/logical/physical任一通道。逐ID试听只继承正式公共ID，不按角度创建或合并ID。

操作者在开始录制前填写环境、数字形式的声源数量、每个声源各自的类型与移动方式，以及噪音来源；只录环境噪音时声源数量可以为0。系统按“环境 · 月日-时分 · 声源数 · 各声源类型（移动方式） · 噪音来源”自动生成列表显示名称，例如“会议室 · 0820-1029 · 2个声源 · 声源1：人声（移动）；声源2：人声（静止） · 噪音：风扇”，并把上述结构化信息随录音写入labels和manifest。录音列表支持原始8通道任选通道试听、使用所选录音进行模拟测试、修改所选名称及移到可恢复回收站；手动改名同步更新labels、manifest、资产哈希、Catalog和审计记录，但不改变Recording UUID、目录或音频资产。“模拟测试”只要求并读取`native_8ch`资产，Test UI不打开、不校验也不注入已录制的`cdc_hotmaps`；Test UI成功启动后，本数据管理窗口自动最小化。热力图资产仍按录音规范保存，供归档和其他离线用途使用。

录制、查询、QA、hash、恢复和索引任务继续在后台执行，不得阻塞实时采集。音频只在本地保存，不自动上传；scratch录音与正式Catalog保持隔离。

当前启动方式保持不变：

```powershell
python scripts/run_audio_data_manager.py --data-root data
```

结束录音后，音频、热力图、环境、逐声源类型与移动方式、噪音来源、录制区间与资产哈希共同写入独立Recording目录并登记到Catalog。

专用测试录音只保存L1原始输入和热力图，不运行方向算法；向导和manifest固定标明“无算法方向ID”。模拟测试后由Test UI重新运行算法产生的ID不回写或冒充原始录音ID。
