# 独立 L1 Spectrum UI

本界面是与 Development Test UI 平行的 L1-only 观察工具。启动后只创建 UAC 麦克风输入、通道校准、
Ingest、IMCRA、可选 IMCRA 预降噪和 L1 电平/频谱分析，不创建 WindowAssembler、L2、L3、L4、
正式录音或数据管理服务。

## 启动

在项目根目录运行：

```powershell
.\scripts\launch_l1_spectrum_ui.ps1
```

界面会自动连接配置中的麦克风，也可以使用“连接麦克风/停止采集”重新连接。逻辑通道与项目现有
6+1契约一致：`MIC0`～`MIC5`为六个环形麦，`Center`是第七个物理麦，`Mix`是硬件混音；默认观察
`Center`，任一时刻只能选择一个通道。

## 四个区域

- 左上：L1状态、IMCRA预降噪开关、CDC灯光开关、八路20 ms RMS电平和观察通道选择。
- 右上：所选通道当前20 ms输入的2048点FFT柱状频谱，横轴固定0～10 kHz，每个新L1块刷新一次。
- 左下：所选物理麦的当前IMCRA噪声PSD、噪声/信号/SNR/SPP数值；`Mix`没有独立IMCRA估计。
- 右下：点击右上“抓拍到右下”时复制并冻结的输入频谱及权威sample/sequence标识。

当前输入频谱与IMCRA噪声谱都换算为便于对照的单频点dBFS显示；IMCRA顶部数值仍采用正式
`ImcraHopSnapshot.noise_features`原始统计口径。
