# 6+1麦克风阵列 v1.4

v1.4 是实时方向定位精简版，只保留真实麦克风输入的 L1 与 L2。完整 L1～L6、录音管理、模型和示例音频保存在不可变标签 `v1.3.6`。

```text
Sipeed 8通道输入 -> 校准 -> L1 IMCRA（20 ms）
  -> P1逐麦概率 -> P2七麦中位数 -> 可选预降噪
  -> 160 ms窗口（每20 ms更新） -> L2 P2 Gate
  -> 2～4 kHz加权NormMUSIC -> IMM-JPDA方向ID
  -> 实时输出：时间、track_id、角度、状态
```

- 音频只在内存中保留最近10秒，不写WAV、不进入录音管理系统。
- `tmp/l2_track_history.txt`只保存ID、持续时间和稀疏轨迹。
- 性能监控只保留最近1秒，显示IMCRA、P、MUSIC、ID、总耗时和帧率。
- Test UI只接收真实麦克风；下半区L3～L6为空白预留。

首次创建精简环境：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
```

启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launch_dev_test_ui.ps1
```

VS Code固定使用`.venv-v1.4`，不安装PyTorch、CUDA、ONNX、CountNet或L4～L6依赖。`data/`、`tmp/`、录音、日志、缓存和虚拟环境不提交。旧系统从`v1.3.6`恢复。

旧架构与迁移说明见 [docs/V1.3.6_TO_V1.4_OVERVIEW_AND_USAGE.md](docs/V1.3.6_TO_V1.4_OVERVIEW_AND_USAGE.md)。
