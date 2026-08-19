from __future__ import annotations

from pathlib import Path

from common.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parallel_runtime_limits_are_loaded_from_the_single_config():
    runtime = load_config(PROJECT_ROOT / "config" / "config.yaml").runtime

    assert (runtime.l2_queue_windows, runtime.l3_queue_windows, runtime.l4_queue_windows) == (4, 3, 3)
    assert runtime.completion_queue_windows == 8
    assert runtime.max_inflight_windows == 16
    assert runtime.compute_cache_max_bytes == 64 * 1024 * 1024
    assert runtime.overflow_policy == "drop_oldest"
    assert runtime.graceful_shutdown_timeout_seconds == 10.0


def test_authoritative_architecture_documents_parallel_timeline_contract():
    architecture = (PROJECT_ROOT / "ARCHITECTURE_V0.3_TARGET.md").read_text(encoding="utf-8")

    assert "WindowKey = (session_id, stream_epoch, window_id, decision_sample)" in architecture
    assert "L2(n) || L3(n-1) || L4(n-2)" in architecture
    assert "同一窗口仍严格依赖" in architecture
    assert "ResultJoiner" in architecture
    assert "正常停机采用drain而不是清空队列" in architecture
    assert "compute_cache_max_bytes" in architecture
    assert "L2/L3/L4均`DROPPED`" in architecture
    assert "L3/L4为`DROPPED`" in architecture
    assert "L4为`DROPPED`" in architecture
    assert "已经开始的SRP、BF或CNN不被强制取消" in architecture
    assert "pre-joiner容量拒绝审计" in architecture
    assert "2*max_inflight_windows + 2*completion_queue_windows" in architecture
    assert "commit → L4 → L3 → L2" in architecture
    assert "FAILED/TIMED_OUT/DROPPED/CANCELLED" in architecture
    assert "完整输出但使用了声明的回退路径时为`degraded`" in architecture
    assert "latest_l4_dev_ui容量1" in architecture
    assert "L4即时显示是唯一例外且只属于UI side channel" in architecture
    assert "后续有序`DROPPED/SKIPPED`帧不得立即清除" in architecture
    assert "Gate已开启且空间响应有效、但没有候选峰时" in architecture


def test_recording_documentation_matches_bounded_atomic_writer_contract():
    architecture = (PROJECT_ROOT / "ARCHITECTURE_V0.3_TARGET.md").read_text(encoding="utf-8")
    recording = (PROJECT_ROOT / "data_management" / "README.md").read_text(encoding="utf-8")

    for text in (
        "append_result_with_watermark",
        "hotmaps.jsonl.partial",
        "chunk_asset_commit_<stem>.json",
        "enhanced_asset_commit.json",
        "quarantine",
    ):
        assert text in architecture
        assert text in recording
    assert "队列溢出时两者均不接纳" in architecture
    assert "满队列时两者均不入队" in recording
    assert "立即写出JSONL/NPZ/sidecar" in architecture
    assert "立即写出该chunk的JSONL/NPZ/sidecar" in recording
    assert "按2秒sample裁剪的音频环和结果pre-roll" in architecture
    assert "只保留最新2秒pre-roll" in recording
    assert "每个合并段只保留一条有界审计" in architecture
    assert "first_window_id/last_window_id" in recording
    assert "不保存逐窗触发列表" in recording
    assert "容量扫描只在新事件段开始前执行" in recording
    assert "从有界音频环补回间隙" in recording
    assert "result_queue_capacity`、`retention_days`和`max_storage_gb`必须大于0" in recording
    assert "`result_queue_capacity`默认256条" in recording
    assert "以256为硬上限" in recording
    assert "崩溃留下的open manifest" in recording
    assert "作为package data打入wheel" in recording


def test_historical_spec_no_longer_claims_only_l2_can_drop():
    specification = (
        PROJECT_ROOT / "CODEX_PROJECT_SPEC_6plus1_2D_voice_direction_v0.2.md"
    ).read_text(encoding="utf-8")

    assert "L2、L3、L4各自使用latest-wins" in specification
    assert "只有L2入口满时" not in specification
    assert "下游队列使用背压且不得静默删除" not in specification
    assert "append_result_with_watermark" in specification
    assert "latest_l4_dev_ui" in specification
    assert "只供右下象限即时显示，不是正式结果" in specification
    assert "不保存随50 Hz增长的逐窗ID列表" in specification
    assert "不得以零容量队列或非法存储预算启动" in specification
    assert "结果队列容量为256条" in specification
    assert "结果队列schema硬上限同样为256条" in specification
    assert "结果队列容量为4096条" not in specification
    assert "chunk_asset_commit_<stem>.json" in specification
    assert "把`config.yaml`作为package data打入wheel" in specification


def test_environment_documents_repeatable_cuda_stage_benchmark_scope():
    environment = (PROJECT_ROOT / "ENVIRONMENT.md").read_text(encoding="utf-8")

    assert "NVIDIA GeForce RTX 5060 Laptop GPU" in environment
    assert "PyTorch `2.12.1+cu132`" in environment
    assert "L3单候选avg/P95 `7.49/11.13 ms`" in environment
    assert "L3双候选的主要瓶颈是波束形成矩阵求解与条件检查" in environment
    assert "warm-cache逐窗stage计算" in environment
    assert "不等于端到端延迟" in environment


def test_test_ui_documentation_forbids_private_runtime_queue_access():
    ui_document = (PROJECT_ROOT / "gui" / "dev_test_ui" / "README.md").read_text(encoding="utf-8")

    assert "processing_status" in ui_document
    assert "不得访问`_processing_windows`" in ui_document
    assert "latest_l4_dev_ui" in ui_document
    assert "完整同窗L2/L3/L4" in ui_document
    assert "有序DROPPED/SKIPPED/缺失帧不立即清掉" in ui_document
    assert "最近1秒实际完成Hz" in ui_document
