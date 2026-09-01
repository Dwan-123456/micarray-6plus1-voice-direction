# 05 ID合并、追踪与预测

本文属于**原理解释 + 技术参考**。MUSIC每次只给出瞬时峰，方向追踪器负责判断新峰属于已有轨迹、一个新轨迹或误报，并在短时无观测时维持方向ID。

## 1. `track_id`表示什么

`track_id`标识**空间方向随时间形成的轨迹**。它不代表人物、声纹或设备身份。

同一个人移动很快、长时间静音或被错误关联时可能获得新ID；不同的人若先后出现在接近方向，也可能被同一空间轨迹覆盖。后续人物级身份需要声纹或其他模态，v1.4.3没有该功能。

未发生显式ID重置时，ID在同一个`session_id`内单调递增。普通epoch变化会清空活动轨迹并保留该session的下一编号。用户关闭ID追踪后再开启会清空tracker和编号，新追踪段从1开始，因此同一session可在不同显式重置段中复用相同数字ID。

## 2. 为什么需要追踪

瞬时MUSIC峰会因噪声、反射、有限样本和取峰顺序变化而抖动。若直接把“本窗第一峰”称为ID 1，两个峰分数交换时ID也会交换。追踪器使用：

- 历史角度和角速度；
- 当前预测不确定度；
- 候选归一化分数；
- 多轨迹/多候选联合关联；
- 出生、确认、滑行和过期状态；
- 短时无观测预测。

## 3. 圆周角度

方向状态内部允许角度临时展开到实数轴，观测创新使用最短圆周差：

```text
delta(theta, reference)
  = ((theta-reference+180) mod 360) - 180
```

例如从`359°`移动到`1°`的创新为`+2°`。内部平均角周期性减去最接近的360°倍数，保持数值在原点附近；公开输出再取模到`[0,360)`。

## 4. 轨迹状态

| 状态 | 含义 | 是否有当前观测 |
| --- | --- | --- |
| `tentative` | 已出生但证据未达到正式门槛 | 可有或无 |
| `confirmed` | 达到观测次数和存在概率门槛 | 有 |
| `coasting` | 已confirmed，本窗没有匹配观测 | 无，输出来自预测 |

`TrackedDirection.is_observed`必须与`measured_theta_deg is not None`一致。confirmed轨迹失去观测后，公开状态变为coasting；未确认轨迹失去观测仍显示tentative，直到恢复或过期。

## 5. IMM状态模型

每条轨迹内部维护两个二维状态：

```text
x = [unwrapped angle, angular velocity]^T
```

两个模型分别描述：

| 模型 | angle process std | velocity std | velocity half-life |
| --- | ---: | ---: | ---: |
| 基本静止 | 0.35° | 3°/s | 0.15 s |
| 缓慢移动 | 1.25° | 15°/s | 0.50 s |

角速度限制为`±60°/s`。速度按半衰期指数衰减，离开观测后逐渐回到0，避免预测无限匀速漂移。

## 6. 模型交互

每次预测前，使用模型转移概率混合两个模型的均值和协方差：

```text
stationary -> moving = 0.02
moving -> stationary = 0.05
```

源模型到目标模型的混合权重来自上一模型概率与转移矩阵。每个目标模型独立预测，再根据观测似然更新模型后验概率。最终方向和协方差是两个模型的概率加权混合。

这就是IMM（Interacting Multiple Model）：同时保留“基本不动”和“正在移动”两种解释，而不在单帧硬切换。

## 7. 预测模型

角速度使用指数衰减。对时间间隔`dt`、半衰期`h`：

```text
gamma = 2^(-dt/h)
integrated = h/ln(2) * (1-gamma)

F = [[1, integrated],
     [0, gamma]]
```

预测：

```text
x^- = F x
P^- = F P F^T + Q
```

`Q`按模型的角度/速度标准差和`max(dt,20 ms)`构造。轨迹存在概率按每秒生存概率进行时间连续衰减，而非按“经过多少UI帧”衰减。

## 8. 观测似然和硬门控

对候选方向`z`，每个模型计算圆周创新：

```text
v = circular_delta(z, predicted_angle)
S = predicted_angle_variance + measurement_variance
measurement std = 5°
```

候选必须同时满足：

```text
|v| <= 50°
v^2 / S <= 20
```

合法时似然近似：

```text
L = exp(-0.5 * v^2/S) * max(0.05, candidate_score)
```

这里使用无量纲指数似然。若直接使用带`1/degree`量纲的高斯密度，5°测量噪声下合法观测可能反而比“虚警”先验更小。

## 9. JPDA联合假设

JPDA同时考虑所有活动轨迹和所有候选，不对每条轨迹独立贪心。联合事件满足：

- 每条轨迹最多匹配一个候选；
- 每个候选最多属于一条轨迹；
- 轨迹可以miss；
- 未使用候选可以是new或false。

当前先验：

| 参数 | 值 |
| --- | ---: |
| detection probability | 0.85 |
| existing-track prior | 0.80 |
| new-source prior | 0.10 |
| false-alarm prior | 0.10 |

对所有合法联合假设递归枚举权重并归一化，得到关联矩阵`beta[track,candidate]`、每个候选的`p_new`和`p_false`。当前最多4个内部轨迹、通常最多2个候选，使枚举保持有界。

## 10. 匈牙利一对一选择

JPDA概率用于软Kalman更新，但公开“当前哪个候选观测了哪条轨迹”需要一对一结果。程序对`-association_probability`运行Hungarian assignment，只接受概率`>=0.20`的组合。

这一步解决多个轨迹争夺同一观测；软更新仍使用整行JPDA概率，不只使用Hungarian选中的一个候选。

## 11. 救援关联

某候选可能因预测协方差、概率先验或突然角度变化而低于0.20。如果它位于已有轨迹预测角或最后观测角的50°范围内，直接出生新ID会形成近邻重复轨迹。

救援阶段对剩余轨迹和候选构造圆周角距成本，再用Hungarian做一对一匹配。合法救援会把对应JPDA行/列改为确定性1.0，并继续执行普通Kalman测量更新。公开角度仍是滤波后验，不会直接复制原始候选。

## 12. JPDA Kalman更新

每个模型对所有候选创新进行概率加权：

```text
detected = sum(beta)
missed = 1 - detected
K = P[:,angle] / S
weighted_innovation = sum(beta_j * innovation_j)
x^+ = x^- + K * weighted_innovation
```

协方差同时包含：

- miss时保留的预测协方差；
- detected时的Kalman收缩；
- 多候选创新分散造成的额外不确定度。

模型概率依据每个模型对候选的证据重新归一化。

## 13. 轨迹存在概率

tentative轨迹在miss时会按检测概率更新而快速降低存在概率；低于`0.05`可提前删除。confirmed轨迹在无观测时只保留时间生存衰减，不再叠加逐窗Bayesian miss坍缩，因此能够在两秒滑行租约内恢复。

有匹配时，观测关联概率和历史存在概率共同更新；confirmed轨迹的存在概率不会因一次正常观测下降。

## 14. 轨迹出生

未关联候选只有满足全部条件才出生：

- 当前MUSIC诊断允许出生；
- `p_new >= 0.45`；
- 离所有已有预测角和最后观测角超过50°；
- 活动轨迹未超过4，或可以淘汰一个最低存在概率的tentative轨迹。

新轨迹初始两个模型都以候选角、0角速度和同一协方差开始；初始模型概率为`0.75 stationary / 0.25 moving`，存在概率至少0.55。

MUSIC在Gate连续OPEN不足10 hop时`births_allowed=False`，已有轨迹仍可关联，新的峰不会建立ID。

## 15. 确认规则

默认要求：

```text
confirmation observations = 10
base confirmation window = 500 ms
existence probability >= 0.70
```

正常20 ms实算时，最近500 ms内10次真实观测即可confirmed。自适应周期变为60–200 ms时，固定500 ms不可能容纳10次真实实算，因此有效确认窗取：

```text
max(500 ms, (10-1) * current_observation_period)
```

复用输出不计为新观测，只延续当前时间轴状态。

## 16. TTL和两秒滑行

| 轨迹类型 | 无观测TTL |
| --- | ---: |
| tentative | 500 ms / 24,000 samples |
| confirmed/coasting | 2,000 ms / 96,000 samples |

以绝对sample计算寿命，不依赖wall-clock或UI刷新率。达到TTL即删除，持续复用也不能无限续命。

若无观测时混合角度标准差超过25°，公开预测角冻结在最后可信输出，避免高不确定度模型在界面上大范围漂移；内部状态仍继续预测。

## 17. 重复ID合并

救援仍可能来不及阻止两个邻近轨迹交替吸收同一真实源。程序只合并满足以下条件的轨迹：

- 混合预测角距离小于50°；
- 最近有效确认窗内两条轨迹都曾被观测；
- 两条轨迹的观测sample不重叠，说明它们不是同时存在的两个峰；
- 合并后的时间序列至少3个观测且所有权切换至少2次。

优先保留已confirmed、最早出生、存在概率更高、ID更小的轨迹。若较新重复轨迹含更新观测，保留ID会吸收其模型状态和分数。

同时被观测的两个邻近轨迹不会按这条规则合并。

## 18. 公开方向选择

追踪器内部最多4条活动轨迹。公开方向按以下优先级：

1. 当前观测到的confirmed；
2. 当前观测到的tentative；
3. 没有当前观测的coasting，按miss时间和分数排序。

公开结果最多3条并继续执行50°间距。由于当前MUSIC实际order为1或2，正常观测通常不超过2条；额外内部轨迹用于滑行和恢复。

## 19. ID追踪开关

关闭ID追踪时：

- MUSIC仍计算；
- UI显示原始峰；
- 不发布`TrackedDirection`；
- 现有tracker安排重置。

重新开启从新的ID状态和编号1开始。数字ID只在两次显式重置之间的连续追踪段内唯一；跨重置比较时必须把重置操作当作身份边界。若用户在两个L2窗口之间快速“关→开”，独立reset revision确保下一窗仍执行重置，不会因为最终开关又是ON而漏掉清空。

当前稀疏轨迹日志按数字`track_id`汇总，未单独写入显式重置段编号。同一session重置后若再次出现相同数字ID，日志内容可能落在同一ID项中；需要分析日志时，应把人工开关ID追踪的时刻作为分段边界。

## 20. Gate关闭和MUSIC跳过

Gate关闭时不产生新MUSIC观测，追踪器仍以空候选更新：

- confirmed进入coasting；
- tentative等待恢复或500 ms过期；
- 两秒后confirmed过期；
- 不出生新轨迹。

Gate打开但计数计划使MUSIC为0/None时也只允许预测。下一次正阶MUSIC必须重新建立连续OPEN上下文。

## 21. 自适应输出复用

Runtime过载时跳过完整ID实算，上一结果会被重新绑定到当前窗口身份：

- confirmed且未观测的公开状态改为coasting；
- `missed_samples`按当前decision sample重算；
- 达到tentative/coasting TTL的轨迹被过滤；
- Gate关闭时不复用候选；
- 当前Gate和当前声源数仍进入原子快照；
- `reused_output=True`和实际period进入诊断。

复用保持20 ms对外时钟，同时明确该窗没有新测量更新。

## 22. 输出DTO

`TrackedDirection`主要字段：

| 字段 | 含义 |
| --- | --- |
| `track_id` | session内一条方向轨迹ID |
| `rank` | 本窗内部输出顺序，从1开始 |
| `measured_theta_deg` | 当前原始观测；coasting为None |
| `theta_deg` | IMM-JPDA滤波/预测角 |
| `track_state` | tentative/confirmed/coasting |
| `is_observed` | 本窗是否有观测 |
| `is_new_track` | 本窗是否刚出生 |
| `first_seen_sample` | 首次出生sample |
| `last_observed_sample` | 最近观测sample |
| `missed_samples` | 当前sample减最近观测sample |
| `kalman_applied` | 当前方向使用状态估计器 |

DTO强制窗口身份、角度范围、状态/观测一致性和绝对sample寿命一致。

## 23. Tracker诊断

`TrackerDiagnostics`记录：

- 联合假设数量；
- 轨迹×候选关联概率矩阵；
- 各候选new/false概率；
- 每条轨迹存在概率；
- 每条轨迹stationary/moving模型概率；
- 本窗重复轨迹合并关系。

这些诊断用于开发和测试，不等于对现实人物身份的概率。

## 24. 参数速查

| 参数 | 默认值 |
| --- | ---: |
| association gate | 50° |
| association chi-square | 20 |
| measurement std | 5° |
| minimum association probability | 0.20 |
| minimum birth probability | 0.45 |
| confirmation observations | 10 |
| confirmation existence | 0.70 |
| deletion existence | 0.05 |
| tentative TTL | 500 ms |
| coasting TTL | 2,000 ms |
| max velocity | 60°/s |
| duplicate birth guard | 15°，实际取max(15°,50°) |
| prediction freeze std | 25° |
| max active tracks | 4 |

## 25. 代码和测试入口

| 内容 | 文件 |
| --- | --- |
| IMM-JPDA实现 | `layer2_source_detection/global_tracker.py` |
| Pipeline生命周期 | `layer2_source_detection/pipeline.py` |
| 公共方向DTO | `common/data_types.py` |
| Runtime复用/故障 | `app/runtime.py` |
| UI方向投影 | `gui/dev_test_ui/contracts.py`、`srp_panel.py` |
| 主追踪测试 | `tests/test_l2_music_tracking.py` |
| 救援/合并测试 | `tests/test_l2_tracker_rescue_association.py` |
| 自适应/TTL测试 | `tests/test_runtime_adaptive_rate.py` |

测试覆盖跨0°、候选rank交换、tentative确认、两秒coasting、长滑行重获原ID、概率降到底仍保活、JPDA一对一、最多4轨迹、出生抑制、50°救援、预测漂移时用最后观测救援、交替重复合并、同时峰不合并、动态确认窗和复用TTL。

## 26. 参考资料边界

- [Grondin等2022 ODAS论文](references/26_Grondin_2022_ODAS.pdf)介绍GCC-PHAT、Kalman/粒子追踪和嵌入式听觉工程，是背景参考。
- 当前项目采用Circular IMM-JPDA和项目特定救援/合并规则，ODAS不是这套实现的逐行算法来源。
- JPDA、IMM和匈牙利方法的专门原始论文仍需在[参考资料索引](references/README.md)补齐。

[上一章：声源数与NormMUSIC](04-source-counting-and-normmusic-doa.md) · [下一章：当前局限](06-limitations-and-open-issues.md) · [返回项目总导航](../../README.md)
