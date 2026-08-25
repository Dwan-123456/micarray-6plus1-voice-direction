# L6整轨声纹识别、MOS择优与静音压缩

L6是每批L4、L5成功完成后自动运行的离线步骤，不参与实时采集，也不修改L2的
`track_id`和角度。Test UI的L4固定不合并，A/B两条候选及其逐20 ms L5标注
完整保留到L6；单人旁路作为该来源唯一A轨参与。

## 执行顺序

```text
每个L4来源的A整轨
  -> 无条件对完整16 kHz音轨提取CAMPPlus声纹

同来源B整轨
  -> |A匹配度-B匹配度| <= 0.20
  -> B匹配度 > 0.50
  -> B MOS > 0.30
  -> 三项同时满足时，提取完整声纹

全部入选整轨声纹
  -> 两两余弦相似度矩阵
  -> 平均链接AHC，阈值0.62，强制最多3个声纹
  -> 一个声纹关联一条或多条完整A/B音轨

逐个声纹
  -> 在本批录音开始至结束的绝对时间线上定位关联音轨
  -> 只填入L5判定为Voice的20 ms帧
  -> 按MOS从高到低处理，重叠帧保留MOS较高音频
  -> 无音频帧保持静音
  -> 删除全部首尾静音
  -> 内部静音不超过2秒时保留，超过2秒时缩短为2秒
  -> 按Speaker A/B/C声纹显示压缩后的16 kHz音频
```

## 声纹与音频关系

L6不再按1.5秒切片建立声纹。每条入选A/B完整音轨只提取一个192维归一化
CAMPPlus声纹，聚类后的`Speaker ID`作为关联键：

```text
Speaker A声纹 -> L2 ID 1的A轨、L2 ID 2的A轨、L2 ID 2的B轨
Speaker B声纹 -> L2 ID 3的A轨
```

`Layer6Result.fragments`中的每条记录现在代表一条完整L4音轨，而不是短片段；
它保存来源ID、A/B序号、整轨波形、L5标注、声纹、匹配度、MOS、与聚类中心
的相似度及最终Speaker ID。`metadata.voiceprint_audio_ids`显式记录一对多关系，
`pairwise_similarity_matrix`保留全部两两声纹相似度。

没有任何L5 Voice帧的音轨仍按上述规则提取声纹并参加两两计算；但仅含此类音轨
的声纹簇没有可填入的有效音频，因此不会生成空白展示行，审计ID记录在
`silent_voiceprint_audio_ids`。

## 时间线与MOS择优

绝对时间线只用于确定每条关联音频在原始录音中的位置。发生重叠时，L4已计算
的0～1 MOS是第一优先级；MOS相同才依次使用L4匹配度、声纹中心相似度和稳定
的A/B/资产ID顺序打破平局。旧版L6的Voice/声纹/DNSMOS/SNR/连续性五项综合
Q分不再参与选择，L6也不重复运行DNSMOS。

静音压缩发生在MOS择优和绝对时间拼接之后。因此输出波形保留讲话顺序及2秒内
停顿，但不再与原录音等长；原始录音起止sample仍保存在输出边界与结果元数据中
供审计。

## 模型和设备

- 声纹：`iic/speech_campplus_sv_zh_en_16k-common_advanced`，CPU，16 kHz，
  80-bin Kaldi fbank，192维归一化声纹。
- MOS：复用L4已生成的DNSMOS综合分，L6不加载或重复计算DNSMOS。
- L6在每批L4/L5完成后自动加载CAMPPlus并执行；重复运行L4会重跑并替换上一批L6结果。

## 公共接口

- `ApplicationRuntime.build_offline_l6_pipeline()`：构建CPU L6，只加载CAMPPlus。
- `OfflineLayer6Pipeline.process(tuple[Layer4OfflineResult, ...])`：返回
  `Layer6Result`。
- `Layer6Result.outputs`：0～3条按声纹显示的`Layer6SpeakerAudio`，包含压缩后
  16 kHz波形、原始录音边界、来源L2 ID、关联音轨ID和平均MOS。
- `Layer6Result.fragments`：每条入选完整A/B音轨到最终声纹的审计关系。
