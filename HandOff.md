> **最高行动纲领见 /kg/迭代准则.md（2026-09-01 用户制定：分析先行→快速探究→胜出全量），所有迭代必须遵守。**

# 三项目统一进度交接文档（HANDOFF）
> 最后更新：2026-08-30 ｜ 用途：新会话直接读本文件即可继续代码优化与实验迭代
> 配套文档：`EXPERIMENT_LOG.md`（实时日志）、`results_contrec口径对比.md`（对照表）、`proposal_反思与改进计划.md`（反思提案）、三项目各自的 `README.md` 与 `ANALYSIS_差距来源.md`

## 1. 目标与总路线
**目标**：三个 from-scratch 的生成式推荐模型在 beauty-2014 同口径下 R@10/NDCG@10 **稳定大幅超过 RPG 官方**（Test R@10=0.0678 / N@10=0.0383，125ep），再扩展数据集（instruments/steam）并撰写三篇顶会论文。
**路线**：精度优先（预算对齐 125-250ep、长码本、混合解码、空间联合微调）→ 达标后再做投机提速（speculative 框架已实现并验证）。

## 2. 环境（新会话必读）
- **GPU**：2× RTX 4090 24GB。**Python**：`/kg/miniconda/envs/kgllm/bin/python`（torch 2.6+cu118）。
- **三项目**：`/kg/DiffSID`（离散扩散 SID，主力）、`/kg/SlotFlow`（连续空间流匹配）、`/kg/ARD-GR`（多级 long-SID + 残差流 + speculative）。
- **数据**（RPGData 加载器统一口径）：`/kg/RPG_KDD2025/cache/AmazonReviews2014/{Sports_and_Outdoors,Beauty}/processed/`（Beauty: 22,363 users/12,101 items/33 万交互）。contrec 口径：`/kg/ContRec/data/{beauty,game,lastfm,ml1m}`。**RPG 的 cache_dir 是 cwd 相对路径 `./cache`——跑 RPG 相关脚本必须 `cd /kg/RPG_KDD2025`**。
- **文本编码器**：`/kg/llmpath/sentence-t5-base`（768d）。RPG-beauty 的 `sent_emb` 已按其格式预生成。
- **RPG 官方 ckpt**（验证器/基线）：`/kg/RPG_KDD2025/ckpt/genrec_default-main.py_--dataset_AmazonReviews2014_--category=Beauty_--sent_emb_model=sentence-t5-base_--sent_emb_dim=768-Aug-30-2026_00-07-fdf651.pth`（Beauty）与 `...Sports_and_Outdoors-Aug-29-2026_11-41-d38778.pth`（Sports）。
- **禁止**：访问 `/kg/bd-rec`。

## 3. beauty-2014 同口径 125ep 对照表（当前完成态）
| 模型/解码 | R@10 | N@10 | NFE |
|---|---|---|---|
| **RPG 官方** | **0.0678** | **0.0383** | 1+ |
| DiffSID hier125 confidence2 | 0.0458 | 0.0259 | 2 |
| DiffSID K=16 PQ 125ep score | 0.0396 | 0.0181 | 1 |
| SlotFlow d128 125ep 2step_id | 0.0313 | 0.0150 | 2 |
| DiffSID hier125 score | 0.0398 | 0.0223 | 1 |
| ARD-GR speculative | =RPG（保证，rank_match_rate=1.0 全量验证） | — | 1+ |
contrec 口径参考：DiffSID hier125 score=0.0440（125ep）、SlotFlow=0.0299（100ep）、RPG 无对照（数据不同）。

## 4. 已证伪/已验证结论（不要重跑、不要重犯）
1. **PQ 布局证伪**（K=8/16、40/125ep 均劣于 hier 同预算）——hier+balance=last 是正轴。
2. **SlotFlow d256 证伪**（< d128）；**SlotFlow 预算饱和**（100ep→125ep 仅 +1%）。瓶颈=空间质量 → **联合微调已实现**（`--joint_epochs`）待跑。
3. **ARD-GR 粗码 AR 八配置证伪**（balanced/contiguous/semantic/V64/L4 × 数据集，c1_recall 1.4-15.8% 但端到端 0.0002-0.0037；探针 train 1.0/test 4.7%）→ 独立精度路线关闭，**定案 speculative 路线**。
4. **speculative 框架全量验证通过**（35,598 用户 rank_match_rate=1.0，error=None）——保证"输出构造性等于 RPG 穷举"。randint 越界（CUDA assert 根因）已修复。
5. **解码排序随目录规模反转**：小目录 hcpd2>confidence2，大目录+语义码 confidence2>score>hcpd2。
6. **密度规律**：交互/物品密度是 SID 类方法的命门（ml1m 300/物品→0.1383 超 ContRec 论文值；steam 168→0.0688；稀疏 10-30→0.01-0.03）。
7. **语义码有效**：balance=last 在 sports 上 +43%（0.0137→0.0196）。
8. **E4 负结果**：causal=bidirectional 注意力（DLM 收益不在注意力方向）。
9. **init_scale 修复 +132%**（SlotFlow 噪声尺度失配是主因，消融 2× 已闭环）。

## 5. 当前运行/池状态（新会话先用这些命令重新确认）
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$pid/cmdline | grep -oE "epochs [0-9]+|--d 768|--K 16|dataset [A-Za-z0-9/]+"; echo; done
ls /kg/scripts/pool/{pending,claimed,done}/
tail -2 /kg/logs/46b_diffsid_k16pq125_b256.log /kg/logs/diffsid_rpgbeauty_hier250.log /kg/logs/140b_diffsid_rpgbeauty_d768.log
```
- **在跑**：DiffSID d768/K=16 PQ/125ep（rpgbeauty，21.9GB，约 12-15h）——扩容主攻。
- **池 pending**：100（ARD-GR score40）/120 已完成、125（SlotFlow d128 125ep）已完成或接近、130（ARD-GR sem125）已完成。
- 结果统一读 `out/<数据集>/summary.json`（键：DiffSID=score/hcpd2/confidence2；SlotFlow=2step_id/2step_hybrid；ARD-GR=b8s2/b4s2/score）。

## 6. 作业池系统（双卡永不空转的机制）
- `/kg/scripts/pool/{pending,claimed,done,parked}`：job 文件按名字序被 worker 认领（mv 原子防重复）。
- worker 启动：`nohup bash /kg/scripts/pool/worker.sh 0 > /kg/logs/worker_gpu0.log 2>&1 &`（GPU1 同理；可选第二参数=哨兵文件路径，存在才开始认领）。
- **加实验 = 往 pending/ 丢 job 文件**（bash 脚本，绝对路径）。非 beauty 数据集任务在 `parked/`。
- 教训：worker 拓扑会因 pkill 交叉错乱——重启前先 `pgrep -af worker.sh` 全清，再按需重启；**pkill -f 会自匹配 harness wrapper 杀死自己的 shell**（已发生 3 次）——清理一律用 PID。

## 7. 差距迭代决策树（距 RPG-beauty 0.0678 还差 1.4×@125ep）
1. **d768 跑完后**：若 R@10 ≥0.055 → 预算再加（250ep rpgbeauty）+ 主表扩展；若 <0.05 → 空间联合微调（SlotFlow 已实现）迁移到 DiffSID（解耦 SPA/PCA128 空间接入 DiffSID——需小改）。
2. **score 解码增强**（已实现，eval-only 即可测）：`--score_levels entropy --score_pop_w 0.5`（层级熵加权 + 流行度先验）——rpgbeauty_hier125 ckpt 上直接重评（`--skip_train`）。
3. **K=16 hier 长码**：hier 布局深层退化（已证），仅在 d768/更密数据下重试。
4. **ARD-GR**：independent 路线关闭（探针证据）；**speculative 路线** = drafter v1（已训：`out/drafter_beauty.pt`，RPG 32×256 sem_ids 空间）+ `run_speculative.py --category Beauty --drafter_ckpt ...`（v1 接线已完成）→ 测 early-exit 率（提速指标；精度保证=与 RPG 逐位一致）。
5. **达标判据**：三模型同口径 R@10 与 N@10 均 > RPG（0.0678/0.0383）且 +20% 以上 → 扩展 instruments/steam → 撰写论文。

## 8. 已修 bug 与代码结构（重要）
- LightGCN 二部图转置索引 bug（影响所有协同信号，已修+重建缓存）；last_hidden 左填充索引 bug（ARD-GR，已修）；draft_v0 randint 越界（CUDA assert 根因，已修）；drafter 索引 1-based/0-based（已修）；score 解码 skip_train/out_dir 陷阱（out_dir 必须含 model.pt）；joint-ft 的 col_t/collab_t 引用（已修）。
- ARD-GR：多级 long-SID（`--levels`）+ `--cluster contiguous --coarse_source semantic` + `--decode_grid spec`（自投机解码：den 起草 + 精确验证 + 可证明 bound）全部实现。
- DiffSID：`--balance last`、`--layout pq`（证伪保留）、score 解码（`--score_levels entropy --score_pop_w`）、`--skip_train`。
- SlotFlow：`--init_scale 0.3`（关键修复）、`--sphere_init`、`--lam_mean/std`、`--eval_grid 2-id,2-hybrid,4-hybrid`、`--joint_epochs`、reflow 1-NFE 蒸馏。
- speculative：`src/speculative.py`（verifier+bound+fallback+draft_diffusion+load_drafter）、`src/run_speculative.py`（v0/v1 runner；**必须从 /kg/RPG_KDD2025 运行**——cache_dir cwd 相对）、`src/train_drafter.py`（K 动态适配 RPG 码本数）。

## 9. 论文规划（三篇）
1. **DiffSID**：语义扩散 SID + 双解码（score 1-NFE / 生成 2-NFE）+ 层级熵分析——当前 beauty-2014 0.0396-0.0458@125ep，需 d768/250ep 推过 0.0678。
2. **SlotFlow**：连续双信号空间 + 联合微调 + 修复后的 2-NFE——当前 0.0313，需联合微调/空间容量突破。
3. **ARD-GR**：speculative 保证框架（全量验证 ✓）+ 粗码 AR 可泛化性分析（负结果章节）——定位为"保证不劣 + 提速"的系统论文。
共享贡献：三层碰撞感知评估协议、密度规律、解码排序反转发现、八配置证伪方法论。

## 10. 遗留
- Gitee push：三仓库本地 commit 全部就绪，等用户私人令牌（remote: gitee.com/arikes3/{diff-sid,slot-flow,ard-gr}）。
- ml1m/game/lastfm 的 contrec 口径结果已有（见 results doc）；rpg 口径仅 beauty 完成。
- SlotFlow 联合微调、DiffSID d768/250ep 的结果回收后更新本文件与对照表。

## ===== 2026-08-30 20:10 v2 架构大改（新会话必读）=====
- **怪圈根因确诊**（详见 /kg/怪圈诊断与v2方案.md）：RPG 源码三大密码 = tied 归一化余弦头 + item 均值池化输入 + 32码并行 gathered-mean；且 RPG-beauty 基线用 sentence-t5-base（非 3-large）——编码器本来就一致，差距全在架构。调参必然饱和的原因找到了。
- **三项目 v2 已重构并上线**（commit c81fa2e/3c2f928/c10b01e）：ARD-GR=PQ16长码+RPG目标ctx塔+全目录残差头+球面流起点；SlotFlow=全目录8192负例InfoNCE空间；DiffSID=tied头+ens集成解码（3 NFE）。编码器统一 st5（用户指示），Qwen 仅探索 flag。
- **在跑**：GPU0 ARD-GR v2 pq16 beauty125（→DiffSID v2 tied k16pq 125）；GPU1 SlotFlow v2 neg8192 beauty125（→DiffSID ens_w 扫描）。日志 /kg/logs/v2_*.log。
- **探针结论**：SlotFlow 空间天花板 R@10≈0.019（bank .0186/fused .0156）——空间可分性是连续路线第一瓶颈，全目录负例正中靶心。
- **新脚本**：/kg/scripts/v2/{launch_gpu0,launch_gpu1,exp1..exp4}*.sh（探索均为 eval-only，可插空跑）。
- 新会话接续点：收 v2 三个 summary.json（out/v2_*/summary.json）对比 RPG 0.0678/0.0383；胜出方向加码（K32/集成权重/联合微调），未胜出读 exp* 日志定位。

## ===== 2026-08-31 12:45 v2.1/v2.2 状态（新会话接续点）=====
- **v2 复盘**：SlotFlow 全目录负例空间 hybrid R@10=0.0380（+21%）；ARD-GR pq16 码建在 PCA-128 上=随机哈希（score 0.005）——码质量是主凶（解码 z-norm 隔离实验证实）；DiffSID batch 256 OOM。
- **v2.1 已上线**：raw 896d OPQ+自然kmeans PQ（ARD-GR K16/K32 码已缓存, DiffSID --layout pq_raw 缓存好）、τ=1/0.07、z-norm/闭式矩校准解码（bound 仍可证明）、_cbs_ref 注册 bug 修复、max_len 50。
- **v2.2**：ARD-GR --w_pos 多位置监督（默认开）。
- **在跑/排队**：GPU0 ARD-GR v2.1 K16（~16:40 完）→watcher 自动接 K32+w_pos；GPU1 SlotFlow v2.1 多正例（~12:50 完）→watcher 自动接 DiffSID v2.1 pq_raw tied max_len50。
- 新会话动作：收 4 个 summary（out/v21_pqraw_beauty125、out/v21_multipos_beauty125、out/v21_pqrawK32_beauty125、out/v21_pqraw_tied_beauty125）对比 RPG 0.0678/0.0383；SlotFlow v2.1 预期 >0.038（多正例+id 2.6 收敛中）；ARD-GR 预期 ctx<3.0 且 score 大幅超 0.005。若 ARD-GR score 接近 RPG：上 ens 式集成（码通道+z-norm 残差）与 K32；若 DiffSID tied+pq_raw 超 0.0482：确认 tied 主线并推 beauty ens_w 扫描。

## ===== 2026-09-01 17:05 v2.2（新会话接续点）=====
- **两个决定性修复**：①ctx 组合 mask NaN 污染（右 padding + 单因果 mask），旧 ckpt 修复后 test 0.0023→0.0312；②残差指纹记忆过拟合（train 0.11/test 0.031），--res_scale 0 纯码输入。诊断工具 src/probe_ctx_ceiling.py。
- **在跑**：GPU0 ARD-GR v2.2 纯码+w_pos 125ep（out/v22_purecode_beauty125，预计 9/2 下午完）；GPU1 DiffSID v2.1 pq_raw tied K16 max_len50 batch64（out/v21_pqraw_tied_beauty125，预计 9/2 深夜完）。
- **SlotFlow 注意**：v2.1 多正例回退（0.025<0.038），SlotFlow 主线回到 v2 配置（--slot_pos_knn 0，neg8192）；hybrid 0.0380 是当前三项目最高。
- **当前记分板（beauty RPG口径 125ep，RPG=0.0678/0.0383）**：SlotFlow hybrid 0.0380 > ARD-GR v2.1(修NaN) 0.0312 > DiffSID v1 hier confidence2 0.0482(最高但码/头未升级) > ARD-GR v2.1(实测) 0.0023(NaN) > ARD-GR v2(pq碎码) 0.005。
- 新会话动作：①收 v22_purecode 与 v21_pqraw_tied 两个 summary；②若 ARD-GR 纯码 test>0.05：上 K32+集成；③SlotFlow 用 v2 配置 + max_len 50 重跑（v2 当时是 max_len 20，50 或再 +10-20%）；④三项目 ens/集成统一化。

## ===== 2026-09-01 23:15 v2.2 结果 + 信号分析（接续点）=====
- **ARD-GR v2.2 纯码 125ep：score R@10=0.0377/N@10=0.0198**（v2.1 修NaN后 0.0312 → +21%）；spec 0.0350（1 NFE）。记分板：DiffSID v1 0.0482 > ARD-GR v2.2 0.0377 ≈ SlotFlow 0.0380 > RPG 0.0678。
- **探针核心发现**：v2.2 train R@10=0.7264 / code-top1=0.546 vs test 0.0350/0.062——纯码+逐位置监督记忆更狠但泛化更好，**"记忆=敌人"假说被修正；train loss/R@10 与 test 严重脱钩，唯 test eval 可信**。
- **loss 信号定性结论**：ARD-GR 纯码 ctx 全程低于指纹版（指纹伤害优化本身）；DiffSID v2.1 ep24 rec 1.0-1.2/his 1.4/tok 0.45（旧码 rec≈5.0）——tied+自然 PQ 让映射好学一个量级（混杂 batch64/len50/τ，仅定性）。
- **在跑**：GPU0 SlotFlow v2+max_len50（out/v2_len50_beauty125 ~55min）；GPU1 DiffSID v2.1（ep24/125 ~21h）。
- **下一杠杆（按证据排序）**：①随机裁剪增强——固定滑窗+逐位置监督=记住全部训练对，SASRec 式随机 crop 是对症药（三项目共享 data.py）②ARD-GR K32 ③label smoothing ④DiffSID v2.1 完成后若超 0.0482 则 tied 主线确认。

## ===== 2026-09-02 07:40 K32 新纪录 0.0589 + 配方推广进行中 =====
- **ARD-GR K32 配方（crop+val早停+pop0.6+纯码+w_pos）全量 test: R@10=0.0589/N@10=0.0293**（R@10 距 RPG -13%）。
  码本冗余 K8→K16→K32 单调改善。w_res×pop 网格确认 (0.5,0.6) 已近最优。
- 配方推广: SlotFlow 完成 0.0388（+2%，该模型从配方获益少）；DiffSID v2.1 被判过训练已杀（ep57），
  GPU1 改跑配方短程 30ep（out/v23_recipe_beauty125，~15:40 出）。
- GPU0: ls_eps=0.1 短程探究进行中（对照 exp_crop15 0.0518）→ 完成后 GPU0 空闲。
- 决策记录: 早停协议下 125ep 预算不再必要（峰值 15-30ep），长跑一律配 val 早停 + 提前收割。

## ===== 2026-09-02 11:30 交接（用户接管实验）=====
- **当前最佳**：ARD-GR K32 配方 0.0589/0.0293（R@10 距 RPG -13%）。五发现+证伪记录见 /kg/0901_三项目代码阅读指南.md 的 0902 追加章。
- **在跑**：GPU0 ARD-GR K32+ls（val ep32=0.0500 已超无-ls 峰，收割命令 user_00）；GPU1 DiffSID 配方 30ep（val ~0.012，收敛慢，跑完直接换 user_02 的 90ep 版）。
- **用户后续实验 scripts**：/kg/scripts/v2/user_00_harvest.sh（通用收割）、user_01_next_experiments.sh（优先级清单）、user_02（DiffSID 90ep）、user_03（SlotFlow 重跑）。
- 代码优化收尾：SlotFlow quick_val dedup 修复已提交；三项目 val 早停/加载 best 均就绪。

## ===== 2026-09-05 23:05 v2.4b 状态（接续点）=====
- **记分板（beauty RPG口径）**：ARD-GR 融合 0.0597 / 单模 K32 0.0589 > DiffSID 0.0482 > SlotFlow **gmean 0.0444**（新）> SlotFlow flow 0.0388。RPG=0.0678/0.0383。
- **SlotFlow 双解码器达成**：flow 0.0388（2NFE）+ gmean 0.0444（1NFE，冻结 trunk+code-only-ft 20ep，out/v24_codeft_beauty125）。干扰训练版证伪。
- **在跑**：GPU0 dual-K48 60ep（val ep28=0.0466，~01:30 完）；GPU1 DiffSID 90ep 配方（ep~9，~20h，val 起步 0.003 极慢——若 ep30 val<0.02 考虑更长预算或放弃该线）。
- v2.4 全部证伪清单见 EXPERIMENT_LOG 2026-09-05 节。下一步候选：dual-K48 收割对比、DiffSID 配方判决、
  ARD-GR 融合部署化（三模型服务）。

## ===== 2026-09-06 01:05 v2.4 收敛判决（接续点）=====
- **结构性杠杆系统性扫尽并证伪**：dual-K48 0.0529、bank768 0.0465、res_deep、alpha0.5、解码端 5 项。
  **冠军：单模 K32 0.0589/0.0293；部署 fuse123 0.0597**。RPG=0.0678/0.0383（-12%）。
- **在跑**：GPU0 K32-d512-90ep（out/v24_k32d512_beauty125，~09:00 完，最后容量牌）；
  GPU1 DiffSID 90ep 配方（ep~11，val 0.0088 爬升慢——晚峰曲线属正常，判决点 ep30）。
- 判决规则：d512 val 峰 vs K32 的 0.0490；DiffSID ep30 val vs 0.02。

## ===== 2026-09-06 06:10 ≤3h 探索轮完结（新会话接续点）=====
- **工作规约（用户）**：只做 ≤3h 实验；3h 内指标好才延续。
- **最终记分板**：ARD-GR fuse123 0.0597 / 单模 K32 0.0589（-12%）；DiffSID 0.0482（-29%）；
  SlotFlow 双解码器 gmean 0.0463 + flow 0.0388（-32%）。RPG=0.0678/0.0383。目标未达成。
- **本轮证伪**：DiffSID 90ep 配方（3h 判负已杀，收敛慢 8 倍）、d512（3h 无优势）、TTA、
  bank768、res_deep、alpha0.5、slot 加权、rank 融合、den 视角、调温、K48、跨模型融合(+SF)。
- **有效留存**：fuse123 融合部署形态；SlotFlow 双解码器（gmean 0.0463 @1NFE 冻结 trunk 码头）；
  analyze_sid_length.py 与 SID 长度分析全部数据（/kg/SID长度深度分析.md）。
- 模型/码本/解码/正则的结构性空间已系统性扫尽（~30 实验）。剩余 -12% 差距的分析结论：
  单点杠杆已无，需要（a）与 RPG 作者核对 0.0678 的确切评估协议与 checkpoint 选择，
  （b）接受 >3h 的大算力训练（需用户特批），或（c）全新的建模想法。
- 关键坑位记录：test-pop 泄漏（pop 先验必须来自 train）；faiss 小数据集堆损坏；
  beam attention 显存 ∝ batch×32×(L·K)²。

## ===== 2026-09-06 12:15 最终状态（新会话接续点）=====
- **冠军不变**：ARD-GR 融合 0.0597 / 单模 0.0589；DiffSID 0.0482；SlotFlow gmean 0.0463。
- **配方空间闭合**：RPG 官方超参迁移证伪（0.0543<0.0589）；RPG 穷举实测 0.0686（差距在
  其数据管道/实现细节，非训练超参、非图传播、非解码）。
- **GPU 全空，无在跑任务**。所有资产见 EXPERIMENT_LOG 09-06 节。
- 若继续冲击：需要（a）重实现 RPG 的数据管道逐字节对齐（tokenizer.py 的具体切分）、
  （b）接受大算力长跑、或（c）新的建模想法。当前证据链完整可复现。

## ===== 2026-09-11 23:30 实验轮完结（接续点）=====
- **ARD-GR 单模冠军: slothead90-b1024 0.0632/0.0328（s43 复现 0.0613，稳健）**。距 RPG 0.0678 -6.8%。
- 本轮新增有效杠杆: per-slot ResBlock 头（+5.6%）、b1024+长训练、crop、val早停、pop/残差通道。
- 证伪: RPG-exact 超参迁移（两次）、dual-K48、bank768、TTA、res_deep、alpha0.5、K16-2L448 平台。
- **GPU 全空**。SlotFlow 最佳 0.0515（gate gmean）；DiffSID 冻结 0.0482。
- 层数结论: 6L 略优于 2L（2-12% 视配方），用户"小层更好"仅在 RPG 官方管道下成立。

## ===== 2026-09-07 03:05 GRPO v2 突破（接续点）=====
- **ARD-GR 单模新纪录: GRPO v2 = 0.0644/0.0338**（slothead90 基础 + GRPO 后训练：tau 1.1 采样 +
  CE mix 0.5 防遗忘 + 组相对优势）。距 RPG -5.0%/-11.7%。脚本: src/post_train_grpo.py。
- GRPO v1（tau 1.5 无 CE mix）= 0.0614（略降）——v2 的两个修正缺一不可。
- 在跑: GPU1 GRPO-long 6000 iters（~08:00 完+eval）；GPU0 wpos1-fixed 2L448 150ep（ep~30）。
- 用户洞察落实状态: ①隐式共现（crop_short）已实现待验证；②GRPO 后训练已验证有效 ✓。
- 距 RPG -5%: 下一批 = ①GRPO-long 收割 ②GRPO 与 crop_short 联合的基础重训+后训练
  ③SlotFlow/DiffSID 移植 GRPO（采样流/掩码轨迹的组相对优势）。

## ===== 2026-09-07 05:25 GRPO 时代（接续点）=====
- **ARD-GR 单模新高: GRPO-G16 = 0.0649/0.0340**（slothead90 基础 + GRPO 后训练）。
  GRPO 脚本: src/post_train_grpo.py；配置: tau 1.1 / CE-mix 0.5 / G16 / 3000it / lr 1e-5。
- 三项目记分板: ARD-GR 0.0649 > SlotFlow 0.0515 > DiffSID 0.0482（RPG 0.0678/0.0383）。
- 用户约束落实: 单模无融合 ✓ from scratch ✓ GRPO 后训练（允许非端到端）✓ 隐式共现（crop_short）已实现待验证。
- 距 RPG: ARD-GR -4.3%。SlotFlow 的 gmean 也用 GRPO 思路改进（采样流端点组）是下一步。

## GRPO 后训练完整配方（复现命令）
```bash
cd /kg/ARD-GR && python src/post_train_grpo.py \
  --ckpt out/v26_slothead90_b1024_beauty125/model_best.pt \
  --levels 32 --iters 3000 --G 16 --batch 128 --lr 1e-5 --tau 1.1 --ce_mix 0.5 \
  --out_dir out/grpo_g16
# 然后标准评估（--skip_train 加载 out/grpo_g16/model_grpo.pt → cp 为 model.pt）
```
关键超参: tau 1.1（低温采样近 argmax）、CE-mix 0.5（防遗忘）、G 16（组大小）、3000 iters、lr 1e-5。

## ===== 2026-09-12 10:10 两篇论文借鉴窗口（接续点）=====
**用户指定阅读并借鉴的两篇 2026 论文**（均读完原文）:
- EPIC (arXiv 2609.03522): SID 扩散推荐中显式 item 后验条件化 → 移植为 score_decode v3.0
  的"保留支持集 + item 后验边缘化 + clip log-残差反馈"（零训练解码时）。**全量证伪**:
  所有 β 配置 -3~-6%（自引用后验放大既有偏差；EPIC 真增益在其 transition field 学习模块,
  与"不加模块"约束冲突）。flags: --epic_M/--epic_beta/--epic_tau（默认关）。
- TGH (arXiv 2605.07125): 转移图启发式（末项→最频后继 top-k）无训练即解 Beauty 类基准;
  学习型模型只在 >3 跳难例上赢 → **shortcut 课程**: target ∉ 末项 top-10 后继的样本
  （hard-frac=0.329）采样加权 ×w（数据级,不注入模型,洞察 1 合规）。
  flags: --shortcut_curriculum W --shortcut_topk 10。
- **在跑对照实验**（b512 —— b1024 当前代码峰值 23.5GB 触 24G 卡 OOM 红线）:
  GPU0 对照 w=0 s46 → out/v30_shortcur_w0.0_s46_*; GPU1 课程 w=2.0 s47 → out/v30_shortcur_w2.0_s47_*。
  链: 70ep base (~2.6h, ~12:25 完) → GRPO-G16 2000it → 全量评估。脚本: scripts/v2/run_shortcur_chain.sh。
- **GRPO 移植已写好待跑**: SlotFlow src/post_train_grpo.py（作用于 gmean code-head,同 ARD-GR 机制）
  ; DiffSID src/post_train_grpo.py（全掩码态并行采样 G 组,REINFORCE+CE-mix）。
  下一窗口: 课程链收割后立即上双卡（各 ~30-40min）。
- 今日修的 bug: --skip_train 缺 ckpt 静默重训 20ep（幽灵 GPU 占用根源）→ 硬报错;
  链脚本 set -e + 产物守卫; 教训: 永不 sed 运行中的脚本/GPU 写死必须参数化。
- cropshort 收割: 0.0617/0.0316 < 冠军 0.0632（crop_short 不加分,不再追）。

## ===== 2026-09-12 15:10 Round-C 决策树（唤醒后执行）=====
- Round-A/B 全部收割后按此决策:
  1. ARD-GR valtune: 若 val-chosen 权重 test 上 > 0.5/0.6 → 同权重重评 GRPO ckpt
     (0.0649 可能直接上 0.066+); 若 ≤ → 权重维持, 协议修复完成。
  2. succmem n8/n16 vs control(0.0588): 若 ≥+0.002 → 剂量细化/种子复制 + 更长 GRPO;
     若 < → 关闭 TGH 注入线(与课程/EPIC 同一结论: 捷径已被内化, 注入只剩噪声)。
  3. DiffSID trajGRPO vs base(0.0482): 若 > → 调 G/tau/iters; 若 ≤ → DiffSID 保持 0.0482
     冻结, 剩余时间给其他项目。
  4. SlotFlow codeft-v2: 若 > 0.0515 → +GRPO 已排队自动完成; 记录新基线。
- ARD-GR 冲刺 RPG 的剩余候选(按先验排序):
  a. valtune 新权重 + GRPO ckpt 重评(零训练)
  b. GRPO 6000it + ada 门控(验证 AdaGRPO 抗崩溃主张; plain 6000it 曾从 0.0649 掉到 0.0643)
  c. 冠军种子复制(s43 var ±0.001)+GRPO → 择优
  d. 损失稀释消融(w_den=0/w_res_loss=0 纯 L_ctx+L_pos 60ep)——多任务混合可能稀释主目标
- 3072d RPG 嵌入不可得(无缓存文件), qwen 编码器为备用大动作(编码 12k item ~20min GPU)。

# ===== 2026-09-13 15:05 状态（新目标线） =====
- **在跑**：Round-J 4-SID ST5 试点（roundJ_4sid.sh，GPU0=full 栈 per_slot_scale+succmem8，GPU1=ctrl），
  流水线 = rq4+sum+rs0.15 基座 80ep → valtune(含 den_w 扩散通道) → GRPO-G16(码本族已对齐) → 终评
  (score/spec/b20s2 全模式 + ms/user)。看门狗 exec_1b73b032，报告 /kg/scripts/v2/rounds/roundJ_report.txt。
- **已交付**：/kg/LITERATURE_OPENALEX_2025.md（50 领域 + 20 跨领域，2025-01~2026-09，CS+词表过滤）。
- **bug 修复**：post_train_grpo.py 码本族感知（pilot2 rq4 的 GRPO 曾加载 PQ-4 码本，+1.7% 证据作废）。
- **禁令入档**：DiffGRM 评估不可信；其 t5_pca256 文件属 Industrial_and_Scientific 数据集，永久禁用。
- Round-I（GRPO6000+ada / champ s43 复制）被用户清空，未收割，结论待重跑。

# ===== 2026-09-13 22:15 夜间收口（4-SID 线） =====
- **4-SID + sentence-t5 最终：0.0609/0.0375**（s43 基座 + spec 解码 0.13ms/user），
  vs RPG 0.0678/0.0383（R@10 -10.2%，N@10 -2.1%）。起点 0.0511 → +19%。
- 复现命令链：roundK_4sid.sh（b1024/90ep full 栈训练）→ tune_decode_on_val --mode spec。
  ckpt: out/v47_rk4_s43_beauty125/model_best.pt。
- 杠杆全表与证伪记录：见 EXPERIMENT_LOG.md 2026-09-13 各节（V512 -9%、120ep -4%、
  GRPO 伤 spec、GRPO-den 锚 v4.8 部分成功、spec 旋钮饱和）。
- 文献交付：/kg/LITERATURE_OPENALEX_2025.md（50+20，2025-01~2026-09）。
- 待议结构方案：条件解码器为主体的 4-SID 模型族 / GRPO-den 加深 / 双塔候选联合重排。
- 两卡空闲。

# ===== 2026-09-14 01:50 最终收口（方案1/3 执行完毕） =====
- 方案1（CD4 条件解码器模型族）：两轮证伪（v1 纯链 0.032；v2 +多任务 0.032/beam 0.017）。
  根因：本基准信号在并行边缘+共现通道（TGH 零训练 0.0622），条件链天然弱——TIGER 式
  束搜在此 regime 不可用（累计 5 次独立失败）。代码保留 src/model_cd4.py 等（v5.0/v5.1）。
- 方案3（union 重排）：证伪（两塔分数不互补）。
- **4-SID + sentence-t5 最终：0.0609/0.0375**（ARD s43 + spec 0.13ms/user），
  距 RPG R@10 -10.2% / N@10 -2.1%；比 beam 快 ~170×。
- 32-SID 主线：ARD-GR 0.0656/0.0341 > SlotFlow 0.0542 > DiffSID 0.0482。
- 未测的后续方向：RPG 图传播解码移植（TGH 方向）/ 32-SID d768 长轴。两卡空闲。

# ===== 2026-09-19 04:20 CoLa-S 突破 =====
- **SlotFlow v4 (CoLa-S) 双种子超 RPG**：0.0766/0.0443 (s43), 0.0798/0.0442 (s46)。
  复现: cd /kg/SlotFlow && python src/train_v4.py --batch 1024 --epochs 90 --seed {43,46}
  --out_dir out/v4rs_colas_s{seed}_beauty125。判据协议与历轮一致。
- 证据链：Round-Q（生成式解码证伪+table塌缩诊断）→ Round-S（判别式连续读出）。
- 设计文档: /kg/SlotFlow/docs/CONTINUOUS_MOTIVATION.md；代码 v3(40c1a2c)/v4(c30c972)。

# ===== 2026-09-19 13:35 终局状态 =====
- SlotFlow/CoLa-S + OpenAI emb + GRPO：0.0825/0.0463，超论文口径判读 0.0809 的 R@10，
  N@10 平手；超 RPG 论文配置复现 +10%/+11.6%。
- 复现：/kg/SlotFlow train_v4.py → post_train_grpo_v4.py → eval_v4.py（命令见
  EXPERIMENT_LOG 2026-09-19 终局节）。ckpt: out/v4rv_grpo_ce3g8/。
- 协议对账：/kg/RPG_KDD2025/eval_sampled.py；论文数字判读 0.0809/0.0464（小数点位移）。
