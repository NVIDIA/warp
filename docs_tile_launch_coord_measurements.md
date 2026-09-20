# Tiled CUDA launch coordinate reconstruction (#1361): full measurement record

## 跨架构复验：RTX 5090 / sm_120 / CUDA 13.0

同一补丁（`3afeb766937b0e151a8d74eaee62fcf660492396`）在第二台机器上复验：
一个 8× RTX 5090 节点（sm_120，CUDA 13.0.88，驱动 580.82.07），解释器为既有的
Python 3.12.13 venv（torch 2.14.0+cu130）。基线 commit 为
`a11346e`（同一上游 main SHA `015e5a1` 的快照）。

### E2E：三段 tiled pipeline

| workload | baseline (ms/pipeline) | patch (ms/pipeline) | speedup | arms | max spread |
|---|---:|---:|---:|---:|---:|
| n=65536, block_dim=256, max_blocks=256 | 0.01970 | 0.01970 | **0.9999x** | 2+2 | 0.13% |
| n=262144, block_dim=256, max_blocks=256 | 0.07192 | 0.07190 | **1.0003x** | 2+2 | 0.07% |
| n=262144, block_dim=256, max_blocks=32 | 0.54875 | 0.54864 | **1.0002x** | 2+2 | 0.02% |
| n=1048576, block_dim=256, max_blocks=256 | 0.28079 | 0.28040 | **1.0014x** | 2+2 | 0.64% |

**geometric-mean speedup：1.0005x**（逐负载 0.9999x .. 1.0014x，4/4 有效）。

### Micro：坐标计算隔离

| workload | baseline (ms/launch) | patch (ms/launch) | speedup | arms | max spread |
|---|---:|---:|---:|---:|---:|
| n_tiles=2048, max_blocks=128 | 0.00910 | 0.00910 | **1.0001x** | 2+2 | 0.35% |
| n_tiles=16384, max_blocks=128 | 0.06259 | 0.06261 | **0.9996x** | 2+2 | 0.08% |
| n_tiles=16384, max_blocks=32 | 0.24609 | 0.24609 | **1.0000x** | 2+2 | 0.02% |

**geometric-mean speedup：0.9999x**（逐负载 0.9996x .. 1.0001x，3/3 有效）。

### 正确性（sm_120）

| 套件 | baseline | patch |
|---|---|---|
| `test_tile_launch_coord.py`（44 CUDA 用例） | 44 / 0 | **44 / 0** |
| `*tile*.py`（35 套件 / 670 用例） | 502 ok / 6 fail | **502 ok / 6 fail** |

两树的失败集合完全相同，且都是与本补丁无关的 sm_120 既有失败：
`test_register_tile_cpu_blocks`、`test_register_tile_oob_reports_tile_index`、
`test_shared_tile_2d_oob_reports_dimension`、`test_shared_tile_negative_oob_reports_tile_index`、
`test_shared_tile_oob_reports_tile_index`、`test_thread_tile_uses_logical_block_dimension`。

> 结论与 L20 一致：sm_120 上也没有可测量收益（1.0005x / 0.9999x，均在噪声内）。
> 补丁在两种架构上都不改变正确性。

---

### G5：Clean PR validation（RTX 5090，2026-09-20）

| 项 | 值 |
|---|---|
| 补丁 SHA | `3afeb766937b0e151a8d74eaee62fcf660492396` |
| clean worktree HEAD | `3afeb766937b0e151a8d74eaee62fcf660492396`（`git worktree add --detach`，tracked 文件无改动） |
| 编译产物 | 全新编译的 `warp.so`，md5 `2e043b696f5cfd647e3c1e142e5bdbd7`，**与增量构建的 `1cd64a452a751f93fab1f9f31243d2f1` 不同**（全新链接，非复用旧库） |
| 解释器 | Python 3.12.13 |
| 设备 | GPU 5（`RTX 5090`，本轮独占空闲） |

正确性（全新进程，未继承任何先前 import / 缓存状态）：

| 套件 | 结果 |
|---|---|
| `test_tile_launch_coord.py` | **44 / 0** |
| `*tile*.py`（35 套件） | **502 ok / 6 fail** |
| 与 baseline 的失败集合比对 | **IDENTICAL_FAILURE_SET**（6 项均为 sm_120 既有失败，与本补丁无关） |

性能（全新进程、BPPB 交替）：

| workload | baseline (ms/pipeline) | patch (ms/pipeline) | speedup | max spread |
|---|---:|---:|---:|---:|
| n=262144, block_dim=256, max_blocks=256 | 0.07206 | 0.07206 | **1.0001x** | 0.48% |
| n=1048576, block_dim=256, max_blocks=256 | 0.28078 | 0.28034 | **1.0016x** | 0.04% |

| workload | baseline (ms/launch) | patch (ms/launch) | speedup | max spread |
|---|---:|---:|---:|---:|
| n_tiles=16384, max_blocks=128 | 0.06258 | 0.06259 | **1.0000x** | 0.13% |

**G5 geometric-mean：E2E 1.0008x、micro 1.0000x** —— 与 L20 和 5090 主测量结论一致：
补丁在 clean 环境下同样没有可测量收益，且不改变正确性。

## 进度条

> 本节是修复缺陷当时的状态快照，保留作过程记录。文中"未完成/未测"的条目后来都已补齐，
> 最终结果见上文「跨架构复验」与「最终提升速度对比」两节。

```
总进度   ████████████████████████████  100%   23/23 项
```

### 阶段进度

| 阶段 | 进度 | 状态 |
|---|---|---|
| G0 固定基线 / 取上游快照 | `██████████` 100% | 完成（main `015e5a1` → 本地 `3601222ef`） |
| G1 Baseline / 复现 | `██████████` 100% | 完成（两树独立构建 + 环境清单） |
| G2 Minimal implementation | `██████████` 100% | 完成；`ff0de67` 含缺陷，已修复并重建 |
| G3 GPU correctness | `██████████` 100% | **已通过**：修复后 `test_tile.py` 157/157、`*tile*.py` 1080/1080，与 baseline 完全一致 |
| G4 Performance screening | `██████████` 100% | **已重测**：修复构建上 E2E 3 档 + micro 2 档有效，全部 1.0000x |
| G5 Clean validation | `██████████` 100% | **完成**：全新 detached worktree（PATCH_SHA）+ 全新编译 + 全新进程 A/B |
| **P0 缺陷修复** | `██████████` 100% | 根因定位 → 数值验证 → 应用 → 重建 → **重测通过** |

### 硬性要求

| 要求 | 进度 | 说明 |
|---|---|---|
| 跑完整 E2E | `██████████` 100% | 3 段 tiled pipeline，graph replay，事件计时 |
| skill 不提供 idea，只走流程 | `██████████` 100% | 三阶段/配对/证据规范均按 skill 执行 |
| test result 用提升速度对比 md 表格 | `██████████` 100% | §3、§4 两张表，含排除项与 spread |
| 提交者 0z5a、不提 AI assist | `██████████` 100% | author/committer 均为 `0z5a`，已 grep 核验 |
| 不写冗长/保守代码 | `██████████` 100% | 无多余 try/except、无 `any`、无 `getattr` 读已存在字段 |

### 待办进度

| 优先级 | 项 | 进度 |
|---|---|---|
| P0 | 定位二维折叠轴回归根因 | `██████████` 完成（`coord_mult` 被当成 `shape[1]`） |
| P0 | 数值验证修复形式 | `██████████` 完成（9354 组合 0 失配） |
| P0 | 应用修复并重建 | `██████████` 完成 |
| P0 | 重跑 `test_tile.py`，回到 baseline 的 157/0 | `██████████` 完成（157/0） |
| P0 | 重跑 `*tile*.py` 全量 + 新增套件 | `██████████` 完成（1080/0） |
| P0 | 提交 PR | `██████████` 完成（[#1976](https://github.com/NVIDIA/warp/pull/1976)） |
| P1 | 用修好的构建重测 E2E / micro 并更新 §3、§4 | `██████████` 完成（`raw-final/` 28 arm） |
| P1 | 重新界定"生成源码逐字节相同"判据 | `██████████` 完成（更正横幅已标注该判据无效） |
| P1 | 给 `test_tile_launch_coord.py` 补二维 tile + 梯度场景 | `██████████` 完成（88 用例；负面对照 8/8 失败） |
| P2 | 全量 tile 套件跑进 1h 上限 | `██████████` 完成（1080/0，未触上限） |
| P2 | 非 sm_89 架构验证 | `██████████` 完成（RTX 5090 / sm_120 / CUDA 13.0 全量复验） |
| P2 | G5 全新 detached worktree 复验 | `██████████` 完成（全新编译 + 全新进程 A/B） |
| P2 | 重测期间主机稳定性 | `██████████` 完成（换至 5090；L20 已拒连，结论不依赖它） |
| P2 | `n_tiles=65536, max_blocks=8` 一档 | `██████████` 完成（独占 GPU 2，spread 0.05%，1.0000x） |


---


## 最终提升速度对比（修复后的构建）

补丁 commit **`30fc6865e3d691fd90ed1c3d12ae7b0558a0c71b`**（author/committer 均为 `0z5a`），
diff 见 `a2_1361_tile_coord_fixed.diff`。计时用 `enable_timing=True` 的 CUDA event 包住 replay
窗口，同步点在窗口外；每 arm 独立进程，BPPB 交替顺序。

### E2E：三段 tiled pipeline，CUDA graph replay

| workload | baseline median (ms/pipeline) | patch median (ms/pipeline) | speedup baseline/patch | arms | max spread |
|---|---:|---:|---:|---:|---:|
| n=1048576, block_dim=256, max_blocks=256 | 0.35834 | 0.35833 | **1.0000x** | 2+2 | 0.03% |
| n=262144, block_dim=256, max_blocks=256 | 0.09209 | 0.09210 | **1.0000x** | 2+2 | 0.10% |
| n=262144, block_dim=256, max_blocks=32 | 0.61730 | 0.61728 | **1.0000x** | 2+2 | 3.40% |

排除：`n=4194304, block_dim=64, max_blocks=1024`（`INVALID_SPREAD`，1 arm）。
**geometric-mean speedup：1.0000x**（逐负载 1.0000x .. 1.0000x）。

### Micro：最小 tiled kernel，坐标计算隔离

| workload | baseline median (ms/launch) | patch median (ms/launch) | speedup baseline/patch | arms | max spread |
|---|---:|---:|---:|---:|---:|
| n_tiles=16384, max_blocks=128 | 0.13635 | 0.13636 | **0.9999x** | 2+2 | 0.50% |
| n_tiles=2048, max_blocks=128 | 0.01897 | 0.01897 | **1.0001x** | 2+2 | 0.36% |

排除：`n_tiles=16384, max_blocks=32`（`INVALID_SPREAD`，1 arm）。
**geometric-mean speedup：1.0000x**（逐负载 0.9999x .. 1.0001x）。

> 结论：修复后的补丁**仍然没有可测量的性能收益**。生成代码层面的原因未在本轮重新验证
> （上一轮的"SASS 相同"判据已作废），因此这里只报告测得的 0.9999x–1.0001x，
> 不声称"编译器已优化掉"这一解释。

### 正确性（修复后，同卡无并发）

| 套件 | baseline | patch |
|---|---|---|
| `warp/tests/tile/test_tile.py` | 157 / 0 | **157 / 0** |
| `warp/tests/tile/*`（34 套件） | 1080 / 0 | **1080 / 0** |
| `test_tile_launch_coord.py`（新增 88 用例） | — | **88 / 0** |
| 负面对照：新增用例在 buggy 构建上 | — | **8 / 8 FAILED** |

---

> ## ✅ 缺陷已修复（进度更新）
>
> 根因：`launch_coord_tile()` 把 `coord_mult` 当成 `shape[1]` 来拆轴，并把 tile 内 lane
> 直接加到 `coord.j` 上。lane 的量级是 `coord_mult`（如 64），远超 `shape[1]`，
> 于是二维 tile 的行号越界、tile 之间互相别名。
>
> 修正后先把 tile 坐标重新卷回线性索引，再用原始 `wp::launch_coord()` 拆一次：
>
> ```cpp
> const size_t linear_tile = linear / bounds.coord_mult;
> const size_t lane_in_tile = linear % bounds.coord_mult;
> // ... unravel coord from linear_tile ...
> size_t folded = coord.i;                       // re-wrap the tile coord
> if constexpr (N > 1) folded = folded * bounds.shape[1] + coord.j;
> if constexpr (N > 2) folded = folded * bounds.shape[2] + coord.k;
> if constexpr (N > 3) folded = folded * bounds.shape[3] + coord.l;
> return launch_coord(folded * bounds.coord_mult + lane_in_tile, bounds);
> ```
>
> 验证（同一张卡、无并发、逐树顺序）：
>
> | 检查 | baseline | 修复前 patch | 修复后 patch |
> |---|---|---|---|
> | 解析式等价性（9354 组合） | — | 315/350 失配（2D 样例） | **0 失配** |
> | `test_tile.py` | 157 / 0 | 68 ok + 6 FAIL + 挂住 | **157 / 0** |
> | `*tile*.py` 全量 | 1080 / 0 | 未跑完 | **1080 / 0** |
>
> **遗留**：修复构建上的 E2E/micro 计时表未取全 —— 主机 load 从 24 涨到 452，
> 样本 spread 超过 5% 阈值被规则排除，随后主机 SSH 拒连。该项仍属未完成。

> ## ⚠️ 更正（同日，晚于下文首次成文）
>
> **下文 §1 与 §5 关于"SASS 完全相同 / 无需修改"的结论是错的，已被后续实验证伪。**
>
> 在干净 GPU 上把 `warp/tests/tile/test_tile.py` 逐树顺序各跑一遍（同一张卡、无并发）：
>
> | 树 | 结果 |
> |---|---|
> | baseline `3601222ef` | **157 passed / 0 failed** |
> | patch `ff0de674a` | **68 passed / 6 FAILED，随后挂住**（两次独立复现，非环境噪声） |
>
> 直接在 patch 树上调用失败用例，拿到真实断言：`test_tile_binary_map` 输出
> **1015 / 1120 个元素为 0**（应为有效值）——这是**真实的功能性回归**，
> 不是计时噪声，也不是设备争用。
>
> 失败集合（稳定复现）：`test_scalar_div_tile_cuda_0`、`test_tile_binary_map_cuda_0`、
> `test_tile_binary_map_mixed_types_cuda_0`、`test_tile_copy_2d_cuda_0`、
> `test_tile_div_elementwise_cuda_0`、`test_tile_div_scalar_cuda_0`。
> 共同点：**二维 tile 启动**（`dim=[M/TILE_M, N/TILE_N]` + `block_dim=TILE_DIM`，
> 即 `i, j = wp.tid()` 且折叠轴 `coord_mult=TILE_DIM`），全部带 `requires_grad=True` 的梯度校验。
>
> **因此：**
> - 本补丁**不可提交**，须先修掉二维折叠轴的坐标回归；
> - 我此前"两 build 生成源码逐字节相同"的判据**无效**——那只能说明两个树的 codegen
>   在纯 Python 层面一致，**不能**证明原生代码路径（`warp.so` / PCH / 实际编译产物）一致；
>   用它去推断"运行时行为相同"是错误推理；
> - E2E/micro 的计时表（§3、§4）本身仍有效，但它们测的是**错误的构建**，
>   不能作为该补丁的收益证据；
> - 远程主机在收尾阶段失联，上述修正只有本地与远端日志，**未完成**修复与重测。


**提交者：** 0z5a · 2026-09-19
**主机：** 8× NVIDIA L20 (sm_89) · CUDA 12.8.61 · Warp 1.19.0.dev0
**上游基线：** `NVIDIA/warp` main @ `015e5a17827d4409782e81026d3de6f61b9ae1a7`（本地基线 commit `3601222efec4b49b20880909d7143c8561a5e7bd`）
**补丁 commit：** `ff0de674a5b144ba949193d2fe2e3ae2b81131aa`（author `0z5a <Dezhen.lu@student.uni-tuebingen.de>`）

> `github.com` 在本机不可达，因此上游快照经 `codeload.github.com` 取得后在本地建仓，
> `BASE_SHA` 是本地 commit，其内容等于上游 main 的该 SHA；核对方式见"来源可追溯"一节。

---

## 1. 结论（先说结果）

| 问题 | 答案 |
|---|---|
| 补丁是否让 tiled launch 更快？ | **没有。可测范围内 0 提升。** |
| 为什么？ | 补丁前后 **SASS 完全相同**——ptxas 已经把这段坐标计算优化到最优，源码写法不影响生成代码。 |
| 那这个改动还有价值吗？ | 只有代码可读性价值，**没有性能价值**；不建议以"性能优化"名义提交。 |
| 建议动作 | 报告结论为 `INCONCLUSIVE`（无收益），不作为加速 PR 提交。详见 §6。 |

**关键证据（按强度排序）：**

1. 两个 build 为同一组 tiled launch 生成的 **CUDA 源码逐字节相同**（`cmp` 无差异），SASS 与寄存器用量也逐项相同（§5）；
2. E2E 4 档有效负载与 micro 3 档有效负载，事件计时下的 geometric-mean speedup 分别是 `1.0002x` 与 `1.0001x`（§3、§4）；
3. 新增 72 个坐标映射用例全绿，`test_launch.py` 42/42、`test_array.py` 147/147 通过。

---

## 2. 改了什么

`wp.launch_tiled(kernel, dim=[N], block_dim=B)` 会把 `B` 作为尾部维度追加，`_build_launch_bounds_from_tuple()`
再把它折叠进 `launch_bounds_t.coord_mult`。因此 `wp.tid()` 必须先经 `launch_coord()` 把线性线程索引中
block 派生的那部分除以 `coord_mult`。这一折叠是冗余的：tiled launch 每 block 持有整数个 tile
（`blockDim.x` 是 `coord_mult` 的整数倍），tile 索引完全由 `blockIdx.x` 决定。

补丁新增 `wp::launch_coord_tile()`：从 tile 索引直接构造 coord，再把 tile 内 lane 从 `threadIdx.x` 折回，
对每个线程与 `launch_coord()` 结果一致；`builtin_tid*` 宏按 `dim.coord_mult` 二选一，
没有折叠轴的 launch 保持原表达式，CPU 路径不传 lane（无 `threadIdx`）。

```
 warp/_src/codegen.py   | 12 +++++++----
 warp/native/builtin.h  | 36 ++++++++++++++++++++++++++++++++++++
 2 files changed, 44 insertions(+), 4 deletions(-)
```

---

## 3. E2E：三段 tiled pipeline，CUDA graph replay（事件计时）

工作负载：`tile_load → scale → +1 → tile_sum → store` 三段互相依赖的 `wp.launch_tiled`，
一个 graph 内含 8 次 pipeline，`--launches-per-sample 100`，取 15 个样本的中位数。
计时用 `enable_timing=True` 的 CUDA event 包住 replay 窗口，同步点在窗口之外。

| workload | baseline median (ms/pipeline) | patch median (ms/pipeline) | speedup baseline/patch | baseline arms | patch arms | max spread |
|---|---:|---:|---:|---:|---:|---:|
| n=1048576, block_dim=128, max_blocks=512 | 0.13315 | 0.13313 | **1.0002x** | 3 | 3 | 0.21% |
| n=1048576, block_dim=256, max_blocks=256 | 0.26021 | 0.26021 | **1.0000x** | 4 | 4 | 0.12% |
| n=262144, block_dim=256, max_blocks=256 | 0.06751 | 0.06748 | **1.0004x** | 5 | 5 | 0.43% |
| n=262144, block_dim=256, max_blocks=32 | 0.48396 | 0.48392 | **1.0001x** | 2 | 2 | 0.06% |

排除的工作负载（统一按 arm 内 spread 判定，对两个 build 一致，不按结果好坏挑选）：

| workload | 原因 | 排除 arm 数 |
|---|---|---:|
| n=4194304, block_dim=64, max_blocks=1024 | `INVALID_SPREAD` | 1 |
| n=65536, block_dim=256, max_blocks=256 | `INVALID_SPREAD` | 1 |

- 有效工作负载：**4**
- **geometric-mean speedup (baseline/patch)：1.0002x**
- 逐负载 speedup：1.0000x .. 1.0004x

---

## 4. Micro：最小 tiled kernel，坐标计算隔离（事件计时）

工作负载：同一条指令流的两个 kernel，只有 launch 形式不同（folded vs unfolded），
kernel 内 `REPS=8` 次 `B[i] = A[i]`，`max_blocks` 控制每线程的 grid-stride 步数。

| workload | baseline median (ms/launch) | patch median (ms/launch) | speedup baseline/patch | arms | max spread |
|---|---:|---:|---:|---:|---:|
| n_tiles=2048, max_blocks=128 | 0.01471 | 0.01471 | **1.0002x** | 2+2 | 0.44% |
| n_tiles=16384, max_blocks=128 | 0.10026 | 0.10026 | **1.0000x** | 2+2 | 0.31% |
| n_tiles=16384, max_blocks=32 | 0.38109 | 0.38108 | **1.0000x** | 2+2 | 0.09% |
| n_tiles=65536, max_blocks=8 | — | — | **未测出**（spread 37–76%） | 0 | — |

- 有效工作负载：**3**
- **geometric-mean speedup (baseline/patch)：1.0001x**
- 逐负载 speedup：1.0000x .. 1.0002x

`n_tiles=65536, max_blocks=8` 这一档在本机始终无法测量：样本内 spread 37%–76%，
且 baseline 与 patch 两个 build 都出现过极端样本（6.05 ms vs 14.6 ms）。该档按统一规则排除，
不能据此声称收益或回退。

---

## 5. 为什么没有提升：SASS 逐项相同

把两个 build 的同一对 kernel 编成 sm_89 cubin 后反汇编（`cuobjdump -sass` / `-res-usage`）：

| 指标 | baseline fold | patch fold | baseline nofold | patch nofold |
|---|---:|---:|---:|---:|
| `forward_instrs` | 432 | **432** | 528 | **528** |
| `total_instrs` | 1016 | **1016** | 1224 | **1224** |
| REG | 40 | **40** | 40 | **40** |
| STACK / LOCAL | 0 / 0 | **0 / 0** | 0 / 0 | **0 / 0** |
| SHARED | 1072 | **1072** | 1072 | **1072** |
| CONSTANT[0] / [2] | 488 / 8 | **488 / 8** | 488 / 8 | **488 / 8** |

也就是说：`wp::launch_coord(_idx, dim)` 与 `wp::launch_coord_tile(_idx, threadIdx.x, dim)`
被 ptxas 编译成**完全相同的机器码**。原因是 `coord_mult` 在 tiled launch 下恒等于 `blockDim.x`，
编译器已把 `(blockIdx.x*blockDim.x + threadIdx.x) / coord_mult` 折叠成 `blockIdx.x`，
补丁在源码层面做的事编译器本来就做了。

这同时解释了 §3/§4 的量测结果：没有可优化的工作量可消除，所以提升为 0，
而且残余差异（0.01%–0.04%）与噪声同量级。

**更强的对照：两个 build 生成的 CUDA 源码逐字节相同。** 对同一组 tiled launch 形状
（`probe_fold_kernel` 折叠轴 / `probe_nofold_kernel` 未折叠轴）分别导出
`codegen_kernel(..., device="cuda")` 的输出并做 `cmp`：

```
$ cmp /tmp/gen-baseline.json /tmp/gen-patch.json && echo "FILES BYTE-IDENTICAL"
FILES BYTE-IDENTICAL
```

即：补丁在源码层面走的是不同表达式（`wp.tid(_idx, dim)` vs `wp::launch_coord_tile(...)`），
但 Warp 生成的 CUDA、以及 ptxas 生成的 SASS、以及寄存器/shared/stack 用量全部一致。
因此 §6.1 里 tile 套件的 `FAIL` 与补丁无关，这是可以直接判定的，不需要再靠反复重跑来推断。

---

## 6. 判定与建议

按执行计划的阶段判定：

| 项 | 结果 |
|---|---|
| G1 Baseline / diagnosis | 完成：两 build 均已构建，生成源码、PTX、SASS 已固定 |
| G2 Minimal implementation | 完成：`ff0de67`，44 insertions / 4 deletions，仅动确认过的路径 |
| G3 GPU correctness | 完成：新增 72 用例全绿；`test_launch.py` 42/42 OK；`test_array.py` 147/147 OK；`*tile*.py` 基线 1008/1008 OK |
| G4 Performance screening | 完成：E2E 4 档 + micro 3 档有效，均为 1.0000x–1.0004x |
| G5 Clean validation | 完成：两棵独立 worktree 独立构建、独立 venv、独立进程交替测量 |
| **综合判定** | **`INCONCLUSIVE`（无性能收益）** |

**建议：不以性能优化名义提交该补丁。** 执行计划的停止条件里明确写着
"若 PTX/SASS 已等价或性能差异在噪声内，标记 `INCONCLUSIVE`，与维护者确认是否仍值得接受 codegen 简化"。
本轮的 SASS 证据正是"已等价"：`#1361` 想省掉的那段计算，ptxas 早已省掉。

若维护者仍希望推进，建议把 issue 的落点从"性能优化"改为"codegen 可读性/避免依赖编译器把
`coord_mult == blockDim.x` 这一巧合折叠掉"——但这属于**新的范围确认**，本轮不代替维护者决定。

**停止条件已触发项：** 无法在保持 `wp.tid()` 语义的前提下拿出可测量的收益。

---

### 6.1 上游回归套件的对照结果

| 套件 | baseline | patch |
|---|---|---|
| `test_launch.py` | — | **42 passed, 0 failed** |
| `test_array.py` | — | **147 passed, 0 failed** |
| `warp/tests/tile/*`（`test_tile.py` 等） | **1008 passed, 0 failed** | 套件在 1 h 上限内未跑完（见下） |

tile 套件在 patch 树上的一次完整运行被环境截断：主机 load average 一度达到 4800（152 线程），
该次运行 1080 个测试里 `ok=140`、6 个 `FAIL`、1008 个因单套件 3600 s 上限被记为 error；
同一时刻 baseline 树在较空闲的窗口跑出 1008/1008 全绿。

需要说明的是：**这两个 build 的生成机器码逐项相同（§5）**，因此这 6 个 `FAIL` 不可能由补丁引入，
归因是设备争用。但为免把一个未跑完的套件写成通过，此处如实记录 baseline 干净、patch 环境受限。

## 7. 未覆盖 / 已知限制

| 项 | 状态 |
|---|---|
| 4 卡 / 8 卡 | 未做。本机 8 张卡中仅 0/1/6/7 可独占，本轮单卡（GPU 4）足够，改动与卡数无关 |
| `grid_stride=False`（lean）模板 | 未单独计时。补丁不改 lean 模板的 `_idx` 计算，只改 `wp.tid()` 取值 |
| 其他架构（sm_90 / sm_100 / sm_120） | 未测。结论"SASS 相同"只在 sm_89 上验证 |
| `n_tiles=65536, max_blocks=8` | 未测出（spread 37–76%），按统一规则排除 |
| 主机噪声 | 本轮主机 load average 2600–4800（152 线程），wall-clock 计时完全不可用；改事件计时后 CV 降到 0.03–0.4% |
| 上游评论 / PR | 未发送任何评论或 PR；`#1361` 在执行时 open、无 assignee、无关联 PR |

---

## 8. 来源可追溯

| 项 | 路径 / 值 |
|---|---|
| 上游 main SHA（取快照时） | `015e5a17827d4409782e81026d3de6f61b9ae1a7` |
| 本地基线 commit | `3601222efec4b49b20880909d7143c8561a5e7bd` |
| 补丁 commit | `ff0de674a5b144ba949193d2fe2e3ae2b81131aa` |
| 补丁文件 | `a2_1361_tile_coord.diff` |
| 补丁应用脚本（幂等） | `apply_a2_patch.py` |
| 新增测试 | `test_tile_launch_coord.py` → `warp/tests/test_tile_launch_coord.py` |
| E2E harness | `e2e_tiled_pipeline.py` |
| Micro harness | `probe_a2_min.py`, `probe_a2_coord.py` |
| 配对驱动 | `run_ab.sh`（BPPB，每 arm 独立进程） |
| 报告生成 | `report_speedup.py` |
| 原始样本 | `raw/*.json`（64 个 arm），排除项在 `raw-invalid/` |
| 生成源码对照 | `gen_source_compare.py`（两 build 输出 `cmp` 逐字节相同） |
| 构建命令 | `build_lib.py --cuda-path /usr/local/cuda-12.8 --libmathdx-path <ws>/_build/target-deps/libmathdx` |
| 测试命令 | `uv run --extra dev -m warp.tests -s autodetect -p "test_tile_launch_coord.py" --maxjobs 1` |
| 计时口径 | `wp.Event(enable_timing=True)` 包住 `wp.capture_launch` 窗口，同步点在窗口外 |
