# CausalPolicyAllocation

[**在线交互展示**](https://sheatheli.github.io/CausalPolicyAllocation/) · [**工程证据与验证索引**](docs/verification-map.md) · [![Smoke CI](https://github.com/SheatheLi/CausalPolicyAllocation/actions/workflows/smoke.yml/badge.svg)](https://github.com/SheatheLi/CausalPolicyAllocation/actions/workflows/smoke.yml)

> 从历史日志中的因果效果估计出发，先审查新策略是否具备可评价性，再将收益估计纳入具有预算约束的分布鲁棒资源分配。项目保留“拒绝作出不可信决策”及收益代价，而不只展示优化器得分。

**工程状态：** 离线分析与合成模拟原型，不是已上线的经营决策自动化系统，不能在缺少识别条件时宣称真实因果收益。

## 为什么需要它？

过去策略下观察到的收益不代表切换策略后一定能实现的收益。即使估计结果可靠，名义最优解仍可能在分布变化时突破预算。

```text
历史观测数据
     ↓
DML / AIPW / Cross-fitting
     ↓
Overlap / ESS / Holdout 门禁 ──支持不足──> REFUSE
     │ 支持条件允许
     ▼
名义策略与TV鲁棒策略求解
     ↓
收益、最坏预算成本、不可行条件共同报告
```

- **识别层：** 区分效果估计方法与识别前提；避免把相关性分数当因果收益。
- **支持域：** 以Overlap/ESS和独立留出样本检查可评价性。
- **优化层：** 用TV不确定集及LP/MILP评估名义/最坏预算约束。
- **审计层：** 报告收益损失、拒判和分布漂移下的边界。

## 快速运行

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_policy.py tests/test_decision_pipeline.py tests/test_tv_robust_allocation.py
python benchmarks/compare.py
```

其中的RCT与“真值”来自合成模拟器，并非真实用户的反事实记录。对于TV预算对照，也可检查`benchmarks/tv_budget_advantage.py`，该入口不等于实际商业A/B实验。

## 关键源码

| 文件 | 内容 |
|---|---|
| [`src/policy.py`](src/policy.py) | DML/AIPW与策略评价 |
| [`src/heldout_policy_gate.py`](src/heldout_policy_gate.py) | 样本支持、留出验证及拒判 |
| [`src/tv_robust_allocation.py`](src/tv_robust_allocation.py) | TV分布鲁棒预算约束与LP/MILP |
| [`benchmarks/`](benchmarks/) · [`tests/`](tests/) · [`evidence/`](evidence/) | 模拟实验、回归与数据收据 |

## 一组必须完整报告的结果

三群体合成实验（TV半径0.26、预算上限0.65）中，名义方案最坏预算成本**0.81**，鲁棒方案**0.56**；但最坏收益也从**0.664下降至0.22**。这支持指定模拟场景下的预算稳健性，**并不证明真实经营收益提升**或消除隐藏混杂。

## 展示与数据

[交互式决策流程与收益风险展示说明](docs/README.md) · [主要证据图（SVG）](docs/assets/figures/CausalPolicyAllocation.svg)

**授权状态：** 根目录暂无LICENSE；是否开源授权需另行决定。
