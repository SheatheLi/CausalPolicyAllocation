# 验证路线｜策略可评价性与鲁棒预算的收益代价

此项目是**离线研究与合成模拟**。不能因为某种策略在合成环境中胜出，就宣称真实用户的反事实收益可识别。

## 三层分开的技术合同

1. **识别：** DML/DR、AIPW和Cross-fitting是估计手段；其结果仍依赖相应识别假设。
2. **支持域：** 候选策略缺少历史动作覆盖、Overlap或ESS不够时，应明确REFUSE，而不是向不受支持人群外推。
3. **预算稳健性：** TV不确定集与LP/MILP求解必须同时报告最坏预算成本、约束和损失的收益。

| 验证问题 | 源码与实验 |
|---|---|
| 离线策略值与估计器行为 | `src/policy.py`、`tests/test_policy.py` |
| 支持不足时的拒判 | `src/heldout_policy_gate.py`、`tests/test_heldout_policy_gate.py` |
| TV半径与预算约束 | `src/tv_robust_allocation.py`、`tests/test_tv_robust_allocation.py` |
| 名义与鲁棒方案的代价 | `benchmarks/tv_budget_advantage.py`、`evidence/targeted_tv_budget_20261014.json` |

在指定三群体合成实验中，名义策略最坏成本0.81，鲁棒策略0.56；同时最坏收益从0.664降为0.22。**成本更稳健不代表收益更好**。该结果不能为真实经营决策自动背书。

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_policy.py tests/test_decision_pipeline.py tests/test_tv_robust_allocation.py tests/test_heldout_policy_gate.py
```
