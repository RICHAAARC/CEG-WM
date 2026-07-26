# Joint Decision Design

## Decision Authority

内容检测器拥有阳性判定权。几何链只能决定是否有足够依据执行坐标恢复。

## Frozen Flow

设冻结内容检测器为 `D`，校准阈值为 `tau`，近阈值负区间下界为 `tau_rescue`：

1. 计算原图分数 `s_raw = D(image, key)`。
2. 若 `s_raw >= tau`，由内容证据判阳性。
3. 若 `s_raw < tau_rescue`，判为内容负样本，不启动几何恢复。
4. 若 `tau_rescue <= s_raw < tau`，检查几何估计。
5. 几何失败或不可靠时，保留原内容负判定。
6. 几何可靠时回正图像并计算 `s_rectified = D(rectified_image, key)`。
7. 仅当 `s_rectified >= tau` 时，由回正后的内容证据判阳性。

当前 `D` 冻结为 CEG-WM HF direct score 形成的 content detector。未来若经设计验证采用 LF/HF 组合 content detector，必须通过权威设计变更同时替换原图和回正图的 `D`，不得只替换救援路径。

## Invariants

- 原图与回正图的 detector identity 相同。
- 原图与回正图的 key semantics 相同。
- 原图与回正图的 threshold identity 相同。
- 几何可靠性不进入阳性分数。
- 几何链不直接返回阳性。
- 非近阈值样本不执行恢复。
- 不存在仅服务回正图的宽松分类器、权重或阈值。
- 恢复失败必须显式记录，不能静默退化为成功。

## Calibration Boundary

`tau` 和 `tau_rescue` 只能在 calibration split 上确定。evaluation split 不得重新拟合阈值、救援区间、几何可靠性或 LF/HF 组合。

`tau_rescue` 控制计算与恢复资格，不改变固定 FPR 主阈值 `tau`。

完整联合检测器的 FPR 必须把 raw 直接阳性与 rescue 后阳性共同计入。只证明 raw detector 达到目标 FPR 不足以证明联合方法达到目标；`tau_rescue` 和几何可靠性必须在独立 calibration 职责数据上拟合，并由完整联合路径进行 calibration check。

## Required Ablations

- raw content only；
- geometry always attempted；
- conditional geometry recovery；
- oracle transform，仅作为诊断上界；
- wrong-key geometry；
- rectification with same detector；
- rectification with detector disabled control。

oracle transform 不能进入正式方法结果。

## Current Status

判定语义已冻结；检测器、阈值和几何组件尚未在本项目实现或校准。
