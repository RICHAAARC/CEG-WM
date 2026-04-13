# 3. Method

## 3.1 问题定义

本文研究文本到图像扩散模型中的可验证水印问题。设文本提示为 $p$，扩散生成模型为 $G$，随机种子与采样噪声统称为 $\xi$，则生成图像记为
$$
x = G(p;\xi).
$$
我们的目标不是仅检测图像中是否存在某种静态扰动，而是在尽量保持生成质量与语义一致性的前提下，构建一套能够同时支持**水印存在性判断**与**生成事件归因**的多证据检测框架。

基于当前系统设计，本文将方法划分为两级目标。第一级是**主干判定目标**，即通过内容证据链与事件级 attestation 判断图像是否属于一次受控生成事件。第二级是**辅助增强目标**，即通过几何鲁棒证据与低容量 payload probe 提供补充一致性信息，用于边界样本分析、攻击后辅助解释与系统消融，而不作为主判定链的唯一依据。

---

## 3.2 方法总览

本文提出一种以**内容主证据链**和**事件级 attestation** 为核心、以**几何补充链**和**轻量 payload probe** 为辅助的扩散水印框架。整体由五个模块组成：

1. 语义掩码与区域路由模块；
2. 轨迹特征与 `Jacobian-Vector Product`（`JVP`）联合驱动的子空间规划模块；
3. 双通道潜空间调制模块；
4. 事件级 attestation 与主判定融合模块；
5. 几何补充链与辅助 payload 一致性模块。

设第 $t$ 步潜变量为 $z_t$，总采样步数为 $T$。系统的主流程可以记为
$$
\mathcal{M}=\bigl(\mathcal{S},\mathcal{P},\mathcal{E}*{\mathrm{LF}},\mathcal{E}*{\mathrm{HF}},\mathcal{A},\mathcal{G},\mathcal{F}\bigr),
$$
其中，$\mathcal{S}$ 表示语义路由，$\mathcal{P}$ 表示子空间规划，$\mathcal{E}*{\mathrm{LF}}$ 与 $\mathcal{E}*{\mathrm{HF}}$ 分别表示低频与高频调制分支，$\mathcal{A}$ 表示事件级 attestation，$\mathcal{G}$ 表示几何辅助证据，$\mathcal{F}$ 表示最终融合规则。

需要强调的是，本文的**主输出**不是消息恢复，而是：

* 内容链分数；
* 事件级 attestation 分数；
* system-level 最终判定。

辅助模块仅在主干判定之外提供补充信息。

---

## 3.3 语义内容自适应区域划分

为避免统一扰动破坏图像语义结构，本文首先基于语义显著性先验构造掩码
$$
M \in [0,1]^{H\times W},
$$
并将其下采样到潜空间分辨率，得到
$$
\widetilde{M}\in [0,1]^{H_z\times W_z}.
$$
根据 $\widetilde{M}$ 的统计结果，将潜空间划分为高频优先区域 $\Omega_{\mathrm{HF}}$ 与低频优先区域 $\Omega_{\mathrm{LF}}$，满足
$$
\Omega_{\mathrm{HF}}\cup \Omega_{\mathrm{LF}}=\Omega,\qquad
\Omega_{\mathrm{HF}}\cap \Omega_{\mathrm{LF}}=\varnothing.
$$

其中，$\Omega_{\mathrm{HF}}$ 对应纹理复杂、局部结构变化更显著的区域，$\Omega_{\mathrm{LF}}$ 对应相对平滑、低频主导的区域。
本文在这两类区域上采用不同的调制与证据提取策略，使扰动强度与局部语义复杂度相匹配，从而提高证据形成效率并降低可感知失真。

---

## 3.4 轨迹特征与 `JVP` 联合驱动的子空间规划

设扩散采样过程中可稳定读取的轨迹特征为 $\phi_t$，沿采样轨迹选取 $K$ 个时刻，可得到
$$
\Phi=[\phi_{t_1},\phi_{t_2},\ldots,\phi_{t_K}]^\top \in \mathbb{R}^{K\times d}.
$$

为了描述局部扰动在扩散轨迹中的传播敏感性，本文不显式构造完整 Jacobian，而通过 `JVP` 估计局部线性响应。设局部映射为 $f(z)$，对方向向量 $v$ 有
$$
J(z)v=\frac{\partial f(z)}{\partial z}v.
$$
对若干方向 ${v_i}_{i=1}^{m}$ 计算 `JVP` 样本矩阵
$$
\Psi=[J(z)v_1,\ldots,J(z)v_m]^\top \in \mathbb{R}^{m\times d}.
$$

随后，对 LF 与 HF 两个区域分别构造联合矩阵
$$
D_{\mathrm{LF}}=
\begin{bmatrix}
\Phi_{\mathrm{LF}}\
\Psi_{\mathrm{LF}}
\end{bmatrix},
\qquad
D_{\mathrm{HF}}=
\begin{bmatrix}
\Phi_{\mathrm{HF}}\
\Psi_{\mathrm{HF}}
\end{bmatrix},
$$
并执行奇异值分解，得到各自的低维基底
$$
B_{\mathrm{LF}},\quad B_{\mathrm{HF}}.
$$

这些基底并非静态固定，而是由当前样本的语义区域、扩散轨迹与局部敏感性联合决定，因此具有明显的内容自适应性。

---

## 3.5 双通道潜空间调制：主内容证据与辅助 payload probe

### 3.5.1 低频分支：一致性调制与轻量 payload probe

低频分支的主要作用是形成稳定的内容证据，而不是承担强鲁棒消息恢复。
具体地，本文在 (B_{\mathrm{LF}}) 张成的低维子空间内施加受控调制：
$$
\Delta z_t^{\mathrm{LF}}=\alpha_t^{\mathrm{LF}},B_{\mathrm{LF}}u_{\mathrm{LF}},
$$
其中，$u_{\mathrm{LF}}$ 为低频调制向量，$\alpha_t^{\mathrm{LF}}$ 表示步依赖强度调度。

在工程实现中，低频分支还可以附带一个**低容量 payload probe**。设原始辅助比特串为 $m$，经 `LDPC` 编码得到码字 $c$，再与幅度模板组合形成轻量 probe：
$$
w_{\mathrm{aux}}=c\odot |\epsilon|,\qquad \epsilon\sim \mathcal{N}(0,I).
$$
但这里的 payload probe 并不作为系统主判定链的核心输入，其输出仅用于形成：

* bit-level agreement；
* codeword consistency；
* auxiliary consistency statistics。

换言之，payload 在本文中是**附加一致性观测**，而不是主判定所依赖的核心证据。

### 3.5.2 高频分支：纹理区鲁棒调制

高频区域更适合承载对局部纹理变化更稳健的结构化扰动。设高频投影系数为 $u_{\mathrm{HF}}$，本文仅保留其稳定尾部结构并施加有界调制：
$$
\widetilde{u}*{\mathrm{HF}}=\mathcal{T}*{\rho}(u_{\mathrm{HF}}),
\qquad
\Delta z_t^{\mathrm{HF}}=\alpha_t^{\mathrm{HF}},B_{\mathrm{HF}}\widetilde{u}_{\mathrm{HF}}.
$$

LF 与 HF 两个分支的总扰动为
$$
\Delta z_t=\mathbf{P}*{\Omega*{\mathrm{LF}}}\bigl(\Delta z_t^{\mathrm{LF}}\bigr)+
\mathbf{P}*{\Omega*{\mathrm{HF}}}\bigl(\Delta z_t^{\mathrm{HF}}\bigr),
$$
并更新潜变量：
$$
z_t'=z_t+\Delta z_t.
$$

该设计使系统能够同时利用低频稳定性与高频纹理鲁棒性形成内容证据，但主判定仍由后续内容链与 attestation 共同完成。

---

## 3.6 事件级 attestation 与主判定链

仅凭扰动存在并不足以证明图像来自一次真实受控生成。
为此，本文在生成端引入事件级声明、摘要与签名绑定，记事件声明为 $e$，其签名结果为 $\sigma(e)$。检测端不仅判断图像中是否存在与本方法一致的内容证据，还验证这些证据是否与某次已声明事件一致。

设内容主证据分数为 $s_c$，事件级 attestation 分数为 $s_a$。本文将二者视为系统主干判定输入，通过主融合函数得到
$$
s_{\mathrm{sys}}=\mathcal{F}*{\mathrm{main}}(s_c,s_a).
$$
最终主判定记为
$$
\hat{y}*{\mathrm{main}}=
\begin{cases}
1, & s_{\mathrm{sys}}\ge \tau_{\mathrm{sys}},\
0, & \text{otherwise},
\end{cases}
$$
其中 $\tau_{\mathrm{sys}}$ 为系统主阈值。

在本文的工程实现与论文口径中，真正承担主要方法贡献的是：

* 内容证据形成能力；
* 事件级 attestation 的可验证性；
* 二者构成的 system-level 决策。

---

## 3.7 几何补充链：单向辅助救回

当图像经历显著几何攻击时，内容链可能在边界区间内失配。为此，本文引入几何补充链，但其定位是**单向辅助救回器**，而非对等主干。

设几何链分数为 $s_g$，内容主链的边界区间为 $\mathcal{B}*{\mathrm{rescue}}$，几何阈值为 $\tau_g$。最终判定规则为
$$
\hat{y}=
\begin{cases}
1, & s*{\mathrm{sys}}\ge \tau_{\mathrm{sys}},\
1, & s_{\mathrm{sys}}<\tau_{\mathrm{sys}},\ s_{\mathrm{sys}}\in \mathcal{B}_{\mathrm{rescue}},\ s_g\ge \tau_g,\
0, & \text{otherwise}.
\end{cases}
$$

该规则意味着：

1. 当主链已判为正时，不再依赖几何链；
2. 几何链只允许执行 $False\rightarrow True$ 的单向救回；
3. 几何链无权反向否决已成立的主判定。

因此，几何模块在本文中是**辅助鲁棒证据**，而不是与内容链、attestation 并列的核心主判定链。

---

## 3.8 模块角色总结

结合当前工程实现与论文主张，本文系统的模块角色应明确区分为：

### 核心主干

1. 语义内容自适应证据链；
2. 事件级 attestation；
3. system-level 主判定。

### 辅助模块

1. 几何单向补充救回链；
2. 低容量 payload probe。

这种角色划分带来的好处是：

* 论文主结论与当前工程主链一致；
* 辅助模块即使效果尚不充分，也不会破坏主方法的完整性；
* 后续可在不改变主框架的前提下，独立增强 payload 或 geometry。
