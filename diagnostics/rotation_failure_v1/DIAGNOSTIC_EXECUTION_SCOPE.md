# 固定 100 对诊断执行准备

当前仅准备入口，尚未授权/启动完整诊断，也未授权 BlindDetection-V2。

- 基线沿用 main e12c7ea；只运行当前诊断分支的冻结实现，不改方法或阈值。
- 输入按 input_reference.json 中固定 Drive 文件 ID 读取；本地缓存 `/content/rotation-diagnostic-input`。
- runtime 资产 `/content/rotation-diagnostic-runtime`。
- 输出固定 `/content/drive/MyDrive/CEG-WM/RotationFailure-Diagnostic-V1/diagnostic-v1`；与 preflight-v1 分离。
- 不限制 GPU 型号/显存，不据此阻断。实际执行错误保留；不自动改方法、降阈值或更换样本。
- 100 对、未攻击/旋转两条件、正负两侧，共 400 条图像条件记录、200 条配对记录。
- science_denominator=0；不得替代旧 N=1000 或用作修改后独立正式测试。
- 不生成/嵌入图片，不调用任何 baseline，不运行重建/Finalizer，不进入方法修改。
- 不自动重试已终结样本。中断后同输出目录保留并跳过已有行；完成后拒绝重复运行。
- 完整运行耗时尚未测得，不把一次合成预检耗时当作整套预算保证。

执行前由用户另行授权冻结提交、代码发布/交付和一次完整诊断。用户可在 Colab 手动执行入口。
Google Drive 授权只用于读取原始文件；程序对原目录没有写接口。诊断输出通过挂载 Drive 写到指定新目录。

准备流程：Drive 挂载 → API 读取固定原图/来源行 → 身份与解码审计 → Secrets/运行依赖 → 一次诊断 → read-only 覆盖审计。
check_results.py 按层、条件、正负侧报告计划/已写/缺失/失败/无效及 replay 不一致数；不按性能决定结果包是否有效。

执行后先独立审阅 400 条原始记录，区分运行完整、几何误差、oracle 证据变化和匹配负样本变化。
原因仍无法确认则停留在诊断，不启动 BlindDetection-V2。
