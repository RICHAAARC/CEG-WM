# 连续角点八条补充：准备包

当前仅准备，尚未运行内容模型或推送。机制与52条原结果终审见`CONTENT_FINDINGS.md`。

固定原0001/0002 × ±15° × CG/G，全部黑色双线性，共八条。
唯一改动为`2*(128*raw+128)/255-1`去掉原256-grid round；继续同角点顺序、
`homography_observed_to_canonical`、PIL恢复、生产`_score_current_rgb`、原key/public assets。
不改生产contracts，不引入truth选择、刚体拟合、native端点截断、中心修正或搜索。
主tau=1.2657276026437319仅描述，保留全部17W/m；缺raw/非法H保留失败，不回退、不重试。

`subpixel_renderer.ipynb`固定代码`74470afc23b858655d516268760cc430c1905d27`：

- 原R0 `4f0bf156.../r0-f1`四图/result只读复制到Colab临时cache。
- CG raw来自随代码的`cg_geometry_reference.json`（原64测量）；G raw来自Drive旧`content-renderer-v1`的pilot/remaining原记录。
- **不重测几何，也不重跑52条参照**；旧pre/post/oracle只读附在新结果中。
- 优先复用同一Colab的`renderer_assets`/`renderer_session.assets`等匹配生产资产；否则使用原factory，初始化单列计时。没有优化loader。
- 首cell为规定两行mount；Secrets不输出；八条在一个独立手动执行单元运行，不自动launch。
- 新输出`MyDrive/CEG-WM/RotationRenderer-Diagnostic-V1/content-subpixel-v1`；保留plan、initialization、rows.jsonl、paired.csv和summary。

八个既有raw集合已在本地纯离线求得连续H，无模型调用；这只是输入可计算性，不能算内容成功。
两项轻量测试检查亚像素坐标保留、方向及非法raw不调用评分器。Notebook已过nbformat与code-cell compile，
未执行Drive挂载、安装、factory、八条评分。即使八条有改善，也不能直接形成独立验证或正式协议结论。
发布和真实执行须控制审阅具体提交，V2全域搜索继续暂停。
