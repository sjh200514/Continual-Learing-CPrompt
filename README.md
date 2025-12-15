# 在LibContinual框架下复现持续学习领域的论文：Consistent Prompting for Rehearsal-Free Continual Learning
我复现了这篇论文中的CPrompt算法，旨在解决持续学习方法中**分类器不⼀致和提⽰不⼀致这两种问题**。

我在已有的LibContinual框架下添加了两个代码文件，分别是 `CPrompt.py`和`CPrompt.yaml`。`CPrompt.py`是核心算法文件，而`CPrompt.yaml`则是CPrompt算法的配置文件。

最后在CIFAR-100数据集上并没有达到论文中的精度，比论文中的精度低2-3个百分点。
