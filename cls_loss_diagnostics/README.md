# E2E 分类损失不下降排查记录

## 目的

这些控制变量实验用于排查：为什么 baseline E2E ResNet/ABMIL 的 `train_loss_cls` 长期停在约 0.68–0.69，不能稳定下降。

下表只记录截图中明确可见的 fold 0 数值，不代表完整三折结果，也不用于宣称性能提升。

## 实验对比

| 实验 | 控制变量 | 要验证的问题 | 截图中的结果 | 当前解释 |
|---|---|---|---|---|
| Attention，8-RHAG，`lambda_sr=0`，overfit16 | 只用 16 个样本并关闭 SR loss | 分类链路能否拟合极小数据集？ | 20 个 epoch 中 cls loss：0.7242 → 0.4995 | 分类链路并非完全断开，loss 可以明显下降；但尚未达到接近 0 的完整过拟合，也不能说明泛化能力。 |
| Attention，8-RHAG，`lambda_sr=0`，no weight decay | 同时去掉 SR 竞争和 L2 正则 | cls loss 不降是否主要由 SR 或 weight decay 引起？ | 可见 2 个 epoch：0.6873 → 0.6889 | 没有立即恢复下降；但 2 个 epoch 太短，不能下最终结论。 |
| Attention，8-RHAG，`lambda_sr=0.1`，no weight decay | 恢复较小的 SR 辅助损失 | SR 监督能否让共享编码器学到更有用的特征？ | cls：0.6892 → 0.6854；SR：0.3221 → 0.1130（5 个 epoch） | SR 分支明显在学习，但分类仍接近二分类随机交叉熵基线；“优化器没有工作”不是充分解释。 |
| Attention，HAT-S6，`lambda_sr=0`，freeze HAT，`lr=3e-4`，no weight decay | 冻结 HAT | 是否因为 HAT 更新不稳定导致分类停滞？ | 4 个 epoch：0.7866 → 0.7332 | 有下降但波动较大、训练太短，不能证明冻结 HAT 能解决问题。 |
| Attention，HAT-S6，`lambda_sr=0`，pretrain，`lr=3e-4`，no weight decay | 使用预训练 HAT 初始化 | 更好的初始化能否解决分类停滞？ | 17 个 epoch：0.7866 → 0.6862，中间明显波动 | 预训练把 loss 拉回约 0.69，但没有形成稳定下降趋势。 |
| Frequency Router，2-RHAG | 更换分类表征路径 | 停滞是否只属于 baseline ResNet 表征？ | cls：0.6850 → 0.6831；SR：0.2901 → 0.0986（12 个 epoch） | SR 明显下降而分类几乎不变；仅替换下游表征并未消除现象。 |

## 目前能确定什么

1. 整个 E2E 优化过程不是全局停滞：两个独立实验中的 SR loss 都明显下降。
2. 分类反向链路不是完全断开：16 样本实验中的 cls loss 从 0.7242 降至 0.4995。
3. SR loss 和 weight decay 不太可能是唯一原因：两者都关闭后，分类 loss 仍停在约 0.69。
4. 排查重点应回到分类信号：标签与数据对齐、类别/预测塌缩、特征可分性，以及 HAT、ResNet、ABMIL 各段的有效梯度。

## 目前不能证明什么

- 截图只是部分 fold 0 记录，不是原始 `history.csv`。
- 各实验的 seed、样本集合和训练长度没有全部严格对齐，不能直接比较最终性能。
- cls loss 接近 0.69 时，还必须同时检查 logits/概率分布、真实/预测类别计数、AUC 和分模块梯度。
- overfit16 尚未达到 loss 接近 0、训练准确率 100%，因此只能证明“能学”，不能证明链路完全正确。

## 下一步最有判别力的检查

1. 保存每组实验的原始 `history.csv`、运行参数和梯度日志。
2. 固定同一批 16 张 WSI，依次测试：只训分类头、ResNet+分类头、完整 HAT+ResNet+ABMIL。
3. 每个 epoch 记录真实类别数、预测类别数和肿瘤概率分布，检查是否长期输出单一类别。
4. 在 `optimizer.step` 前检查 HAT、ResNet、ABMIL 的 finite/nonzero 梯度范数。
5. 用相同 split、seed、初始化、学习率和 epoch 数严格比较 `lambda_sr=0` 与 `lambda_sr=0.1`。

原始手机照片没有上传，因为仓库是公开的，照片会暴露服务器和会话信息。结构化抄录见 [experiment_matrix.csv](experiment_matrix.csv)。
