# M3.1 逐题结果

未运行正式 holdout；以下为 dev 失败诊断，不是最终 B2.1 结果。

诊断候选：`r3_07_title_section_quote`

- W/T/L: `27/14/7`
- Subset hit deltas: `{"exact_term_definition": 3, "method_section_location": -1, "experiment_number_table": 3, "cross_paper_or_section": -1}`
- Failed checks: `losses_at_most_3, p95_ratio_at_most_1_35`

| Case | Category | B1 rank | Diagnostic rank | Result |
| --- | --- | ---: | ---: | --- |
| term-01-transformer | exact_term_definition | - | 5 | win |
| term-02-multi-head | exact_term_definition | 7 | 6 | win |
| term-03-scaled-dot-product | exact_term_definition | 6 | 7 | loss |
| term-04-positional-encoding | exact_term_definition | 8 | 2 | win |
| term-05-lstm-dependency | exact_term_definition | 1 | 1 | tie |
| term-06-cell-state | exact_term_definition | 1 | 1 | tie |
| term-07-pointer-network | exact_term_definition | - | 5 | win |
| term-08-residual-learning | exact_term_definition | - | - | tie |
| term-09-dilated-convolution | exact_term_definition | 3 | 1 | win |
| term-10-mpnn | exact_term_definition | - | 1 | win |
| term-11-ntm-memory | exact_term_definition | 3 | 2 | win |
| term-12-rnn-dropout | exact_term_definition | 1 | 5 | loss |
| method-01-transformer-encoder | method_section_location | 3 | 1 | win |
| method-02-gpipe-microbatch | method_section_location | 4 | 2 | win |
| method-03-resnet-bottleneck | method_section_location | 10 | 4 | win |
| method-04-preactivation | method_section_location | 2 | 5 | loss |
| method-05-relation-network | method_section_location | 2 | 1 | win |
| method-06-rmc | method_section_location | - | - | tie |
| method-07-nmt-alignment | method_section_location | - | - | tie |
| method-08-deepspeech-bn | method_section_location | 1 | - | loss |
| method-09-mdl-selection | method_section_location | - | 6 | win |
| method-10-alexnet-relu | method_section_location | 2 | 1 | win |
| method-11-read-process-write | method_section_location | 1 | 1 | tie |
| method-12-vlae-information | method_section_location | 4 | - | loss |
| number-01-transformer-bleu | experiment_number_table | 4 | 3 | win |
| number-02-alexnet-error | experiment_number_table | - | - | tie |
| number-03-resnet-error | experiment_number_table | - | 5 | win |
| number-04-gpipe-amoeba | experiment_number_table | 3 | 2 | win |
| number-05-rnn-penn | experiment_number_table | 4 | 3 | win |
| number-06-nmt-bleu | experiment_number_table | 1 | 1 | tie |
| number-07-relation-clevr | experiment_number_table | - | - | tie |
| number-08-deepspeech-wer | experiment_number_table | 1 | 1 | tie |
| number-09-dilated-iou | experiment_number_table | 1 | 1 | tie |
| number-10-vlae-nll | experiment_number_table | - | 3 | win |
| number-11-rmc-accuracy | experiment_number_table | - | 9 | win |
| number-12-pointer-accuracy | experiment_number_table | - | - | tie |
| cross-01-transformer-nmt | cross_paper_or_section | - | 3 | win |
| cross-02-resnet-identity | cross_paper_or_section | 1 | 3 | loss |
| cross-03-rnn-lstm | cross_paper_or_section | 4 | 1 | win |
| cross-04-pointer-ntm | cross_paper_or_section | 2 | 1 | win |
| cross-05-rn-rmc | cross_paper_or_section | 4 | 8 | loss |
| cross-06-alexnet-resnet | cross_paper_or_section | - | 6 | win |
| cross-07-gpipe-method-result | cross_paper_or_section | 2 | 2 | tie |
| cross-08-scaling-allocation | cross_paper_or_section | 2 | 1 | win |
| cross-09-dilated-method-result | cross_paper_or_section | 1 | 1 | tie |
| cross-10-mdl-intro-selection | cross_paper_or_section | - | 7 | win |
| cross-11-order-method-result | cross_paper_or_section | 2 | 1 | win |
| cross-12-coffee-complexity | cross_paper_or_section | 4 | 2 | win |

## 坏例

| Case | Category | Tags | Gold rank | Question |
| --- | --- | --- | ---: | --- |
| term-08-residual-learning | exact_term_definition | 中英混合 | - | Deep Residual Learning 如何定义 residual mapping？ |
| method-06-rmc | method_section_location | 缩写, 章节定位 | - | Relational Recurrent Neural Networks 的 RMC 如何让 memory slots 交互？ |
| method-07-nmt-alignment | method_section_location | 章节定位, 中英混合 | - | Neural Machine Translation 的 proposed model 如何 jointly learn to align and translate？ |
| method-08-deepspeech-bn | method_section_location | 缩写, 章节定位 | - | Deep Speech 2 在 3.2 节怎样对 recurrent network 使用 Batch Normalization？ |
| method-12-vlae-information | method_section_location | 缩写, 章节定位 | - | Variational Lossy Autoencoder 的 3.1 节如何显式安排 lossy code 的信息？ |
| number-02-alexnet-error | experiment_number_table | 表格, 缩写, 数字 | - | AlexNet 的 ILSVRC-2012 top-5 test error 是多少？ |
| number-07-relation-clevr | experiment_number_table | 表格, 数字, 缩写 | - | Relation Network 的 CLEVR accuracy 表中 RN 达到多少，是否超过 95.5%？ |
| number-12-pointer-accuracy | experiment_number_table | 表格, 数字 | - | Pointer Network 的 convex hull 实验表中 99.9% accuracy 对应哪个序列长度？ |
| cross-01-transformer-nmt | cross_paper_or_section | 跨论文, 缩写 | 3 | 对比 Transformer 的 multi-head attention 与 RNNsearch 的对齐机制：两者分别如何汇聚输入信息？ |
| cross-02-resnet-identity | cross_paper_or_section | 跨论文, 中文术语 | 3 | Deep Residual Learning 的 residual mapping 与 Identity Mappings 的 pre-activation unit 如何衔接？ |
| cross-03-rnn-lstm | cross_paper_or_section | 跨论文, 缩写 | 1 | 普通 RNN 的 hidden state 更新与 LSTM 处理 long-term dependency 的设计有何不同？ |
| cross-05-rn-rmc | cross_paper_or_section | 跨论文, 缩写 | 8 | Relation Network 的 g_theta 成对关系计算与 RMC 的 memory interaction 各自如何做 relational reasoning？ |
| cross-06-alexnet-resnet | cross_paper_or_section | 跨论文, 表格, 数字 | 6 | AlexNet 与 ResNet 的 ImageNet 表格分别报告了 15.3% 和 3.57% 哪种 error？ |
| cross-07-gpipe-method-result | cross_paper_or_section | 跨章节, 表格, 数字 | 2 | GPipe 的 micro-batch algorithm 与 AmoebaNet 84.4% ImageNet 结果分别位于哪些章节，如何关联？ |
| cross-09-dilated-method-result | cross_paper_or_section | 跨章节, 表格, 数字 | 1 | Dilated convolution 的 receptive-field 方法如何对应 Pascal VOC mean IoU 69.8/71.3 的实验结果？ |
| cross-10-mdl-intro-selection | cross_paper_or_section | 跨章节, 缩写 | 7 | MDL 教程从 Minimum Description Length 概念到 refined model selection 的原则是什么？ |
| cross-11-order-method-result | cross_paper_or_section | 跨章节, 表格 | 1 | Order Matters 的 Read-Process-and-Write 方法在 sorting experiment 中为什么 processing step 大于零后更好？ |

## 人工核查记录

- 已检查全部 48 个 B1/诊断候选 rank 变化，包含 win 与 loss；诊断候选因 loss 超过 3 条而失败。
- 已按表格、缩写、跨章节、中文术语标签各检查至少 3 题；中文术语仅 1 题为 miss，其余检查包含退化、持平和改善案例。
- 已抽查 5 个 blended rerank trace；final rank 同时使用 fusion rank 与 rerank rank，未发现硬编码保留前 N。
- 26 个 dev pipeline 的 answer preview metadata prefix leak 合计为 0。
- 已检查 old dev/new holdout 最相似问题对；最高文本相似对分别询问 ILSVRC-2012 单值与 ILSVRC-2010 两个值，不是简单同义改写。
- Active index pointer 在实验前后均为空，未发生修改。
