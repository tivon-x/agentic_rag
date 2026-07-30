# M3.2 逐题结果

## holdout

| Case | Category | B1 rank | S1 rank | Result |
| --- | --- | ---: | ---: | --- |
| holdout-term-01-label-smoothing | exact_term_definition | 1 | 1 | tie |
| holdout-term-02-gpipe-bubble | exact_term_definition | 1 | 1 | tie |
| holdout-term-03-overlapping-pooling | exact_term_definition | 1 | 1 | tie |
| holdout-term-04-mpnn-readout | exact_term_definition | 1 | 1 | tie |
| holdout-term-05-sortagrad | exact_term_definition | 2 | 1 | win |
| holdout-term-06-ntm-addressing | exact_term_definition | 2 | 1 | win |
| holdout-term-07-bits-back | exact_term_definition | 3 | 1 | win |
| holdout-term-08-light-cone-complexity | exact_term_definition | 4 | 3 | win |
| holdout-term-09-identity-shortcut | exact_term_definition | 1 | - | loss |
| holdout-term-10-rmc-memory-slots | exact_term_definition | 4 | 1 | win |
| holdout-term-11-row-convolution | exact_term_definition | 4 | 2 | win |
| holdout-term-12-autoregressive-flow-prior | exact_term_definition | 3 | 3 | tie |
| holdout-method-01-transformer-warmup | method_section_location | 6 | 1 | win |
| holdout-method-02-gpipe-interface | method_section_location | 1 | 1 | tie |
| holdout-method-03-resnet-shortcut-options | method_section_location | 7 | 2 | win |
| holdout-method-04-mpnn-set2set | method_section_location | 1 | 1 | tie |
| holdout-method-05-ntm-location-shift | method_section_location | 1 | 1 | tie |
| holdout-method-06-pointer-training | method_section_location | 1 | 1 | tie |
| holdout-method-07-alexnet-weight-decay | method_section_location | 1 | 1 | tie |
| holdout-method-08-sortagrad-order | method_section_location | 1 | 1 | tie |
| holdout-method-09-row-convolution-placement | method_section_location | 1 | 1 | tie |
| holdout-method-10-vlae-flow-implementation | method_section_location | 1 | 1 | tie |
| holdout-method-11-scaling-training | method_section_location | 1 | 1 | tie |
| holdout-method-12-transformer-decoding | method_section_location | 3 | 1 | win |
| holdout-number-01-transformer-beam | experiment_number_table | 1 | 2 | loss |
| holdout-number-02-alexnet-2010-errors | experiment_number_table | - | 10 | win |
| holdout-number-03-resnet-1202 | experiment_number_table | 6 | 3 | win |
| holdout-number-04-nmt-rnnencdec | experiment_number_table | 1 | 1 | tie |
| holdout-number-05-deepspeech-depth | experiment_number_table | 1 | 2 | loss |
| holdout-number-06-deepspeech-oov | experiment_number_table | 1 | 2 | loss |
| holdout-number-07-mpnn-input-information | experiment_number_table | 3 | 1 | win |
| holdout-number-08-gpipe-speedup | experiment_number_table | 1 | 1 | tie |
| holdout-number-09-vlae-flow-steps | experiment_number_table | 1 | 1 | tie |
| holdout-number-10-scaling-batch | experiment_number_table | 1 | - | loss |
| holdout-number-11-label-smoothing-value | experiment_number_table | 1 | 1 | tie |
| holdout-number-12-pointer-training-budget | experiment_number_table | 1 | 1 | tie |
| holdout-cross-01-transformer-regularization | cross_paper_or_section | 7 | 2 | win |
| holdout-cross-02-gpipe-interface-bubble | cross_paper_or_section | 1 | 1 | tie |
| holdout-cross-03-resnet-shortcut-propagation | cross_paper_or_section | - | - | tie |
| holdout-cross-04-mpnn-readout-ablation | cross_paper_or_section | 4 | 1 | win |
| holdout-cross-05-sortagrad-depth-result | cross_paper_or_section | 2 | - | loss |
| holdout-cross-06-ntm-addressing-modes | cross_paper_or_section | 1 | 1 | tie |
| holdout-cross-07-vlae-coding-prior | cross_paper_or_section | 5 | 1 | win |
| holdout-cross-08-alexnet-pooling-optimization | cross_paper_or_section | 2 | 1 | win |
| holdout-cross-09-transformer-nmt-decoding | cross_paper_or_section | - | 2 | win |
| holdout-cross-10-scaling-schedule | cross_paper_or_section | 1 | 5 | loss |
| holdout-cross-11-pointer-training-validity | cross_paper_or_section | 1 | 1 | tie |
| holdout-cross-12-deepspeech-streaming | cross_paper_or_section | 2 | 1 | win |

## old_dev

| Case | Category | B1 rank | S1 rank | Result |
| --- | --- | ---: | ---: | --- |
| term-01-transformer | exact_term_definition | - | 5 | win |
| term-02-multi-head | exact_term_definition | 7 | 6 | win |
| term-03-scaled-dot-product | exact_term_definition | 6 | 7 | loss |
| term-04-positional-encoding | exact_term_definition | 8 | 2 | win |
| term-05-lstm-dependency | exact_term_definition | 1 | 1 | tie |
| term-06-cell-state | exact_term_definition | 1 | 1 | tie |
| term-07-pointer-network | exact_term_definition | - | 6 | win |
| term-08-residual-learning | exact_term_definition | - | - | tie |
| term-09-dilated-convolution | exact_term_definition | 3 | 1 | win |
| term-10-mpnn | exact_term_definition | - | 1 | win |
| term-11-ntm-memory | exact_term_definition | 3 | 2 | win |
| term-12-rnn-dropout | exact_term_definition | 1 | 8 | loss |
| method-01-transformer-encoder | method_section_location | 3 | 2 | win |
| method-02-gpipe-microbatch | method_section_location | 4 | 3 | win |
| method-03-resnet-bottleneck | method_section_location | 10 | 4 | win |
| method-04-preactivation | method_section_location | 2 | 6 | loss |
| method-05-relation-network | method_section_location | 2 | 1 | win |
| method-06-rmc | method_section_location | - | - | tie |
| method-07-nmt-alignment | method_section_location | - | - | tie |
| method-08-deepspeech-bn | method_section_location | 1 | - | loss |
| method-09-mdl-selection | method_section_location | - | 9 | win |
| method-10-alexnet-relu | method_section_location | 2 | 1 | win |
| method-11-read-process-write | method_section_location | 1 | 1 | tie |
| method-12-vlae-information | method_section_location | 4 | - | loss |
| number-01-transformer-bleu | experiment_number_table | 4 | 4 | tie |
| number-02-alexnet-error | experiment_number_table | - | - | tie |
| number-03-resnet-error | experiment_number_table | - | 4 | win |
| number-04-gpipe-amoeba | experiment_number_table | 3 | 2 | win |
| number-05-rnn-penn | experiment_number_table | 4 | 3 | win |
| number-06-nmt-bleu | experiment_number_table | 1 | 1 | tie |
| number-07-relation-clevr | experiment_number_table | - | - | tie |
| number-08-deepspeech-wer | experiment_number_table | 1 | 1 | tie |
| number-09-dilated-iou | experiment_number_table | 1 | 3 | loss |
| number-10-vlae-nll | experiment_number_table | - | 1 | win |
| number-11-rmc-accuracy | experiment_number_table | - | 8 | win |
| number-12-pointer-accuracy | experiment_number_table | - | - | tie |
| cross-01-transformer-nmt | cross_paper_or_section | - | 4 | win |
| cross-02-resnet-identity | cross_paper_or_section | 1 | 6 | loss |
| cross-03-rnn-lstm | cross_paper_or_section | 4 | 1 | win |
| cross-04-pointer-ntm | cross_paper_or_section | 2 | 2 | tie |
| cross-05-rn-rmc | cross_paper_or_section | 4 | - | loss |
| cross-06-alexnet-resnet | cross_paper_or_section | - | 2 | win |
| cross-07-gpipe-method-result | cross_paper_or_section | 2 | 2 | tie |
| cross-08-scaling-allocation | cross_paper_or_section | 2 | 2 | tie |
| cross-09-dilated-method-result | cross_paper_or_section | 1 | 1 | tie |
| cross-10-mdl-intro-selection | cross_paper_or_section | - | 5 | win |
| cross-11-order-method-result | cross_paper_or_section | 2 | 1 | win |
| cross-12-coffee-complexity | cross_paper_or_section | 4 | 3 | win |

## M4 困难查询输入

以下是 old dev 中 S1 相比 B1 的退化题及其 trace 信号；它们不是运行时 gold 规则。

- `term-03-scaled-dot-product`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": false, "table_or_number_localization": false, "abbreviation": false, "cross_section": false, "cross_paper": false, "multiple_constraints": false, "first_context_incomplete": false}`
- `term-12-rnn-dropout`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": true, "table_or_number_localization": false, "abbreviation": true, "cross_section": false, "cross_paper": false, "multiple_constraints": false, "first_context_incomplete": false}`
- `method-04-preactivation`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": false, "table_or_number_localization": false, "abbreviation": true, "cross_section": false, "cross_paper": false, "multiple_constraints": false, "first_context_incomplete": false}`
- `method-08-deepspeech-bn`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": false, "table_or_number_localization": false, "abbreviation": true, "cross_section": false, "cross_paper": false, "multiple_constraints": false, "first_context_incomplete": true}`
- `method-12-vlae-information`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": false, "table_or_number_localization": false, "abbreviation": true, "cross_section": false, "cross_paper": false, "multiple_constraints": false, "first_context_incomplete": true}`
- `number-09-dilated-iou`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": false, "table_or_number_localization": true, "abbreviation": true, "cross_section": false, "cross_paper": false, "multiple_constraints": false, "first_context_incomplete": false}`
- `cross-02-resnet-identity`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": false, "table_or_number_localization": false, "abbreviation": false, "cross_section": false, "cross_paper": true, "multiple_constraints": true, "first_context_incomplete": true}`
- `cross-05-rn-rmc`：`{"dense_sparse_top_result_disagreement": null, "top_score_gap_small": false, "table_or_number_localization": false, "abbreviation": true, "cross_section": false, "cross_paper": true, "multiple_constraints": true, "first_context_incomplete": true}`

## 补充可观测案例

以下为 old dev 的真实 trace 补充；它们用于让 M4 在自己的 route/answer 数据上验证信号，不得直接把本表 gold 标签写成运行时规则。

| 信号 | 案例 | Trace 依据 |
| --- | --- | --- |
| dense/sparse Top 不一致 | `term-12-rnn-dropout` | dense 与 sparse 的首条 passage ID 不同；该题是 S1 loss。 |
| Top 分数差距小、缩写 | `term-12-rnn-dropout` | S1 loss，`top_score_gap_small=true`，标签含“缩写”。 |
| 表格/数字定位 | `number-09-dilated-iou` | S1 loss，标签含“表格、数字”，且 `table_or_number_localization=true`。 |
| 跨论文、多约束 | `cross-02-resnet-identity`、`cross-05-rn-rmc` | 两题均为 S1 loss，标签为“跨论文”，首轮 context 未覆盖全部 gold。 |
| 跨章节、首轮 context 不完整 | `cross-07-gpipe-method-result`、`cross-09-dilated-method-result` | 标签含“跨章节、表格、数字”；两题 context recall 都是 0.5，且 dense/sparse 首条不同。 |
