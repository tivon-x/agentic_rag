# M3 冻结检索逐题结果

数据集：`evals/datasets/retrieval_v2_core.jsonl`
SHA-256：`e1da7d23d352cd17a1601f56280a5c9820ff81002a36dc5ad786cb3a8f90c936`

`—` 表示 gold passage 未进入统一的 rerank top-10。B1→B2 只比较首个 gold rank；
两者都未命中记为 tie。完整阶段 trace 保存在
`artifacts/evals/v2_core/*/report.json`。

| Case | Category | B0 rank | B1 rank | B2 rank | B3 rank | B1→B2 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| term-01-transformer | exact_term_definition | 6 | — | — | — | tie |
| term-02-multi-head | exact_term_definition | 6 | 4 | 10 | 10 | loss |
| term-03-scaled-dot-product | exact_term_definition | 7 | 6 | 7 | 7 | loss |
| term-04-positional-encoding | exact_term_definition | 1 | — | — | — | tie |
| term-05-lstm-dependency | exact_term_definition | 1 | 1 | 1 | 1 | tie |
| term-06-cell-state | exact_term_definition | 1 | 1 | 1 | 1 | tie |
| term-07-pointer-network | exact_term_definition | 5 | — | — | — | tie |
| term-08-residual-learning | exact_term_definition | — | — | — | — | tie |
| term-09-dilated-convolution | exact_term_definition | 1 | 1 | 3 | 3 | loss |
| term-10-mpnn | exact_term_definition | 1 | 7 | — | — | loss |
| term-11-ntm-memory | exact_term_definition | 2 | 3 | 5 | 5 | loss |
| term-12-rnn-dropout | exact_term_definition | 6 | 5 | 1 | 1 | win |
| method-01-transformer-encoder | method_section_location | 1 | 6 | 2 | 2 | win |
| method-02-gpipe-microbatch | method_section_location | 3 | 5 | 5 | 5 | tie |
| method-03-resnet-bottleneck | method_section_location | 3 | 4 | — | — | loss |
| method-04-preactivation | method_section_location | 6 | 1 | 2 | 2 | loss |
| method-05-relation-network | method_section_location | 1 | 4 | 4 | 4 | tie |
| method-06-rmc | method_section_location | — | 6 | — | — | loss |
| method-07-nmt-alignment | method_section_location | — | — | — | — | tie |
| method-08-deepspeech-bn | method_section_location | — | 9 | 1 | 1 | win |
| method-09-mdl-selection | method_section_location | — | 5 | — | — | loss |
| method-10-alexnet-relu | method_section_location | 1 | 3 | 2 | 2 | win |
| method-11-read-process-write | method_section_location | — | 1 | 1 | 1 | tie |
| method-12-vlae-information | method_section_location | — | — | 9 | 9 | win |
| number-01-transformer-bleu | experiment_number_table | 3 | 2 | 4 | 4 | loss |
| number-02-alexnet-error | experiment_number_table | — | — | 10 | 10 | win |
| number-03-resnet-error | experiment_number_table | 4 | — | — | — | tie |
| number-04-gpipe-amoeba | experiment_number_table | 2 | 2 | 3 | 3 | loss |
| number-05-rnn-penn | experiment_number_table | 5 | 5 | 4 | 4 | win |
| number-06-nmt-bleu | experiment_number_table | 9 | — | 1 | 1 | win |
| number-07-relation-clevr | experiment_number_table | — | — | — | — | tie |
| number-08-deepspeech-wer | experiment_number_table | 1 | 1 | 1 | 1 | tie |
| number-09-dilated-iou | experiment_number_table | 2 | 1 | 1 | 1 | tie |
| number-10-vlae-nll | experiment_number_table | 8 | 7 | — | — | loss |
| number-11-rmc-accuracy | experiment_number_table | 7 | — | — | — | tie |
| number-12-pointer-accuracy | experiment_number_table | — | — | — | — | tie |
| cross-01-transformer-nmt | cross_paper_or_section | 3 | 6 | 5 | 5 | win |
| cross-02-resnet-identity | cross_paper_or_section | 5 | 3 | 1 | 1 | win |
| cross-03-rnn-lstm | cross_paper_or_section | 1 | 2 | 6 | 6 | loss |
| cross-04-pointer-ntm | cross_paper_or_section | 3 | 4 | 3 | 3 | win |
| cross-05-rn-rmc | cross_paper_or_section | — | — | 2 | 2 | win |
| cross-06-alexnet-resnet | cross_paper_or_section | 1 | — | — | — | tie |
| cross-07-gpipe-method-result | cross_paper_or_section | 5 | 2 | 3 | 3 | loss |
| cross-08-scaling-allocation | cross_paper_or_section | 1 | 1 | 3 | 3 | loss |
| cross-09-dilated-method-result | cross_paper_or_section | 5 | 1 | 1 | 1 | tie |
| cross-10-mdl-intro-selection | cross_paper_or_section | 7 | 4 | — | — | loss |
| cross-11-order-method-result | cross_paper_or_section | 1 | 1 | 2 | 2 | loss |
| cross-12-coffee-complexity | cross_paper_or_section | 3 | 3 | — | 10 | loss |
