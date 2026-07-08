# Table 7. Description and consensus constraint ablation excluding zero-constraint cases

Key: same formatting as Table 4, but each side/metric excludes rows where the corresponding required/forbidden constraint set is empty.

| Dataset  | Side      | Metric    | Average no desc            | Average desc               | Consensus no desc          | Consensus desc         |
| -------- | --------- | --------- | -------------------------- | -------------------------- | -------------------------- | ---------------------- |
| bnlearn  | Forbidden | Count     | 20.483 +/- 13.627          | 25.467 +/- 24.387          | 9.600 +/- 5.941            | 8.167 +/- 1.722        |
| bnlearn  | Forbidden | Precision | 0.959 +/- 0.062            | 0.973 +/- 0.041            | **1.000 +/- 0.000** \*\*\* | **0.948 +/- 0.085**    |
| bnlearn  | Forbidden | Recall    | **0.423 +/- 0.225**        | **0.472 +/- 0.219**        | **0.289 +/- 0.234**        | **0.275 +/- 0.216**    |
| bnlearn  | Forbidden | F1        | **0.551 +/- 0.244**        | **0.606 +/- 0.224**        | **0.405 +/- 0.300**        | **0.392 +/- 0.281**    |
| bnlearn  | Required  | Count     | 9.920 +/- 6.448            | 11.172 +/- 11.738          | 6.750 +/- 1.708            | 5.500 +/- 3.873        |
| bnlearn  | Required  | Precision | **0.562 +/- 0.213**        | **0.618 +/- 0.210**        | **0.562 +/- 0.195**        | **0.757 +/- 0.206**    |
| bnlearn  | Required  | Recall    | **0.535 +/- 0.316**        | **0.632 +/- 0.312**        | **0.454 +/- 0.399**        | **0.422 +/- 0.394**    |
| bnlearn  | Required  | F1        | **0.482 +/- 0.172**        | **0.574 +/- 0.211**        | **0.426 +/- 0.233**        | **0.481 +/- 0.294**    |
| CauseNet | Forbidden | Count     | 17.857 +/- 12.937          | 20.795 +/- 16.569          | 5.574 +/- 4.777            | 4.583 +/- 3.293        |
| CauseNet | Forbidden | Precision | 0.966 +/- 0.061            | 0.949 +/- 0.132            | **0.982 +/- 0.069**        | **0.986 +/- 0.053** \* |
| CauseNet | Forbidden | Recall    | **0.297 +/- 0.215** \*\*\* | **0.314 +/- 0.220** \*\*\* | 0.141 +/- 0.168            | 0.129 +/- 0.172        |
| CauseNet | Forbidden | F1        | **0.415 +/- 0.228** \*\*\* | **0.434 +/- 0.231** \*\*\* | 0.217 +/- 0.218            | 0.196 +/- 0.222        |
| CauseNet | Required  | Count     | 12.664 +/- 9.571           | 14.664 +/- 11.499          | 4.531 +/- 3.170            | 5.020 +/- 3.216        |
| CauseNet | Required  | Precision | 0.491 +/- 0.247            | 0.468 +/- 0.227            | **0.626 +/- 0.317** \*\*   | **0.544 +/- 0.320**    |
| CauseNet | Required  | Recall    | **0.490 +/- 0.252** \*\*\* | **0.520 +/- 0.239** \*\*\* | 0.269 +/- 0.204            | 0.294 +/- 0.241        |
| CauseNet | Required  | F1        | **0.454 +/- 0.203** \*\*   | **0.457 +/- 0.187** \*\*   | 0.346 +/- 0.214            | 0.354 +/- 0.242        |
