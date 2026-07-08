# Table 6. Description and consensus constraint ablation

Key: green marks the overall best setting in each quality row; arrows on with-description cells indicate whether descriptions improve or reduce the corresponding no-description setting. Values are mean +/- std. Counts are not bolded because more constraints is not necessarily better.

| Dataset  | Side      | Metric    | Average no desc            | Average desc               | Consensus no desc   | Consensus desc      |
| -------- | --------- | --------- | -------------------------- | -------------------------- | ------------------- | ------------------- |
| bnlearn  | Forbidden | Count     | 19.800 +/- 13.902          | 25.467 +/- 24.387          | 8.000 +/- 6.603     | 8.167 +/- 1.722     |
| bnlearn  | Forbidden | Precision | **0.927 +/- 0.186**        | **0.973 +/- 0.041**        | **0.833 +/- 0.408** | **0.948 +/- 0.085** |
| bnlearn  | Forbidden | Recall    | **0.409 +/- 0.234**        | **0.472 +/- 0.219**        | **0.241 +/- 0.240** | **0.275 +/- 0.216** |
| bnlearn  | Forbidden | F1        | **0.533 +/- 0.260**        | **0.606 +/- 0.224**        | **0.338 +/- 0.316** | **0.392 +/- 0.281** |
| bnlearn  | Required  | Count     | 8.267 +/- 6.968            | 10.800 +/- 11.713          | 4.500 +/- 3.728     | 3.667 +/- 4.131     |
| bnlearn  | Required  | Precision | **0.468 +/- 0.288**        | **0.597 +/- 0.235**        | **0.375 +/- 0.327** | **0.505 +/- 0.422** |
| bnlearn  | Required  | Recall    | **0.445 +/- 0.352**        | **0.611 +/- 0.328**        | **0.303 +/- 0.388** | **0.281 +/- 0.375** |
| bnlearn  | Required  | F1        | 0.401 +/- 0.240            | **0.555 +/- 0.232** \*     | 0.284 +/- 0.284     | **0.321 +/- 0.337** |
| CauseNet | Forbidden | Count     | 17.593 +/- 13.021          | 20.641 +/- 16.604          | 4.852 +/- 4.835     | 4.074 +/- 3.425     |
| CauseNet | Forbidden | Precision | **0.952 +/- 0.132**        | **0.942 +/- 0.155**        | **0.855 +/- 0.339** | **0.877 +/- 0.317** |
| CauseNet | Forbidden | Recall    | **0.293 +/- 0.216** \*\*\* | **0.312 +/- 0.221** \*\*\* | 0.123 +/- 0.164     | 0.115 +/- 0.168     |
| CauseNet | Forbidden | F1        | **0.409 +/- 0.232** \*\*\* | **0.431 +/- 0.234** \*\*\* | 0.189 +/- 0.216     | 0.174 +/- 0.218     |
| CauseNet | Required  | Count     | 12.570 +/- 9.597           | 14.556 +/- 11.525          | 4.111 +/- 3.295     | 4.741 +/- 3.332     |
| CauseNet | Required  | Precision | **0.487 +/- 0.249**        | **0.464 +/- 0.230**        | **0.568 +/- 0.353** | **0.514 +/- 0.335** |
| CauseNet | Required  | Recall    | **0.486 +/- 0.255** \*\*\* | **0.516 +/- 0.242** \*\*\* | 0.244 +/- 0.210     | 0.278 +/- 0.244     |
| CauseNet | Required  | F1        | **0.450 +/- 0.206** \*\*   | **0.453 +/- 0.190** \*\*   | 0.314 +/- 0.228     | 0.334 +/- 0.249     |
