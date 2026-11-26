# llm-technical-report-study

LLM 관련 논문과 기술 보고서(Technical Report)를 읽고 정리하는 스터디 레포입니다.

## 논문 목록

| # | 논문 | 연도 | 키워드 | 원문 | 정리 |
|---|------|------|--------|------|------|
| 01 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | 2021 | MoE, Sparse, Routing | [arXiv](https://arxiv.org/abs/2101.03961) | [정리](papers/01-switch-transformers/README.md) |
| 02 | Mixtral of Experts | 2024 | MoE, Top-2 Routing, Open-weight | [arXiv](https://arxiv.org/abs/2401.04088) | [정리](papers/02-mixtral-of-experts/README.md) |
| 03 | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | 2024 | MoE, Fine-grained Expert, Shared Expert | [arXiv](https://arxiv.org/abs/2401.06066) | [정리](papers/03-deepseekmoe/README.md) |
| 04 | DeepSeek-V3 Technical Report | 2024 | MoE, MLA, MTP, FP8, Aux-loss-free | [arXiv](https://arxiv.org/abs/2412.19437) | [정리](papers/04-deepseek-v3/README.md) |

## 구조

```
papers/
└── {번호}-{논문명}/
    ├── README.md   # 정리 본문
    └── images/     # figure, 다이어그램
```
