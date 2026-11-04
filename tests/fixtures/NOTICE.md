# Test fixtures: sources

These files are small excerpts used only by the CPU test suite. Each one keeps the license of its source; see the
linked model or dataset card.

## Tokenizers

| Directory | Source |
|---|---|
| `tokenizers/qwen3_1_7b/` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (`tokenizer.json`, `tokenizer_config.json`) |
| `tokenizers/r1_distill_qwen_1_5b/` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |

## Dataset rows (`datasets/*.jsonl`)

A few rows per dataset, fetched from the Hugging Face datasets server, so the loaders can be tested offline.

| File | Source |
|---|---|
| `aime_2024.jsonl` | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| `aime_2025.jsonl` | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) |
| `aime_2026.jsonl` | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) |
| `hmmt_feb_2026.jsonl` | [MathArena/hmmt_feb_2026](https://huggingface.co/datasets/MathArena/hmmt_feb_2026) |
| `amc23.jsonl` | [math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23) |
| `math500.jsonl` | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| `math_train.jsonl` | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| `deepscaler.jsonl` | [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) |
| `gpqa_diamond.jsonl` | **Synthetic.** It only mimics the column layout of [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa). GPQA asks that its questions not be published in plain text, so no real GPQA rows are stored here. |

`deer_output.jsonl` is a synthetic example of the output format of the official DEER repository.
