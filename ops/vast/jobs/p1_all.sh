# P1 GPU steps with checkpointed stages and explicit test_main approval.
source ops/vast/jobs/common.sh
need_approval
python3 ops/vast/p1_run.py --verify-handoff
TTALLOC_DTYPE=bfloat16 python3 -m pytest -m gpu tests/gpu -q

# Keep the upstream grading dependencies out of the model and research-grading environment.
scorer=third_party/deer-scorer
python3 -m pip install --target "$scorer" 'sympy==1.12' 'antlr4-python3-runtime==4.11.1' 'word2number==1.1' 'regex==2026.9.29'
python3 -m pip install --target "$scorer/antlr47" --no-deps 'latex2sympy2==1.9.1' 'antlr4-python3-runtime==4.7.2'
python3 -c 'from ttalloc.baseline.deer_official import ensure_repo; from ttalloc.config import load_config; ensure_repo(load_config("configs/experiments/p1_safe.yaml"))'
python3 -I ops/vast/deer_score.py --smoke
python3 ops/vast/p1_run.py
