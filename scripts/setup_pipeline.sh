#!/usr/bin/env bash
# One-time setup of the training/eval environment (week 1, ~20-30 min, several GB of disk).
#
# - Checks out the pinned Daniels pipeline submodule.
# - Creates .venv with Python 3.12 (the pipeline requires >= 3.12).
# - Installs the pipeline's requirements, EXCEPT its pinned tinker-cookbook fork: that fork
#   (Aug 2026) predates the DeepSeek-V4.1 renderers. Upstream tinker-cookbook is installed
#   instead, pinned to a commit that has them. The fork's only change is pipelined DPO
#   requests, which GRPO does not use.
# - Makes the pipeline importable (a .pth file) and installs this package.
set -euo pipefail
cd "$(dirname "$0")/.."

PIPELINE=third_party/iter-dpo-reward-hacking-generalization
COOKBOOK_COMMIT=70767c738ba5f523b66210a70e50b0dc14c9fc1a  # upstream main, 2026-10-09

git submodule update --init "$PIPELINE"

[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

grep -v 'tinker-cookbook' "$PIPELINE/requirements.txt" > .venv/pipeline-requirements.txt
uv pip install -r .venv/pipeline-requirements.txt
uv pip install "tinker-cookbook[inspect] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@${COOKBOOK_COMMIT}"
uv pip install -e . pytest

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/$PIPELINE" > "$SITE/rhinoc-pipeline.pth"

python - <<'EOF'
import rewardhacking_training, tinker, tinker_cookbook
from tinker_cookbook.renderers import get_renderer  # noqa: F401
print("pipeline import OK")
EOF

for tool in node ruby lua; do
  command -v "$tool" >/dev/null || echo "WARNING: '$tool' not found; impossible_mbpp tests need it (brew install $tool)"
done
[ -f .env ] || echo "NOTE: copy .env.example to .env and fill in the keys"
