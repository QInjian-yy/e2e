"""Legacy Mean-ResNet evaluation launcher."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.evaluate import main as _shared_main  # noqa: E402


def main(argv=None):
    return _shared_main(argv, expected_model="mean_resnet")


if __name__ == "__main__":
    main()

