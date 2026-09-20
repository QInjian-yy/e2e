"""Legacy Mean-ResNet launcher; shared implementation lives in training.train."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import *  # noqa: F401,F403,E402
from training.train import main as _shared_main  # noqa: E402


def main(argv=None):
    return _shared_main(argv, default_model="mean_resnet")


if __name__ == "__main__":
    main()

