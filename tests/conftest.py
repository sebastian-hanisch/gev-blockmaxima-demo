"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Mini-Instanz: fünf Block-Maxima 2, 3, 5, 6, 9 (Handrechnung der L-Momente: l1 = 5, l2 = 17/10, l3 = 3/10, τ3 = 3/17)."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture
def mini():
    return [2.0, 3.0, 5.0, 6.0, 9.0]
