import logging
import os
import glob

import cooler
import numpy as np
import pytest
import itertools

from pathlib import Path

logger = logging.getLogger(__name__)


@pytest.mark.e2e()
def test_end2end_regression():
    """
    end2end validation with runtime calling (slow).
    """
    assert False
