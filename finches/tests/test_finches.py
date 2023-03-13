"""
Unit and regression test for the finches package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import finches


def test_finches_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "finches" in sys.modules
