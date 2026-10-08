from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import pytest
from generate_fixtures import sensor_header

@pytest.fixture
def header():
    return sensor_header()
