import pytest
from deep_researcher.config import Config
from deep_researcher.errors import ConfigValidationError


class TestConfigValidation:
    def test_valid_config(self):
        c = Config(start_year=2020, end_year=2025)
        assert c.start_year == 2020

    def test_start_year_after_end_year(self):
        with pytest.raises(ConfigValidationError, match="start_year"):
            Config(start_year=2025, end_year=2020)

    def test_start_year_too_old(self):
        with pytest.raises(ConfigValidationError, match="start_year"):
            Config(start_year=1800)

    def test_max_iterations_clamped(self):
        c = Config(max_iterations=100)
        assert c.max_iterations == 50  # clamped by __post_init__ before validate()

    def test_breadth_clamped(self):
        c = Config(breadth=10)
        assert c.breadth == 5
