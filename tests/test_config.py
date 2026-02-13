import pytest
from pathlib import Path
from pydantic import ValidationError
from teval.config import TevalConfig, StatsConfig, AnimationConfig

def test_default_config_is_valid():
    """Ensure the default configuration instantiates without error."""
    config = TevalConfig()
    assert config.io.input_dir == Path("data")
    assert config.viz.animation.fps == 8

def test_path_conversion():
    """Ensure strings are automatically converted to Path objects."""
    config = TevalConfig(io={"input_dir": "/tmp/custom_data"})
    assert isinstance(config.io.input_dir, Path)
    assert config.io.input_dir == Path("/tmp/custom_data")

def test_quantiles_validation():
    """Ensure quantiles must be between 0 and 1."""
    # Valid
    StatsConfig(quantiles=[0.1, 0.9])
    
    # Invalid (> 1)
    with pytest.raises(ValidationError) as excinfo:
        StatsConfig(quantiles=[0.1, 1.5])
    assert "Quantiles must be between 0 and 1" in str(excinfo.value)

    # Invalid (< 0)
    with pytest.raises(ValidationError):
        StatsConfig(quantiles=[-0.1, 0.5])

def test_fps_validation():
    """Ensure FPS is within a reasonable range."""
    # Invalid (0 FPS)
    with pytest.raises(ValidationError):
        AnimationConfig(fps=0)
    
    # Invalid (too high)
    with pytest.raises(ValidationError):
        AnimationConfig(fps=120)

def test_load_from_yaml(tmp_path):
    """Test loading a real YAML file."""
    config_file = tmp_path / "test_config.yaml"
    yaml_content = """
    io:
      input_dir: "custom_input"
    viz:
      animation:
        fps: 20
        enabled: true
    """
    config_file.write_text(yaml_content)
    
    config = TevalConfig.from_yaml(config_file)
    
    assert config.io.input_dir == Path("custom_input")
    assert config.viz.animation.fps == 20
    assert config.viz.animation.enabled is True
    # Check default was preserved
    assert config.viz.hydrographs.enabled is True