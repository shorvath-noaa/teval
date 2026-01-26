import pytest
import matplotlib.pyplot as plt
from teval import viz, stats

def test_hydrograph_runs_without_error(fake_ensemble):
    """Smoke test for hydrograph plotting."""
    ds = fake_ensemble
    # Add a fake feature_id so .sel() works
    ds = ds.expand_dims(feature_id=[12345])
    
    stats_ds = stats.calculate_basics(ds)
    
    fig, ax = plt.subplots()
    try:
        viz.static.hydrograph(stats_ds, feature_id=12345, var_name="streamflow", ax=ax)
    except Exception as e:
        pytest.fail(f"Hydrograph plotting failed with error: {e}")
    finally:
        plt.close(fig)

def test_hydrograph_handles_missing_feature(fake_ensemble):
    """Test graceful failure on bad ID."""
    ds = fake_ensemble
    ds = ds.expand_dims(feature_id=[12345])
    stats_ds = stats.calculate_basics(ds)
    
    # Should print an error but not crash
    viz.static.hydrograph(stats_ds, feature_id=99999)