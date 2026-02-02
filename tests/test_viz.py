import pytest
import matplotlib.pyplot as plt
import teval.viz.static as tviz
import teval.viz.interactive as tinteractive
import teval.stats as stats
import os

def test_hydrograph_runs_without_error(fake_ensemble):
    """Smoke test for hydrograph plotting."""
    ds = fake_ensemble
    stats_ds = stats.calculate_basics(ds)
    
    fig, ax = plt.subplots()
    try:
        tviz.hydrograph(stats_ds, feature_id=12345, ax=ax)
    except Exception as e:
        pytest.fail(f"Hydrograph plotting failed with error: {e}")
    finally:
        plt.close(fig)

def test_hydrograph_handles_missing_feature(fake_spatial_ensemble):
    """Test graceful failure on bad ID."""
    ds = fake_spatial_ensemble
    stats_ds = stats.calculate_basics(ds)
    
    fig, ax = plt.subplots()
    tviz.hydrograph(stats_ds, feature_id=99999, ax=ax)
    plt.close(fig)
    
def test_map_network_runs(fake_spatial_ensemble, fake_hydrofabric):
    """Test static map generation."""
    ds = fake_spatial_ensemble
    stats_ds = stats.calculate_basics(ds)
    
    fig, ax = plt.subplots()
    
    # Run mapping function
    tviz.map_network(
        fake_hydrofabric,
        stats_ds,
        var_name="streamflow_mean",
        ax=ax,
        add_basemap=False 
    )
    
    # Check that we plotted something
    assert len(ax.collections) > 0 or len(ax.lines) > 0
    plt.close(fig)

def test_map_network_handles_empty_merge(fake_spatial_ensemble, fake_hydrofabric):
    """Test that it doesn't crash if IDs don't match."""
    ds = fake_spatial_ensemble.assign_coords(feature_id=[99998, 99999])
    stats_ds = stats.calculate_basics(ds)
    
    fig, ax = plt.subplots()
    tviz.map_network(
        fake_hydrofabric,
        stats_ds,
        var_name="streamflow_mean",
        ax=ax,
        add_basemap=False
    )
    plt.close(fig)

def test_interactive_map_generation(fake_spatial_ensemble, fake_hydrofabric, tmp_path):
    """Test interactive map generation and HTML saving."""
    ds = fake_spatial_ensemble
    stats_ds = stats.calculate_basics(ds)
    
    output_file = tmp_path / "test_map.html"
    
    m = tinteractive.map_folium(
        fake_hydrofabric,
        stats_ds,
        var_name="streamflow_mean",
        output_html=str(output_file)
    )
    
    assert m is not None
    assert output_file.exists()
    assert output_file.stat().st_size > 0