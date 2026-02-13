import yaml
import textwrap
from pathlib import Path
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator

# Define Sub-Models for Sections
class IOConfig(BaseModel):
    """Configuration for Input/Output paths and file patterns."""
    input_dir: Path = Field(
        default=Path("data"), 
        description="Directory containing input NetCDF ensemble files and other input data."
    )
    output_dir: Path = Field(
        default=Path("output"), 
        description="Directory where all results (plots, maps, GIFs) will be saved."
    )
    ensemble_pattern: str = Field(
        default="troute_output_formulation_*.nc", 
        description="Glob pattern to match NetCDF ensemble member files within input_dir."
    )
    stats_file: Path = Field(
        default=Path("ensemble_stats.nc"),
        description="Filename for the pre-calculated ensemble statistics. Used to cache results."
    )
    hydrofabric_path: Optional[Path] = Field(
        default=None, 
        description="Path to the specific hydrofabric GeoPackage (.gpkg). If null, auto-detects first .gpkg in input_dir."
    )
    observations_file: Optional[Path] = Field(
        default=None, 
        description="Path to a CSV file containing USGS observations for validation."
    )
    auto_download_usgs: bool = Field(
        default=False,
        description="If True and observations_file is missing/null, download data from USGS for all gages in the domain."
    )
    save_downloaded_obs: Optional[Path] = Field(
        default=None,
        description="If provided, auto-downloaded USGS data will be saved to this specific path (e.g., 'data/usgs_cache.csv'). If null, data is not saved."
    )

    @field_validator("input_dir", "output_dir")
    def convert_to_path(cls, v):
        return Path(v) if v else v

class DataConfig(BaseModel):
    """Configuration for data slicing and subsetting."""
    time_slice: Optional[List[Union[int, str]]] = Field(
        default=None, 
        description="Subset the data by time. Can be indices [0, 48] or ISO dates ['2023-01-01', '2023-01-03']. Set to null to use all time."
    )
    feature_ids: Union[List[int], Literal["all"]] = Field(
        default="all", 
        description="List of integer Feature IDs to process. Use 'all' to process the entire domain."
    )

class StatsConfig(BaseModel):
    """Configuration for statistical calculations."""
    enabled: bool = Field(
        default=True,
        description="Whether to calculate statistics from the ensemble files. If False, the pipeline attempts to load pre-calculated stats from 'stats_file'."
    )
    quantiles: List[float] = Field(
        default=[0.05, 0.95], 
        description="Quantiles to calculate for uncertainty bands (0.0 to 1.0)."
    )
    metrics: List[str] = Field(
        default=["kge", "nse", "rmse"], 
        description="List of performance metrics to calculate against observations."
    )

    @field_validator("quantiles")
    def validate_quantiles(cls, v):
        if not all(0 <= q <= 1 for q in v):
            raise ValueError("Quantiles must be between 0 and 1.")
        return v

class HydrographConfig(BaseModel):
    """Settings for Hydrograph plots."""
    enabled: bool = Field(
        default=True, 
        description="Whether to generate hydrograph plots.")
    target_ids: List[int] = Field(
        default=[], 
        description="Specific Feature IDs to plot. If empty, the pipeline plots the first 5 found."
    )
    plot_uncertainty: bool = Field(
        default=True, 
        description="Include shaded uncertainty bands in the plot.")
    plot_members: bool = Field(
        default=False,
        description="If True, plots each individual ensemble member trace in a light color (spaghetti plot)."
    )

class StaticMapConfig(BaseModel):
    """Settings for static map generation."""
    enabled: bool = Field(True, description="Whether to generate static maps.")
    variables: List[str] = Field(
        default=["streamflow_mean"], 
        description="List of variables to map (e.g., 'streamflow_mean', 'velocity_mean')."
    )
    basemap: bool = Field(True, description="Add a contextily background map (requires internet).")

class InteractiveMapConfig(BaseModel):
    """Settings for HTML interactive maps."""
    enabled: bool = Field(True, description="Whether to generate an interactive Folium map.")
    variable: str = Field("streamflow_mean", description="Variable to display on the interactive map.")

class AnimationConfig(BaseModel):
    """Settings for generating GIFs."""
    enabled: bool = Field(False, description="Whether to generate an animation (time-intensive).")
    variable: str = Field("streamflow_mean", description="Variable to animate.")
    fps: int = Field(8, ge=1, le=60, description="Frames per second for the GIF.")
    log_scale: bool = Field(True, description="Use logarithmic color scaling (recommended for streamflow).")
    cmap: str = Field("hydro_flow", description="Colormap name (e.g., 'hydro_flow', 'viridis', 'coolwarm').")

class VizConfig(BaseModel):
    """Visualization grouping."""
    hydrographs: HydrographConfig = HydrographConfig()
    static_maps: StaticMapConfig = StaticMapConfig()
    interactive_map: InteractiveMapConfig = InteractiveMapConfig()
    animation: AnimationConfig = AnimationConfig()

# Main Config Model 
class TevalConfig(BaseModel):
    """Root configuration object for TEVAL."""
    io: IOConfig = IOConfig()
    data: DataConfig = DataConfig()
    stats: StatsConfig = StatsConfig()
    viz: VizConfig = VizConfig()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TevalConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

# Generators

def generate_default_config(path: str = "teval_config.yaml"):
    """
    Generates a default YAML configuration file based on the Pydantic model defaults.
    """
    # Create an instance with default values
    default_model = TevalConfig()
    
    # Convert to dict
    default_dict = default_model.model_dump()
    
    # Helper to clean Path objects into strings for YAML serialization
    def clean_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                clean_dict(v)
            elif isinstance(v, Path):
                d[k] = str(v)
    
    clean_dict(default_dict)
    
    # Write to file
    with open(path, "w") as f:
        yaml.dump(default_dict, f, sort_keys=False, default_flow_style=False)

def generate_config_help() -> str:
    """
    Introspects the TevalConfig model to generate a readable help guide.
    """
    lines = []
    lines.append("TEVAL CONFIGURATION GUIDE")
    lines.append("=" * 80)
    lines.append("This guide explains all available configuration options for the teval_config.yaml file.\n")

    # Iterate over the main sections (fields of TevalConfig)
    for section_name, field_info in TevalConfig.model_fields.items():
        # Get the sub-model class (e.g., IOConfig)
        section_model = field_info.annotation
        
        lines.append(f"SECTION: {section_name}")
        lines.append("-" * 40)
        
        # Check if it has a docstring
        if section_model.__doc__:
             lines.append(f"{section_model.__doc__}\n")

        # Iterate over fields in the sub-model
        for key, prop in section_model.model_fields.items():
            # Get default value
            default_val = prop.default
            
            # Format description with wrapping
            desc = prop.description or "No description provided."
            wrapped_desc = textwrap.fill(desc, width=70, initial_indent="    ", subsequent_indent="    ")

            lines.append(f"  • {key} (Default: {default_val})")
            lines.append(wrapped_desc)
            lines.append("")
        
        lines.append("")

    return "\n".join(lines)