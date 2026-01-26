
import pandas as pd
import dataretrieval.nwis as nwis
from typing import List, Union, Optional

# Conversion constant: CFS to CMS
CFS_TO_CMS = 0.028316847

def fetch_usgs_streamflow(
    site_ids: List[str],
    start_date: str,
    end_date: str,
    to_cms: bool = True,
    to_utc: bool = True
) -> pd.DataFrame:
    """
    Fetches daily or instantaneous streamflow (parameter 00060) from USGS NWIS.
    
    Args:
        site_ids: List of USGS gage IDs (strings, e.g. ["01111500"]).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        to_cms: If True, converts from CFS to CMS.
        to_utc: If True, converts index to UTC timezone.
        
    Returns:
        pd.DataFrame: Index is Datetime, Columns are site_ids. Values are flow.
    """
    # 00060 = Discharge (cfs), 00065 = Gage height
    # iv = instantaneous values (usually 15-min)
    # dv = daily values
    # Try 'iv' first for high-res validation, fall back to 'dv' if needed. 
    # For t-route, 'iv' is usually preferred.
    
    print(f"Fetching USGS data for {len(site_ids)} sites from {start_date} to {end_date}...")
    
    try:
        df_flow = nwis.get_record(
            sites=site_ids, 
            service='iv', 
            start=start_date, 
            end=end_date, 
            parameterCd='00060'
        )
    except Exception as e:
        # Sometimes dataretrieval fails if no data found
        print(f"Error fetching from NWIS: {e}")
        return pd.DataFrame()

    if df_flow.empty:
        print("Warning: No USGS data returned for these sites/dates.")
        return pd.DataFrame()

    # Clean up the DataFrame
    # dataretrieval returns a messy frame with qualifiers/codes
    # The value column is usually '00060_Mean' or similar, but the index is the time.
    
    # We pivot to get shape: (Time x Sites)
    # The raw dataframe has a 'site_no' column and a value column.
    
    # Identify the value column (it usually ends in 00060)
    val_cols = [c for c in df_flow.columns if c.endswith('00060')]
    if not val_cols:
        # Fallback if specific column name varies
        val_cols = [c for c in df_flow.columns if "00060" in c and "cd" not in c]
    
    if not val_cols:
        raise ValueError("Could not identify streamflow value column in NWIS response.")
    
    val_col = val_cols[0]
    
    # Pivot so each column is a site
    df_pivot = df_flow.pivot_table(
        index=df_flow.index, 
        columns='site_no', 
        values=val_col
    )
    
    # Timezone conversion
    if to_utc:
        # USGS usually returns timezone-aware timestamps (e.g. America/New_York)
        # We convert to UTC.
        if df_pivot.index.tz is not None:
            df_pivot.index = df_pivot.index.tz_convert('UTC')
        else:
            # If naive, assume UTC or warn user
            print("Warning: USGS data returned timezone-naive. Assuming UTC.")
            df_pivot.index = df_pivot.index.tz_localize('UTC')

    # Unit conversion
    if to_cms:
        df_pivot = df_pivot * CFS_TO_CMS

    return df_pivot