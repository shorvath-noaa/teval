from abc import ABC, abstractmethod
import xarray as xr

class Ensembler(ABC):
    """
    Abstract Base Class for all ensembling strategies.
    
    All subclasses must implement the `process` method, which takes 
    a raw ensemble Dataset and returns a processed statistics Dataset.
    """

    @abstractmethod
    def process(self, ds_ensemble: xr.Dataset) -> xr.Dataset:
        """
        Process the raw ensemble into a statistical summary.

        Args:
            ds_ensemble (xr.Dataset): Input dataset containing ensemble members.
                                      Must have 'time', 'feature_id', and a member dim.

        Returns:
            xr.Dataset: A dataset with collapsed member dimension (e.g., mean, median).
        """
        pass