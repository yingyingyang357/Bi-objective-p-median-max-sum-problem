"""Data classes for problem instance."""

import pandas as pd
import numpy as np
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class BaseData(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True, validate_default=True, str_strip_whitespace=True
    )


class Location(BaseData):
    """Base dataclass for location data.

    Attributes:
        index (int): The index of the location.
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        name (str, optional): The name of the location. Defaults to None.
    """

    index: int
    latitude: float
    longitude: float
    name: Optional[str] = None


class LocationData(BaseData):
    number: int
    locations: List[Location]


class ProblemData(BaseData):
    """Base dataclass for problem instance data.

    Attributes:
        number_of_facilities (int): The number of facilities.
        p_value (int, optional): An optional parameter for the problem. Defaults to None.
        location_data (LocationData, optional): The location data for the problem. Defaults to None.
        distance_matrix (np.ndarray, optional): A precomputed distance matrix as numpy array. Defaults to None.
        dispersion_threshold (float, optional): The dispersion threshold for the problem. Defaults to 0.0.
        time_limit (int, optional): The time limit for solving the problem in seconds. Defaults to 3600.
    """

    number_of_facilities: int
    p_value: Optional[List[int]] = None
    location_data: Optional[LocationData] = None
    distance_matrix: Optional[np.ndarray] = None
    dispersion_threshold: float = 0.0
    time_limit: int = 3600

    class Config:
        arbitrary_types_allowed = True  # allow numpy arrays

    def __init__(self, data=None, **kwargs):
        """Initialize ProblemData from a DataFrame and optional arguments."""
        if data is not None and isinstance(data, pd.DataFrame):
            # parse DataFrame into structured data
            parsed_data = self._parse_dataframe(data)
            # merge with any additional kwargs
            parsed_data.update(kwargs)
            super().__init__(**parsed_data)
        else:
            raise ValueError("Data must be provided as a pandas DataFrame.")

    @classmethod
    def _parse_dataframe(cls, df: pd.DataFrame) -> dict:
        """Parse DataFrame into ProblemData attributes."""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # If DataFrame is a full distance matrix
        if df.shape[0] == df.shape[1]:
            # Assume it's a square distance matrix
            distance_matrix = df.values.astype(np.float64)
            n_locations = df.shape[0]
            return {
                "number_of_facilities": n_locations,
                "distance_matrix": distance_matrix,
            }
        
        # If DataFrame is in (i, j, distance) format
        elif len(df.columns) == 3:
            # data is given as (instance, instance, distance) format
            # we get distance matrix instead of locations
            distance_matrix_df = df.pivot(
                index=df.columns[0], columns=df.columns[1], values=df.columns[2]
            )
            # make symmetric and fill NaN with 0
            distance_matrix_df = distance_matrix_df.fillna(distance_matrix_df.T).fillna(
                0
            )
            # convert to numpy array
            distance_matrix = distance_matrix_df.values.astype(np.float64)

            # get unique locations count
            n_locations = len(distance_matrix_df.index)

            return {
                "number_of_facilities": n_locations,
                "distance_matrix": distance_matrix,
            }

        else:
            # create Location objects from DataFrame rows
            locations = []
            for idx, row in df.iterrows():
                location = Location(
                    index=row.get("instance", idx),
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    name=row.get("name", None),
                )
                locations.append(location)

            # Create LocationData
            location_data = LocationData(number=len(locations), locations=locations)

            return {"number_of_facilities": len(df), "location_data": location_data}

    @staticmethod
    def find_closest_and_up_from_list(lst: List[float], a: float) -> Optional[float]:
        """Find the smallest element in a list that is greater than or equal to a threshold.

        This method is used in bi-section search methods for p-dispersion problems, to find the next
        candidate value from a sorted list of distances.

        Args:
            lst (List[float]): List of numerical values (typically distances).
                              For optimal performance, the list should be sorted.
            a (float): Threshold value to search against.

        Returns:
            Optional[float]: The smallest element in lst that is >= a.
                           Returns None if no such element exists.
        """
        return min((x for x in lst if x >= a), default=None)
