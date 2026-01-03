import pandas as pd
from typing import List, Tuple, ClassVar
from pydantic import BaseModel, ConfigDict


class OptimisationModelBase(BaseModel):
    """
    A base class the extends pydantics BaseModel that provides generic
    parameters and functions for wrapping optimisation models.

    Classes that extend this class can be easily used by the optimiser tool

    Args:
        BaseModel: Pydantic BaseModel that this class extends

    """

    # class-level name attribute for solver identification
    name: ClassVar[str] = "Base Optimisation Model"

    time_limit: int = 3600
    model_config = ConfigDict(
        use_enum_values=True, validate_default=True, str_strip_whitespace=True
    )

    def optimise(self) -> Tuple[List[str], pd.DataFrame]:
        """
        The main optimise function of the model.  This should be overridden
        by any subclass implementing this class

        Returns:
            A tuple containing:
                List[str]: A list of warning messages produced by the optimisation model
                pd.DataFrame: A dataframe containing an optimised schedule
        """
        pass
