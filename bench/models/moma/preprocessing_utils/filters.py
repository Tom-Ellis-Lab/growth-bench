import abc
import enum

import pandas as pd


class FilterTypes(enum.Enum):
    INTERSECTION = "intersection"


class FilterFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_filter(self, filter_type: "FilterTypes") -> "Filter":
        """
        Create a filter based on the filter type.

        Parameters
        ----------
        filter_type : FilterTypes
            The type of filter to create.

        Returns
        -------
        Filter
            The filter.
        """
        pass


class FilterFactory(FilterFactoryInterface):
    def create_filter(self, filter_type: "FilterTypes") -> "Filter":
        """
        Create a filter based on the filter type.

        Parameters
        ----------
        filter_type : FilterTypes
            The type of filter to create.

        Returns
        -------
        Filter
            The filter.
        """
        if filter_type == FilterTypes.INTERSECTION:
            return IntersectionFilter()
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")


class Filter(abc.ABC):
    @abc.abstractmethod
    def filter_data(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Filter the dataframes.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The dataframes to filter.

        Returns
        -------
        dict[str, pd.DataFrame]
            The filtered dataframes.
        """
        pass


class IntersectionFilter(Filter):
    def filter_data(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Filter the dataframes by the intersection of the indices.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The dataframes to filter.

        Returns
        -------
        dict[str, pd.DataFrame]
            The filtered dataframes.

        """
        if len(data) < 2:
            raise ValueError(
                "At least two datasets are required to perform intersection."
            )

        # Find the common indices across all dataframes
        common_knockouts = set(data[next(iter(data))].index)
        for df in data.values():
            common_knockouts &= set(df.index)

        # Filter each dataframe by the common indices
        result = {
            name: df[df.index.isin(common_knockouts)] for name, df in data.items()
        }

        return result
