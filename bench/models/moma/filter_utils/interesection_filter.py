import abc


class IntersectionFilterInterface(abc.ABC):

    @abc.abstractmethod
    def filter(self, *dicts) -> list[dict]:
        pass


class IntersectionFilter(IntersectionFilterInterface):

    def filter(self, *dicts) -> list[dict]:
        """Intersects multiple dictionaries by their keys and returns the common keys and filtered dictionaries.

        Parameters:
        ----------
            - *dicts (dict): Dictionaries to intersect.

        Returns:
        --------
            - A list of dictionaries filtered by the common keys.
        """
        key_sets = [set(d.keys()) for d in dicts]
        common_keys = set.intersection(*key_sets)
        result = [{k: d[k] for k in common_keys} for d in dicts]

        return result
