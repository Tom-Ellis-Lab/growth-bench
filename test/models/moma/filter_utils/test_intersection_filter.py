from bench.models.moma.filter_utils import interesection_filter


def test_intersection_filter():
    input = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6, "d": 7},
        {"a": 8, "b": 9, "c": 10, "e": 11},
    ]
    expected_output = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
        {"a": 8, "b": 9, "c": 10},
    ]
    filter = interesection_filter.IntersectionFilter()
    assert filter.filter(*input) == expected_output
