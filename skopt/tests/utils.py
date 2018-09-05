"""Test utils
"""


def assert_greater_list(e, others):
    """Asserts whether the element e is strictly greater than any element e in the others list
    """
    for o in others:
        assert e > o


def assert_less_list(e, others):
    """Asserts whether the element e is strictly less than any element e in the others list
    """
    for o in others:
        assert e < o

