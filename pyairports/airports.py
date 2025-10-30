"""Minimal subset of airport records for outlines-guided decoding."""
from typing import List, Tuple

# Each entry mirrors the layout expected by outlines.types.airports: a tuple where
# index 3 contains the IATA code. The remaining positions are placeholders.
AIRPORT_LIST: List[Tuple[str, str, str, str]] = [
    ("United States", "St. Louis", "KSTL", "STL"),
    ("United States", "Chicago", "KORD", "ORD"),
    ("United States", "Los Angeles", "KLAX", "LAX"),
    ("United States", "New York", "KJFK", "JFK"),
]

__all__ = ["AIRPORT_LIST"]
