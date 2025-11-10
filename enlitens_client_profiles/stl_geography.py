"""Curated St. Louis regional geography reference data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class STLRegionData:
    core_neighborhoods: List[str]
    north_county_municipalities: List[str]
    south_county_municipalities: List[str]
    west_county_municipalities: List[str]
    metro_east_illinois: List[str]
    jefferson_county: List[str]
    st_charles_county: List[str]

    @property
    def all_municipalities(self) -> List[str]:
        combined: List[str] = []
        for bucket in (
            self.core_neighborhoods,
            self.north_county_municipalities,
            self.south_county_municipalities,
            self.west_county_municipalities,
            self.metro_east_illinois,
            self.jefferson_county,
            self.st_charles_county,
        ):
            combined.extend(bucket)
        return combined


STL_REGION_DATA = STLRegionData(
    core_neighborhoods=[
        "Central West End",
        "Soulard",
        "The Grove",
        "Tower Grove South",
        "Old North St. Louis",
        "Benton Park",
        "Shaw",
        "Lafayette Square",
        "Hyde Park",
        "Downtown West",
        "Carondelet",
        "Forest Park Southeast",
        "Debaliviere Place",
    ],
    north_county_municipalities=[
        "Florissant",
        "Ferguson",
        "Jennings",
        "Hazelwood",
        "Dellwood",
        "Berkeley",
        "Bel-Nor",
        "Black Jack",
        "Spanish Lake",
        "Riverview",
        "Northwoods",
        "Cool Valley",
    ],
    south_county_municipalities=[
        "Affton",
        "Mehlville",
        "Oakville",
        "Lemay",
        "Sunset Hills",
        "Kirkwood",
        "Shrewsbury",
        "Webster Groves",
        "Crestwood",
        "Maplewood",
        "Richmond Heights",
    ],
    west_county_municipalities=[
        "Chesterfield",
        "Ballwin",
        "Manchester",
        "Wildwood",
        "Ellisville",
        "Des Peres",
        "Town and Country",
        "Maryland Heights",
        "Creve Coeur",
        "Sunset Hills",
    ],
    metro_east_illinois=[
        "Alton",
        "Edwardsville",
        "Collinsville",
        "Granite City",
        "Belleville",
        "East St. Louis",
        "O'Fallon (IL)",
        "Fairview Heights",
        "Shiloh",
        "Godfrey",
        "Glen Carbon",
    ],
    jefferson_county=[
        "Arnold",
        "Festus",
        "Crystal City",
        "Imperial",
        "Herculaneum",
        "Pevely",
        "De Soto",
        "Barnhart",
    ],
    st_charles_county=[
        "St. Charles",
        "St. Peters",
        "O'Fallon (MO)",
        "Wentzville",
        "Lake St. Louis",
        "Cottleville",
        "Dardenne Prairie",
        "Defiance",
        "New Melle",
    ],
)


def build_geographic_context() -> Dict[str, List[str]]:
    """Provide a dictionary summarising regional clusters for prompts."""

    return {
        "core_city_neighborhoods": STL_REGION_DATA.core_neighborhoods,
        "north_county": STL_REGION_DATA.north_county_municipalities,
        "south_county": STL_REGION_DATA.south_county_municipalities,
        "west_county": STL_REGION_DATA.west_county_municipalities,
        "metro_east": STL_REGION_DATA.metro_east_illinois,
        "jefferson_county": STL_REGION_DATA.jefferson_county,
        "st_charles_county": STL_REGION_DATA.st_charles_county,
    }

