"""Curated St. Louis regional context data for downstream agents.

The dataset intentionally emphasises municipal diversity so that
client-facing agents can reference neighbourhoods and nearby towns
instead of repeating the generic "St. Louis" label.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


STL_CITY_NEIGHBORHOODS: List[str] = [
    "Downtown",
    "Downtown West",
    "Near North Riverfront",
    "Old North St. Louis",
    "Carr Square",
    "Columbus Square",
    "St. Louis Place",
    "JeffVanderLou",
    "Midtown",
    "Grand Center",
    "Central West End",
    "Forest Park Southeast",
    "Cortex Innovation District",
    "DeBaliviere Place",
    "Skinker-DeBaliviere",
    "The Ville",
    "Greater Ville",
    "Fairground",
    "Hyde Park",
    "Baden",
    "O'Fallon",
    "North Pointe",
    "Walnut Park East",
    "Walnut Park West",
    "Kingsway East",
    "Kingsway West",
    "Academy",
    "Hamilton Heights",
    "Wells-Goodfellow",
    "Visitation Park",
    "Penrose",
    "Mark Twain",
    "College Hill",
    "Gravois Park",
    "Marine Villa",
    "Cherokee Street",
    "Benton Park",
    "Benton Park West",
    "Soulard",
    "Lafayette Square",
    "Fox Park",
    "Compton Heights",
    "Tower Grove East",
    "Tower Grove South",
    "Shaw",
    "Botanical Heights",
    "Forest Park",
    "Southwest Garden",
    "The Hill",
    "Clifton Heights",
    "Ellendale",
    "Franz Park",
    "Hi-Pointe",
    "Cheltenham",
    "Dogtown",
    "St. Louis Hills",
    "Southampton",
    "Princeton Heights",
    "Bevo Mill",
    "Dutchtown",
    "Mount Pleasant",
    "Carondelet",
    "Holly Hills",
]


STL_COUNTY_MUNICIPALITIES: List[str] = [
    "Florissant",
    "Hazelwood",
    "Ferguson",
    "Berkeley",
    "Kinloch",
    "Jennings",
    "Dellwood",
    "Calverton Park",
    "Cool Valley",
    "Bel-Ridge",
    "Bel-Nor",
    "Hanley Hills",
    "Normandy",
    "Pagedale",
    "Vinita Park",
    "Vinita Terrace",
    "Uplands Park",
    "Beverly Hills",
    "Country Club Hills",
    "Northwoods",
    "Pasadena Hills",
    "Pasadena Park",
    "Glen Echo Park",
    "Bellerive",
    "Woodson Terrace",
    "Edmundson",
    "St. Ann",
    "Bridgeton",
    "Maryland Heights",
    "Creve Coeur",
    "Olivette",
    "Ladue",
    "Frontenac",
    "Huntleigh",
    "Clayton",
    "Richmond Heights",
    "Maplewood",
    "Brentwood",
    "Rock Hill",
    "Glendale",
    "Kirkwood",
    "Webster Groves",
    "Shrewsbury",
    "Sunset Hills",
    "Des Peres",
    "Town and Country",
    "Manchester",
    "Ballwin",
    "Ellisville",
    "Wildwood",
    "Chesterfield",
    "Eureka",
    "Pacific",
    "Valley Park",
    "Twin Oaks",
    "Grantwood Village",
    "Marlborough",
    "Oakland",
    "Lakeshire",
    "Wilbur Park",
    "Moline Acres",
    "Spanish Lake",
    "Affton",
    "Lemay",
    "Mehlville",
    "Sappington",
    "Oakville",
]


ST_CHARLES_MUNICIPALITIES: List[str] = [
    "St. Charles",
    "St. Peters",
    "O'Fallon",
    "Wentzville",
    "Lake Saint Louis",
    "Cottleville",
    "Dardenne Prairie",
    "New Melle",
    "Weldon Spring",
    "Weldon Spring Heights",
    "Harvester",
    "Defiance",
    "St. Paul",
    "Josephville",
    "Portage Des Sioux",
    "Augusta",
    "Matson",
]


JEFFERSON_MUNICIPALITIES: List[str] = [
    "Arnold",
    "Byrnes Mill",
    "Crystal City",
    "De Soto",
    "Festus",
    "Herculaneum",
    "Hillsboro",
    "Kimmswick",
    "Olympian Village",
    "Pevely",
    "Scotsdale",
    "Cedar Hill Lakes",
    "Lake Tekawitha",
    "Parkdale",
    "Peaceful Village",
    "Barnhart",
    "Cedar Hill",
    "High Ridge",
    "Imperial",
    "LaBarque Creek",
    "Murphy",
    "Antonia",
    "House Springs",
    "Otto",
    "Mapaville",
    "Horine",
]


METRO_EAST_COMMUNITIES: List[str] = [
    "East St. Louis",
    "Belleville",
    "Collinsville",
    "Edwardsville",
    "Granite City",
    "Alton",
    "Fairview Heights",
    "O'Fallon (IL)",
    "Swansea",
    "Shiloh",
    "Glen Carbon",
    "Madison",
    "Brooklyn",
    "Cahokia Heights",
    "Centreville",
    "Sauget",
    "Venice",
    "Hartford",
    "Wood River",
    "Roxana",
    "Troy",
    "Highland",
    "Mascoutah",
    "Waterloo",
    "Columbia",
]


REGIONAL_THEMES: Dict[str, List[str]] = {
    "immigration_heritage": [
        "German brewing legacy in Soulard, The Hill, and Dutchtown",
        "Bosnian refugee communities revitalising Bevo Mill",
        "Vietnamese and Latinx small businesses along South Grand",
        "West African entrepreneurs in University City and North County",
        "Historic Black neighborhoods shaped by the Great Migration",
    ],
    "economic_clusters": [
        "Aerospace corridor from Hazelwood to St. Charles County",
        "Cortex innovation district and biotech startups in Midtown",
        "River logistics hubs in Granite City, Madison, and the Riverfront",
        "Health systems anchored in Barnes-Jewish, Mercy, SSM, and BJC",
        "Arts and music scenes in Grand Center, Cherokee Street, and The Loop",
    ],
    "health_disparities_focus": [
        "Asthma and environmental justice in North City and Sauget",
        "Maternal health gaps in Jennings, Berkeley, and East St. Louis",
        "Behavioral health deserts in Jefferson County's rural corridors",
        "Transportation barriers for seniors in Lemay, Mehlville, and Pacific",
        "Opioid recovery networks from Franklin County into South St. Louis",
    ],
}


REGIONAL_CONTEXT_DATA: Dict[str, List[str] | Dict[str, List[str]]] = {
    "stl_city_neighborhoods": STL_CITY_NEIGHBORHOODS,
    "stl_county_municipalities": STL_COUNTY_MUNICIPALITIES,
    "st_charles_municipalities": ST_CHARLES_MUNICIPALITIES,
    "jefferson_municipalities": JEFFERSON_MUNICIPALITIES,
    "metro_east_communities": METRO_EAST_COMMUNITIES,
    "regional_themes": REGIONAL_THEMES,
}


def _build_lookup() -> Dict[str, str]:
    """Create a lowercase -> canonical mapping for municipality matching."""

    lookup: Dict[str, str] = {}
    for collection in [
        STL_CITY_NEIGHBORHOODS,
        STL_COUNTY_MUNICIPALITIES,
        ST_CHARLES_MUNICIPALITIES,
        JEFFERSON_MUNICIPALITIES,
        METRO_EAST_COMMUNITIES,
    ]:
        for name in collection:
            lookup[name.lower()] = name
    return lookup


MUNICIPALITY_LOOKUP: Dict[str, str] = _build_lookup()


def match_municipalities(text: str) -> Dict[str, int]:
    """Return frequency counts for municipalities mentioned in text."""

    counter: Dict[str, int] = defaultdict(int)
    lowered = text.lower()
    for key, canonical in MUNICIPALITY_LOOKUP.items():
        if key in lowered:
            counter[canonical] += 1
    return dict(counter)


__all__ = [
    "REGIONAL_CONTEXT_DATA",
    "MUNICIPALITY_LOOKUP",
    "match_municipalities",
]

