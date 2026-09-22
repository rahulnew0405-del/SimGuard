"""
SIMULATED IP-to-location lookup + Haversine distance, for the geo-velocity
("impossible travel") signal in risk_engine.py.

*** THIS IS A SIMULATION. There is NO GeoIP service, database or network call
*** here. IP_PREFIX_TO_CITY is a tiny hand-written table so the feature can be
*** demoed and tested offline. The prefixes are the RFC 5737 documentation
*** ranges (reserved for examples, never routable), so a real client IP can't
*** accidentally match a city. A production system would replace lookup_city()
*** with a real GeoIP provider; the rest of the pipeline wouldn't change.

The Haversine great-circle distance is the standard approach used in
location-based SIM-swap / account-takeover detection (per the team's
literature review reference R8, Cheruyot et al.); this module doesn't
depend on that paper's content beyond using the formula.
"""
import math
from typing import Optional, Tuple

EARTH_RADIUS_KM = 6371.0088

# prefix -> (city, latitude, longitude). Demo data only.
IP_PREFIX_TO_CITY = {
    "203.0.113.": ("Mumbai", 19.0760, 72.8777),
    "198.51.100.": ("Delhi", 28.6139, 77.2090),
    "192.0.2.": ("London", 51.5074, -0.1278),
}


def lookup_city(ip: Optional[str]) -> Optional[Tuple[str, float, float]]:
    """Return (city, lat, lon) for a simulated IP prefix, else None (unknown)."""
    if not ip:
        return None
    for prefix, location in IP_PREFIX_TO_CITY.items():
        if ip.startswith(prefix):
            return location
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
