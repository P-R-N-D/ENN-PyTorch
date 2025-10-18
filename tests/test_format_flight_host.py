import pytest

from stnet.pipeline import distributed


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("fe80::1%eth0", "[fe80::1]"),
        ("[2001:db8::1%en0]", "[2001:db8::1]"),
    ],
)
def test_ipv6_zone_id_is_stripped(candidate: str, expected: str) -> None:
    host = distributed._format_flight_host(candidate, fallback="backup")
    assert host == expected


@pytest.mark.parametrize(
    "candidate",
    ["host%en0", "[invalid]", "2001:db8::zz"],
)
def test_invalid_candidates_fall_back(candidate: str) -> None:
    host = distributed._format_flight_host(candidate, fallback="backup")
    assert host == "backup"


def test_hostname_without_ip_passthrough() -> None:
    host = distributed._format_flight_host("example.com", fallback="ignored")
    assert host == "example.com"
