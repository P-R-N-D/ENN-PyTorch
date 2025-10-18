from stnet.pipeline import distributed


def test_ipv6_zone_id_is_stripped_and_bracketed():
    host = distributed._format_flight_host("fe80::1%eth0", fallback="backup")
    assert host == "[fe80::1]"


def test_bracketed_ipv6_zone_id_is_sanitized():
    host = distributed._format_flight_host("[2001:db8::1%en0]", fallback="backup")
    assert host == "[2001:db8::1]"


def test_invalid_zone_identifier_falls_back():
    host = distributed._format_flight_host("host%en0", fallback="backup")
    assert host == "backup"


def test_hostname_without_ip_passthrough():
    host = distributed._format_flight_host("example.com", fallback="ignored")
    assert host == "example.com"
