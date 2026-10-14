from recsys import hello


def test_hello() -> None:
    ret = hello()
    assert ret == "Hello from recsys!"