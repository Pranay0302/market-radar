"""auth: the login gate accepts the configured pair and rejects everything else."""

from marketradar.auth import DEFAULT_PASS, DEFAULT_USER, check_credentials


def test_correct_credentials_pass():
    assert check_credentials(DEFAULT_USER, DEFAULT_PASS)


def test_wrong_password_fails():
    assert not check_credentials(DEFAULT_USER, "wrong")


def test_wrong_username_fails():
    assert not check_credentials("someone", DEFAULT_PASS)


def test_empty_credentials_fail():
    assert not check_credentials("", "")


def test_override_expected_credentials():
    assert check_credentials("u", "p", expected_user="u", expected_pass="p")
    assert not check_credentials(DEFAULT_USER, DEFAULT_PASS,
                                 expected_user="u", expected_pass="p")