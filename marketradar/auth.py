"""tiny credential check for the app's login gate.

kept separate from app.py (which imports streamlit and runs on import) so the
check stays pure and unit-testable. the credentials are hardcoded "for now"; the
app can override them from st.secrets without touching this module.

todo: real user management + hashed secrets once this prototype is approved.
"""

from __future__ import annotations

import hmac

DEFAULT_USER = "aifund001"
DEFAULT_PASS = "AiFund@1357"


def check_credentials(user: str, password: str,
                      expected_user: str = DEFAULT_USER,
                      expected_pass: str = DEFAULT_PASS) -> bool:
    """constant-time compare of a username/password pair against the expected."""
    user_ok = hmac.compare_digest(str(user), expected_user)
    pass_ok = hmac.compare_digest(str(password), expected_pass)
    return user_ok and pass_ok