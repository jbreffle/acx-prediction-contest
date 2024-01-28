"""Main test file for the streamlit app

Test can be run locally in the terminal with "pytest -v"

See <https://github.com/streamlit/llm-examples> for an exmaple of how to run the tests
with GitHub Actions.

"""

# Imports
import os

import pyprojroot
from streamlit.testing.v1 import AppTest

streamlit_pages_dir = pyprojroot.here() / "streamlit/pages"


def test_home():
    """ "Test that Home.py runs without error"""
    # Run the app
    at = AppTest.from_file("Home.py", default_timeout=30).run()

    # Check that it runs without error
    assert not at.exception

    # Check that "blind_mode_df" is in st.session_state
    assert "blind_mode_df" in at.session_state

    return


def test_all_pages():
    """Loop over all streamlit files in streamlit/pages/ and check for runtime errors"""

    # Loop over all files in streamlit/pages
    for _, file in enumerate(os.listdir(streamlit_pages_dir)):
        # Skip __init__.py
        if file == "__init__.py":
            continue

        # Run the app
        at = AppTest.from_file(
            f"{streamlit_pages_dir}/{file}", default_timeout=30
        ).run()

        # Check that it runs without error
        assert not at.exception

        # Check that "blind_mode_df" is in st.session_state
        # Note: commented out since not all pages are completed yet
        # assert "blind_mode_df" in at.session_state

    return
