"""
conftest.py – Make the project root importable from the tests directory.
"""
import sys
import os

# Add the ragkit-simple directory to sys.path so that imports like
# "from pdf_processor import ..." work when pytest is run from this directory.
sys.path.insert(0, os.path.dirname(__file__) + "/..")
