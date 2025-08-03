import os
import sys

"""use absolute path to import src directory to src/test/config_test.py"""
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.configuration import Configuration

config = Configuration()
print(config)
