from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.process import dbhelper

dbhelper.merge_from(".cache/tab.sqlite")