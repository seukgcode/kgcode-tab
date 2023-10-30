from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.process import dbhelper

# dbhelper.db["correction_failure"].delete_where("true")
print(dbhelper.db["correction_failure"].count)