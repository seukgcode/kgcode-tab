import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.evaluators.report import generate_report

generate_report(".result")
