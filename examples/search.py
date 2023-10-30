from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.searchmanage import SearchManage, SpellCheck
from src.searchmanage.tools import AnalysisTools

# print(SearchManage().search_run(["[bom2000]j218-354"], keys="all"))
# print(SearchManage().search_run(["Q5"], keys="all"))

sp = SpellCheck(m_num=5)

# print(sp.search_run(["Corot 1896-1875"], timeout=1000, block_num=2, function_=AnalysisTools.bing_page))
# print(sp.search_run(["Gliee 829"], timeout=1000, block_num=2, function_=AnalysisTools.bing_page))
# print(
#     sp.search_run(["Ercole Ramazzani de lb Rocha - Aspetti del Manierismo nelle Marche della Controriforma"],
#                   timeout=1000,
#                   block_num=2,
#                   function_=AnalysisTools.bing_page)
# )
# print(SearchManage(key="ids").search_run(["Q5"], keys=["qid", "labels/en", "id", "description"]))

print(SearchManage(key="ids").search_run(["Q1798105", "Q84203926"], keys=["labels/en", "description/en", "properties"]))
