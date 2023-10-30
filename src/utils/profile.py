import os
from pathlib import Path

from . import PathLike


class Profile:
    def __init__(self, profile_path: PathLike = ".profiles", snakeviz: bool = True) -> None:
        self.profile_path = Path(profile_path)
        self.profile_path.mkdir(parents=True, exist_ok=True)
        self.snakeviz = snakeviz
        import cProfile

        self.pr = cProfile.Profile()

    def __enter__(self):
        self.pr.enable()

    def __exit__(self, exc_type, exc_value, traceback):
        import io
        import pstats
        import time

        save_path = self.profile_path / f"{int(time.time())}.prof"

        self.pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats("cumtime")
        ps.print_stats()
        ps.dump_stats(save_path)

        save_path.with_suffix(".txt").write_text(s.getvalue(), encoding="utf8")

        if self.snakeviz:
            os.system(f"snakeviz {save_path}")
