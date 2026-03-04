import os
from turbo.scripts.find_consecutive_frames import do_report

from pathlib import Path

root = Path("/home/mayo/Downloads/LMA3_Sim_Files_20250611/summary/fixed")

paths = sorted(root.rglob("*.bin"))
paths.sort(key=lambda path: os.stat(path).st_size)
output_folder = Path("/home/mayo/Downloads/LMA3_Sim_Files_20250611/summary/fixed/output")
output_folder.mkdir(parents=True, exist_ok=True)

for index, path in enumerate(paths, start=1):
    size = os.stat(path).st_size

    is_t6 = "t6" in path.name
    is_t3 = "t3" in path.name
    assert (is_t3 + is_t6) == 1, "BAD"

    denominator = 3 if is_t3 else 6
    word_size = 1115 * 8
    if "62p5" in str(path):
        word_size = 223 * 8
    print(
        f"***** #{index} {path.relative_to(root)} {size / (1024 * 1024):0.3f} MB *****"
    )
    print(f"  *** {denominator=}  {word_size=} ***")
    do_report(
        input_path=path,
        output_path=output_folder / (path.stem + ".report.json"),
        word_size=word_size,
        pseudorandomization="legacy",
        denominator=denominator,
    )
