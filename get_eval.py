from pathlib import Path
import re 
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train for CLWE")
    parser.add_argument(
        "-p", 
        "--path", 
        type=str,
        default=None,
        help="path where the result text is stored"
        )

    args = parser.parse_args()

    results = Path(args.path)
    results = results / "evaluated.txt"
    print(f"SAVED TO {results}")

    with results.open(mode='r') as r:
        text = r.read()

    maps = re.findall(r"\sMAP@5:\s(\d+\.\d+)\s+", text)
    ps = re.findall(r"\sP@5:\s(\d+\.\d+)\s+", text)


    with results.open(mode="a") as f:
        f.write(f"\n ++++++++++++++++++++++++++++++++++++++++++++\n")

        f.write(f"P@\n")
        for p in ps[:10]:
            f.write(f"{p}\n")

        f.write(f"\n ============================================\n")

        f.write(f"MAP\n")
        for p in maps[:10]:
            f.write(f"{p}\n")
        f.write(f"\n ++++++++++++++++++++++++++++++++++++++++++++\n")