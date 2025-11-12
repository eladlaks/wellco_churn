import os


def read_baseline_auc(baseline_path):
    if not os.path.exists(baseline_path):
        return None
    with open(baseline_path, "r") as f:
        for line in f:
            if "=" in line:
                parts = line.strip().split("=")
                try:
                    return float(parts[1])
                except:
                    continue
    return None
