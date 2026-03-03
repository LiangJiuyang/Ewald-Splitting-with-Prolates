#!/usr/bin/env python3

import math
import os
import re
from pathlib import Path

import numpy as np


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    out = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p)
    return out


def load_total(long_path: Path, short_path: Path) -> np.ndarray:
    long_force = np.loadtxt(long_path)
    short_force = np.loadtxt(short_path)
    if long_force.shape != short_force.shape:
        raise ValueError(f"shape mismatch: {long_path} vs {short_path}")
    total = long_force + short_force
    if total.ndim != 1:
        total = total.reshape(-1)
    if total.size % 3 != 0:
        raise ValueError(f"component count is not multiple of 3: {total.size}")
    return total.reshape(-1, 3)


def delta_and_num_den(approx: np.ndarray, ref: np.ndarray):
    diff = approx - ref
    num = float(np.sum(diff * diff))
    den = float(np.sum(ref * ref))
    if den == 0.0:
        return float("nan"), num, den
    return math.sqrt(num / den), num, den


def main():
    root = Path(__file__).resolve().parent
    cases_root = root / "cases"
    out_table = root / "force_error_summary.tsv"
    out_global = root / "force_error_global.txt"
    esp_ref_mode = os.environ.get("ESP_REF_MODE", "PME_REF").upper()
    if esp_ref_mode == "ESP_REF":
        esp_ref_label = "ESP_REF"
    elif esp_ref_mode == "PME_REF":
        esp_ref_label = "PME_REF"
    else:
        raise ValueError(f"unsupported ESP_REF_MODE: {esp_ref_mode}")
    esp_long = os.environ.get("ESP_LONG_FORCE_FILE", "force_longrange_esp.txt")
    esp_short = os.environ.get("ESP_SHORT_FORCE_FILE", "force_shortrange_esp.txt")
    esp_ref_long = os.environ.get("ESP_REF_LONG_FORCE_FILE", esp_long)
    esp_ref_short = os.environ.get("ESP_REF_SHORT_FORCE_FILE", esp_short)
    pme_long = os.environ.get("PME_LONG_FORCE_FILE", "force_longrange_pme.txt")
    pme_short = os.environ.get("PME_SHORT_FORCE_FILE", "force_shortrange_pme.txt")

    case_dirs = sorted([d for d in cases_root.iterdir() if d.is_dir()], key=natural_key) if cases_root.is_dir() else []
    rows = []
    skipped = []

    esp_num_sum = 0.0
    esp_den_sum = 0.0
    pme_num_sum = 0.0
    pme_den_sum = 0.0

    for case in case_dirs:
        try:
            esp = load_total(
                case / "ESP" / esp_long,
                case / "ESP" / esp_short,
            )
            pme = load_total(
                case / "PME" / pme_long,
                case / "PME" / pme_short,
            )
            pme_ref = load_total(
                case / "PME_REF" / pme_long,
                case / "PME_REF" / pme_short,
            )
            if esp_ref_mode == "ESP_REF":
                esp_ref = load_total(
                    case / "ESP_REF" / esp_ref_long,
                    case / "ESP_REF" / esp_ref_short,
                )
            else:
                esp_ref = pme_ref

            if not (esp.shape == esp_ref.shape == pme.shape == pme_ref.shape):
                raise ValueError(
                    f"shape mismatch ESP={esp.shape} ESP_REF_TARGET={esp_ref.shape} PME={pme.shape} PME_REF={pme_ref.shape}"
                )

            d_esp, n_esp, de_esp = delta_and_num_den(esp, esp_ref)
            d_pme, n_pme, de_pme = delta_and_num_den(pme, pme_ref)

            esp_num_sum += n_esp
            esp_den_sum += de_esp
            pme_num_sum += n_pme
            pme_den_sum += de_pme

            rows.append((case.name, esp.shape[0], d_esp, d_pme))
        except Exception as exc:  # noqa: BLE001
            skipped.append((case.name, str(exc)))

    with out_table.open("w", encoding="utf-8") as f:
        f.write(f"case\tN_atoms\tdelta_ESP_vs_{esp_ref_label}\tdelta_PME_vs_PME_REF\n")
        for name, n_atoms, d_esp, d_pme in rows:
            f.write(f"{name}\t{n_atoms}\t{d_esp:.16e}\t{d_pme:.16e}\n")

    global_esp = math.sqrt(esp_num_sum / esp_den_sum) if esp_den_sum > 0 else float("nan")
    global_pme = math.sqrt(pme_num_sum / pme_den_sum) if pme_den_sum > 0 else float("nan")

    with out_global.open("w", encoding="utf-8") as f:
        f.write(f"valid_cases\t{len(rows)}\n")
        f.write(f"skipped_cases\t{len(skipped)}\n")
        f.write(f"global_delta_ESP_vs_{esp_ref_label}\t{global_esp:.16e}\n")
        f.write(f"global_delta_PME_vs_PME_REF\t{global_pme:.16e}\n")
        if skipped:
            f.write("skipped_list\n")
            for name, reason in skipped:
                f.write(f"{name}\t{reason}\n")

    print(f"valid_cases={len(rows)} skipped_cases={len(skipped)}")
    print(f"global_delta_ESP_vs_{esp_ref_label}={global_esp:.16e}")
    print(f"global_delta_PME_vs_PME_REF={global_pme:.16e}")
    print(f"wrote: {out_table}")
    print(f"wrote: {out_global}")


if __name__ == "__main__":
    main()
