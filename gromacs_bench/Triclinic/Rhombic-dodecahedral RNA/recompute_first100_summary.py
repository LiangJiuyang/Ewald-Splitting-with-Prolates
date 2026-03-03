#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: recompute_first100_summary.py <out_dir>", file=sys.stderr)
        return 2

    out = Path(sys.argv[1]).resolve()
    cases = out / "cases"
    if not cases.is_dir():
        print(f"Error: missing cases dir: {cases}", file=sys.stderr)
        return 1

    rows = []
    missing = []
    num_pme = 0.0
    den_pme = 0.0
    num_esp = 0.0
    den_esp = 0.0

    for i in range(1, 101):
        frame = f"frame_{i:03d}"
        ref_l = cases / frame / "PME_REF" / "force_longrange_pme.txt"
        ref_s = cases / frame / "PME_REF" / "force_shortrange_pme.txt"
        pme_l = cases / frame / "PME" / "force_longrange_pme.txt"
        pme_s = cases / frame / "PME" / "force_shortrange_pme.txt"
        esp_l = cases / frame / "ESP" / "force_longrange_esp.txt"
        esp_s = cases / frame / "ESP" / "force_shortrange_esp.txt"

        files = [ref_l, ref_s, pme_l, pme_s, esp_l, esp_s]
        if not all(p.exists() for p in files):
            missing.append(frame)
            continue

        ref = (np.loadtxt(ref_l) + np.loadtxt(ref_s)).reshape(-1, 3)
        pme = (np.loadtxt(pme_l) + np.loadtxt(pme_s)).reshape(-1, 3)
        esp = (np.loadtxt(esp_l) + np.loadtxt(esp_s)).reshape(-1, 3)

        if ref.shape != pme.shape or ref.shape != esp.shape:
            raise ValueError(
                f"shape mismatch in {frame}: ref={ref.shape}, pme={pme.shape}, esp={esp.shape}"
            )

        d_pme = pme - ref
        d_esp = esp - ref
        n_pme = float(np.sum(d_pme * d_pme))
        n_esp = float(np.sum(d_esp * d_esp))
        d_ref = float(np.sum(ref * ref))

        delta_pme = math.sqrt(n_pme / d_ref) if d_ref > 0 else float("nan")
        delta_esp = math.sqrt(n_esp / d_ref) if d_ref > 0 else float("nan")

        num_pme += n_pme
        den_pme += d_ref
        num_esp += n_esp
        den_esp += d_ref
        rows.append((frame, ref.shape[0], delta_pme, delta_esp))

    summary = out / "force_error_summary.tsv"
    with summary.open("w", encoding="utf-8") as f:
        f.write("case\tN_atoms\tdelta_PME_vs_PME_REF\tdelta_ESP_vs_PME_REF\n")
        for frame, n_atoms, dp, de in rows:
            f.write(f"{frame}\t{n_atoms}\t{dp:.16e}\t{de:.16e}\n")

    global_pme = math.sqrt(num_pme / den_pme) if den_pme > 0 else float("nan")
    global_esp = math.sqrt(num_esp / den_esp) if den_esp > 0 else float("nan")

    global_txt = out / "force_error_global.txt"
    with global_txt.open("w", encoding="utf-8") as f:
        f.write(f"valid_cases\t{len(rows)}\n")
        f.write(f"global_delta_PME_vs_PME_REF\t{global_pme:.16e}\n")
        f.write(f"global_delta_ESP_vs_PME_REF\t{global_esp:.16e}\n")
        f.write("missing_cases\t" + (",".join(missing) if missing else "none") + "\n")

    print(f"wrote: {summary}")
    print(f"wrote: {global_txt}")
    print(f"valid_cases={len(rows)}")
    print("missing=" + (",".join(missing) if missing else "none"))
    print(f"global_delta_PME_vs_PME_REF={global_pme:.16e}")
    print(f"global_delta_ESP_vs_PME_REF={global_esp:.16e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

