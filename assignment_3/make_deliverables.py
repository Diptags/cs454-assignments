"""Run every attack and save the files the assignment asks for into submission/.

    python make_deliverables.py                      # everything (~1 hour)
    python make_deliverables.py --norms linf         # only L-inf, all 5 images
    python make_deliverables.py --images knot,orca   # only some images
    python make_deliverables.py --no-perlin          # skip the bonus

For each image and norm it writes
    submission/<image>_<norm>.png       visualisation from get_viz_for
    submission/raw_<image>_<norm>.png   raw tensor (graded automatically)
and for the bonus
    submission/<image>_perlin.png, submission/raw_<image>_perlin.png
    submission/<image>_perlin.json      evolved genotype [[octave, magnitude], ...]
Results (epsilon, time, new label) are collected in submission/results.csv;
re-running for a subset updates only those rows.
"""
import argparse
import csv
import json
import logging
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.image").setLevel(logging.ERROR)  # silence imshow clipping notes

from util import *
import epsilon_ball as EB
import perlin_fog as PF

OUT = "submission"
IMAGES = ["knot", "mink", "orca", "sock", "tank"]
NORMS = {
    "linf": (EB.make_Linf_adversarial_from, get_Linf_norm_for),
    "l2": (EB.make_L2_adversarial_from, get_L2_norm_for),
    "l0": (EB.make_L0_adversarial_from, get_L0_norm_for),
}
FIELDS = ["image", "norm", "epsilon", "seconds", "orig_label", "new_label", "new_label_name", "new_prob", "fooled"]


def save_viz(adv, name):
    """get_viz_for writes <name>.png and raw_<name>.png into the current folder."""
    get_viz_for(adv.clone(), save_to=f"{name}.png")
    plt.close("all")
    for f in (f"{name}.png", f"raw_{name}.png"):
        shutil.move(f, os.path.join(OUT, f))


def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(r["image"], r["norm"]): r for r in csv.DictReader(f)}


def save_results(path, rows):
    order = {n: i for i, n in enumerate(list(NORMS) + ["perlin"])}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(rows, key=lambda k: (IMAGES.index(k[0]) if k[0] in IMAGES else 99, order.get(k[1], 99))):
            w.writerow(rows[key])


def record(rows, image, norm, eps, seconds, adv, path):
    orig = IMAGES_ORG_PREDS[path]
    prob, idx = get_model_most_likely_for(adv)
    rows[(image, norm)] = {
        "image": image, "norm": norm, "epsilon": eps, "seconds": f"{seconds:.1f}",
        "orig_label": orig, "new_label": idx, "new_label_name": get_name_for(idx),
        "new_prob": f"{prob:.4f}", "fooled": idx != orig,
    }
    status = "OK" if idx != orig else "NOT FOOLED!"
    print(f"[{status}] {image:5s} {norm:6s} eps={eps:<12.6g} {seconds:6.1f}s -> {get_name_for(idx)} ({prob:.3f})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default=",".join(IMAGES))
    parser.add_argument("--norms", default=",".join(NORMS))
    parser.add_argument("--no-perlin", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "results.csv")
    rows = load_results(csv_path)
    images = [i for i in args.images.split(",") if i]
    norms = [n for n in args.norms.split(",") if n]

    for image in images:
        path = f"images/{image}.jpg"
        org = get_image_tensor_from(path)
        for norm in norms:
            attack, measure = NORMS[norm]
            t = time.time()
            adv = attack(path)
            seconds = time.time() - t
            record(rows, image, norm, measure(adv - org), seconds, adv, path)
            save_viz(adv, f"{image}_{norm}")
            save_results(csv_path, rows)

        if not args.no_perlin:
            t = time.time()
            genes = PF.adv_func(path)
            seconds = time.time() - t
            adv = PF.add_noise_to_image(PF.genotype2phenotype(genes), org)
            record(rows, image, "perlin", sum(m for _, m in genes), seconds, adv, path)
            save_viz(adv, f"{image}_perlin")
            with open(os.path.join(OUT, f"{image}_perlin.json"), "w") as f:
                json.dump([[int(o), float(m)] for o, m in genes], f)
            save_results(csv_path, rows)

    print(f"\nDone. Files are in {OUT}/, summary in {csv_path}")


if __name__ == "__main__":
    main()
