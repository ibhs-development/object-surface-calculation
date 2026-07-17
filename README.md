# Leaf area measurement

This program measures the projected (one-sided) area of one or more flat leaves
in square centimetres. It detects a known-size ArUco marker, uses all four marker
corners to correct scale and perspective, segments each leaf from a white
background, and saves annotated and mask images.

## Setup


```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

The defaults match a 5 cm marker. For a single image, the output directory gets
an annotated image, a combined leaf mask, and a CSV with one row per leaf:

```bash
python leaf_area.py input/IMG_4659.JPG --output-dir output
```

This creates `output/IMG_4659_measured.jpg`, `output/IMG_4659_mask.png`, and
`output/IMG_4659_leaf_areas.csv`. The terminal output lists every detected leaf
and their total area. Use `--output`, `--mask-output`, or `--csv-output` to
choose individual paths. Add `--json` for machine-readable terminal output.

Leaves are numbered from top to bottom, then left to right. The annotated image
uses the same numbers as the CSV and JSON output. The CSV contains
`leaf_number`, `leaf_area_cm2`, `leaf_count`, and `total_area_cm2`; the count and
total repeat on each leaf row to make grouped analysis straightforward.

### Process a folder

Pass a folder instead of one image to process every supported image inside it:

```bash
python leaf_area.py input --output-dir output
```

While the command runs, it prints progress such as:

```text
Found 3 image(s) in input
[1/3] Processing leaf_01.jpg ...
    Done: 2 leaves, total 31.688 cm^2
```

For each source image, batch mode creates
`NAME_measured.jpg` and `NAME_mask.png`. It also creates
`output/leaf_areas.csv` with one row per detected leaf. A bad image gets one
error row and does not stop the rest of the batch.

Use `--csv-output results.csv` to choose another CSV path, or `--recursive` to
include images in subfolders. Run `python leaf_area.py --help` for all marker
and segmentation options.

## Important capture details

- `--marker-size-cm` is the side length of the **black ArUco square**, not the
  surrounding white margin. For the generator in the question, it is 5.0 cm.
  Verify the black square with a ruler after printing.
- Keep the entire marker and leaf visible, sharp, on the same flat plane, and
  avoid strong shadows or glare. A camera roughly perpendicular to the paper
  gives the best accuracy even though perspective tilt is corrected.
- The result is the leaf's 2D projected area. Curled or folded leaves cannot be
  measured accurately from one photograph.

The default dark-value threshold is deliberately conservative so neutral gray
shadows are not joined to a leaf. If a shadow is still included, lower
`--max-dark-value` below `105`. For a pale leaf that is missed, first lower
`--min-saturation` (for example, `20`); only raise `--max-dark-value` when the
leaf has genuinely gray or nearly black edge tissue. Inspect the saved mask
whenever you change these values: the white region should contain the leaf only.

Small segmented specks below `0.25 cm²` are ignored by default. If you are
measuring very small leaves, lower `--min-leaf-area-cm2`. Leaves must not touch
one another or the image edge: touching leaves form one contour, while
edge-connected regions are rejected as incomplete objects or background.
