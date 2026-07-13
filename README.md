# Leaf area measurement

This program measures the projected (one-sided) area of a flat leaf in square
centimetres. It detects a known-size ArUco marker, uses all four marker corners
to correct scale and perspective, segments the leaf from a white background,
and saves an annotated result.

## Setup


```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

The defaults match a 5 cm marker:

```bash
python leaf_area.py input/IMG_4659.JPG \
  --output output/IMG_4659_measured.jpg \
  --mask-output output/IMG_4659_mask.png
```

The command prints the leaf area in cm² and creates
`output/IMG_4659_measured.csv` with the image name and area. Use
`--csv-output results.csv` to choose another CSV path. Add `--json` for
structured terminal output.

### Process a folder

Pass a folder instead of one image to process every supported image inside it:

```bash
python leaf_area.py input --output-dir output
```

While the command runs, it prints progress such as:

```text
Found 3 image(s) in input
[1/3] Processing leaf_01.jpg ...
    Done: 17.051 cm^2
```

For each source image, batch mode creates
`NAME_measured.jpg` and `NAME_mask.png`. It also creates
`output/leaf_areas.csv` with the image name, area, status, and any error. A bad
image is reported and recorded without stopping the rest of the batch.

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

For a pale leaf that is missed, lower `--min-saturation` (for example, `20`) or
raise `--max-dark-value` (for example, `210`). Inspect the saved mask whenever
you change these values: the white region should contain the leaf only.
