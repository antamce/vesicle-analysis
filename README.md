# Vesicle Analysis

Vesicle Analysis is a desktop application for analyzing single-channel Nikon
ND2 time series of synaptic-vesicle pools. It provides rectangular analysis
regions, recording-wide peak detection, automatic pool segmentation, manual
correction tools, background selection, trace extraction, and ImageJ-compatible
exports.

The application runs locally. Recordings are not uploaded anywhere.

## Requirements

- Windows or macOS
- Anaconda, Miniconda, Miniforge, or Mambaforge
- An internet connection during the first setup and later dependency updates

No existing Conda environment is modified. The launcher creates and manages a
dedicated environment for Vesicle Analysis. If its preferred environment name
is already in use, it chooses a different name.

## Installation and launch

Download or clone this repository, then keep all included files together.

### Windows

Double-click **Launch Vesicle Analysis.bat**. The launcher locates Conda,
creates the application environment when necessary, adds a desktop shortcut,
and opens the app.

You can also drag one ND2 file or a folder containing ND2 recordings onto the
launcher or its desktop shortcut.

### macOS

Double-click **Launch Vesicle Analysis.command**. On its first launch, macOS may
require you to right-click the file and choose **Open**. The launcher locates
Conda, creates the application environment when necessary, adds a desktop
shortcut, and opens the app.

If the command lost its executable permission during a manual file transfer,
open Terminal in the project folder and run:

```bash
chmod +x "Launch Vesicle Analysis.command"
```

You can drag one ND2 file or a recording folder onto the desktop shortcut to
open it directly.

### If Conda is not found

Select either the Conda executable or the top-level installation folder when
prompted. The launcher accepts Anaconda, Miniconda, Miniforge, and Mambaforge.
An invalid selection displays an explanation and reopens the chooser.

Setup progress remains visible in the terminal. If setup fails, read the error
shown there, check the internet connection, and run the launcher again.

## Analysis workflow

### Open recordings

Use **Open ND2 File** to open one recording or **Open Folder** to load all ND2
files directly inside a folder. Each recording must be a single-channel time
series of two-dimensional images from one fixed field of view; Z stacks are not
used.

Select recordings in the list on the left. Navigate frames with the slider,
frame-number control, arrow keys, or playback button. Display contrast and the
colour map can be adjusted without changing the raw data used for analysis.

### Correct XY motion

Before defining analysis regions, choose one option independently for each
recording: **No correction**, **Translation**, or **Rigid body**. Translation
uses whole pixels. Rigid correction uses nearest-neighbor rotation followed by
whole-pixel translation, so recorded intensities are never averaged. Empty
borders are filled with zero and the original ND2 is never changed.

Use **Calculate current** or apply the selected mode to all loaded recordings
with **Calculate all**. Correction uses frame 1 as the reference and may take
several minutes for a large recording. Progress is shown by recording and
frame. Review the shift/rotation traces and separate original/corrected previews,
inspect warning frames, and optionally adjust individual frames before clicking
**Accept correction**.

After acceptance, **Save corrected images…** can create a lossless
`<recording>_registered.ome.tif` together with a transform/QC CSV and metadata
JSON. The original dimensions, integer pixel type, calibration, timestamps,
and exposure metadata are retained wherever the output format supports them.

### Define analysis regions and detect peaks

1. Enable **Edit analysis regions**.
2. Drag one or more rectangles over the parts of the recording to analyze.
3. Move or resize rectangles while editing is enabled, then finish region
   editing to lock them.
4. Repeat for each loaded recording.
5. Click **Calculate all recording peaks** once.

The app calculates a separate strict raw-intensity maximum for each recording.
If an exact tie occurs, the earliest tied frame is used. Overlapping analysis
regions are blocked unless explicitly approved, and approved overlaps are
counted only once. An optional manual peak override is available.

### Segment vesicle pools

Click **Run automatic segmentation** to detect bright vesicle pools on the
effective peak frame. The **Basic** tab controls the detection threshold. The
**Advanced filtering** tab controls peak prominence, absolute area limits,
per-metric outlier filtering, optional absolute cutoffs, and the population
baseline. The default baseline is the median with scaled MAD; mean/standard
deviation and manual-threshold-only modes are also available.

The exact current detection formulas, defaults, quality rules, identifier
behavior, and review procedure are recorded in
[Current Vesicle-Pool Segmentation Protocol](SEGMENTATION_PROTOCOL.md).

The accepted segmentation, pool-ID, valid/invalid review, and automatic-split
review overlays have synchronized switches in the top toolbar and side panel.
IDs are placed at mask centroids; crowded labels are moved with leader lines,
and very dense groups remain hidden until the view is zoomed in.

Quality filtering is calculated independently in each analysis rectangle from
area, circularity, solidity, local contrast, and within-object intensity
variation. Population filtering is skipped when a rectangle contains fewer
than five candidates. Every detected candidate retains its stable ID and mask
for optional review. **Review valid/invalid results** shows valid candidates in
green and invalid candidates in a reason-specific colour; clicking toggles only
that candidate's validity. **Review automatic splits** independently toggles a
whole proposed group between split and merged. Invalid children are excluded
from a merged result, and disconnected valid regions remain separate pools.

Only one review mode can receive clicks at a time. Starting either review turns
off **Show accepted segmentation results** and turns on the relevant review
overlay; ending review restores all previous overlay settings. Both kinds of
decisions share the normal undo/redo history. Rerunning segmentation resets
the review decisions from the newly calculated candidates.

Automatic results can be corrected with the following tools:

- Delete or merge selected pools.
- Draw a freehand, oval, or circular replacement boundary.
- Separate a selected pool with a line.
- Draw a pool hint and rerun automatic segmentation locally.
- Undo, redo, or reset corrections.

Click a pool overlay or use the pool list to select it. Hold Ctrl to select
multiple pools. Corrected boundaries stay inside the analysis rectangles, and
overwriting another pool requires explicit confirmation.

### Calculate pool areas

Click **Calculate areas** after accepting the pool boundaries. The app exports
both the complete peak-frame mask area and the number of pixels detected as the
pool in every frame. Framewise detection is recalculated independently from
local noise and is strictly limited to the accepted peak mask; it cannot expand
beyond that boundary. Missing detections are stored as zero with a QC status.

Use **Review masks…** to inspect any frame before export. Cyan/white pixels are
the framewise detection and yellow is the fixed peak-mask boundary. Review is
optional. Area results are cached and automatically cleared if motion
correction, segmentation settings, or pool boundaries change.

### Select backgrounds

Choose between **One background per pool** and **One common background for all
pools**, then click **Find backgrounds**.

- Pool-wise backgrounds preserve the shape and pixel count of their associated
  pool and are placed nearby.
- A common background is a square region shared by all pools.

The automatic search avoids pools and bright or highly variable structures.
Distances and common-background size can be adjusted in **Advanced settings**.

If a pool-wise background cannot be found, the unresolved pool appears in the
manual-background dropdown and is highlighted in red. Backgrounds can be
drawn, moved, deleted, redrawn, undone, redone, or reset.

### Export results

When every pool has a valid background, use **Review current recording…** or
**Review all / batch export…**. The review window can move between one or many
recordings and switch between raw and normalized pool traces. Uncheck an
individual pool to omit it from scientific exports and record a preset or
free-text reason. Excluded pools remain in the workbook audit summary.

For crowded recordings, the plot shows 15 pool traces per trace batch. Use the
previous/next trace-batch buttons to move through them, or click a curve to
select and center its pool in the list. The graph automatically fits all traces
in the visible batch; **Fit traces** restores that view after manual zooming.

The normalized display and export use:

`(pool - assigned background) / mean(pool - assigned background, frames 1-10)`

Pool-wise mode uses each pool's assigned background; common mode uses the
shared background. If that first-ten-frame baseline is zero or near zero, the
reviewer shows the raw trace with a clear warning and leaves the corresponding
normalized export column blank.

Review decisions can be saved to and loaded from a JSON manifest. A manifest
matches recordings by source path, or by file name and file size, and is only
applied when the accepted pool identifiers also match. Batch export processes
ready recordings sequentially and reports incomplete recordings as ignored.

Each exported recording produces:

- `<recording>_analysis.xlsx` with metadata, raw pool/background traces,
  normalized pool traces, inclusion/exclusion audit information, peak-frame
  pool areas, framewise pool areas and QC, analysis-region geometry, and the
  recording-mean peak trace.
- `<recording>_rois.zip` with ImageJ-compatible pool, background, and analysis
  rectangle ROIs.
- `<recording>_display.lut` with the 256-colour ImageJ LUT currently displayed
  in the app.
- `<recording>_manifest.json` with included pools, excluded pools and reasons,
  exported backgrounds, the normalization formula, and output file names.

An excluded pool's pool-wise background is also omitted. A common background
is retained whenever common-background mode was used. Excluded pools are
omitted from raw/normalized trace columns, area time series, and ImageJ ROIs.

ND2 timestamps are used when available. If timestamps are missing, the app asks
for seconds per frame; if no interval is supplied, the workbook uses frame
numbers. Raw trace sheets always preserve the original mean intensities.

Existing exports are replaced only after confirmation. Output files are
validated before the app reports a successful export.

## Important usage notes

- Analysis settings, corrections, trace caches, and review exclusions are
  retained separately for every loaded recording while the application remains
  open. Saved trace-review manifests restore inclusion/exclusion decisions in a
  later session; other analysis state is not yet restored.
- Moving or changing an analysis rectangle invalidates dependent peak,
  segmentation, and background results for that recording.
- Display contrast and LUT changes never modify raw measurements.

## Manual launch

To create and start the environment without the automatic launcher:

```text
conda env create -f environment.yml
conda activate vesicle-analysis
python -m vesicle_analysis
```

An optional ND2 file or directory can be passed on the command line:

```text
python -m vesicle_analysis "path/to/recording-or-folder"
```

Launcher settings and application logs are stored per user under Local AppData
on Windows or Application Support on macOS.
