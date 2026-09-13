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
effective peak frame. Advanced settings provide controls for detection
threshold, accepted area, and seed separation. The coloured overlay can be
shown or hidden while comparing it with the raw image.

Automatic results can be corrected with the following tools:

- Delete or merge selected pools.
- Draw a freehand, oval, or circular replacement boundary.
- Separate a selected pool with a line.
- Draw a pool hint and rerun automatic segmentation locally.
- Undo, redo, or reset corrections.

Click a pool overlay or use the pool list to select it. Hold Ctrl to select
multiple pools. Corrected boundaries stay inside the analysis rectangles, and
overwriting another pool requires explicit confirmation.

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

When every pool has a valid background, click **Export current recording…** and
choose an output folder. Each recording produces:

- `<recording>_analysis.xlsx` with metadata, raw pool traces, raw background
  traces, analysis-region geometry, and the recording-mean peak trace.
- `<recording>_rois.zip` with ImageJ-compatible pool, background, and analysis
  rectangle ROIs.
- `<recording>_display.lut` with the 256-colour ImageJ LUT currently displayed
  in the app.

ND2 timestamps are used when available. If timestamps are missing, the app asks
for seconds per frame; if no interval is supplied, the workbook uses frame
numbers. Traces contain raw mean intensities and are not background-subtracted,
normalized, or converted to delta-F/F0.

Existing exports are replaced only after confirmation. Output files are
validated before the app reports a successful export.

## Important usage notes

- Analysis settings and corrections are retained separately for every loaded
  recording while the application remains open.
- Export completed work before closing the application; sessions are not saved
  for later restoration.
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
