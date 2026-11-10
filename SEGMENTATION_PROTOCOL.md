# Current Vesicle-Pool Segmentation Protocol

Updated: 2026-11-10

This document records the segmentation procedure implemented by Vesicle
Analysis. It covers automatic candidate detection, quality filtering, split
review, identifier rules, and the user actions that determine the final
accepted mask used by downstream analysis.

## Scope and inputs

- The input is one single-channel ND2 time series of two-dimensional images.
- Segmentation is performed on the recording's effective peak frame: the
  automatically detected strict recording peak unless the user enabled a
  manual peak override.
- If motion correction was selected, only the explicitly accepted corrected
  frame is used. Otherwise the original frame is used.
- Each analysis rectangle is segmented independently. Candidate masks and
  coordinates are then placed into a full-image label mask.
- Display contrast, zoom, LUT, and overlay visibility never change the pixel
  values used for segmentation.

## Default parameters

| Parameter | Default | Role |
| --- | ---: | --- |
| Smoothing Gaussian sigma | 1.2 px | Suppress pixel-scale noise |
| Background Gaussian sigma | 8.0 px | Estimate slow spatial background |
| Foreground threshold | 1.0 local-noise units | Define the watershed domain |
| Seed threshold | 6.0 local-noise units | Require a bright regional maximum |
| Peak prominence | 1.0 local-noise units | Reject shallow maxima without imposing a seed distance |
| Absolute area range | 8--150 px² | Flag candidates outside the permitted range |
| Population baseline | Median / scaled MAD | Establish per-rectangle metric baselines |
| Outlier multiplier | 3.0 spreads | Flag population outliers for each enabled metric |
| Minimum population | 5 finite candidates | Disable population filtering for sparse rectangles |

The seed threshold is in the **Basic** tab. Peak prominence, baseline mode,
absolute area range, per-metric filters, outlier multipliers, and optional
absolute metric cutoffs are in **Advanced filtering**. Settings are stored
independently for every loaded recording.

## Automatic candidate detection

For each analysis rectangle:

1. Convert the peak-frame crop to `float32`. Retain the measured intensities;
   display transformations are not applied.
2. Calculate a smoothed image with Gaussian sigma 1.2 px and a slow background
   with Gaussian sigma 8 px. The enhanced response is:

   `enhanced = smoothed - slow_background`

3. Let `m` be the median enhanced response. Estimate local noise as:

   `noise = 1.4826 * median(abs(enhanced - m))`

   If this value is zero or non-finite, use the enhanced-image standard
   deviation. A crop with no finite positive noise estimate produces no
   candidates.
4. Define foreground pixels by:

   `enhanced > m + foreground_threshold * noise`

5. Find thresholded regional maxima within that foreground. A seed must be
   above `m + seed_threshold * noise` and have the configured prominence above
   its surrounding saddle. There is no fixed minimum distance between seeds;
   close maxima are retained when both have sufficient prominence.
6. Run marker-controlled watershed on `-enhanced`, restricted to the foreground
   mask.
7. Inspect each connected foreground component. When it contains multiple
   watershed children, keep the automatic split only if every child has at
   least the configured absolute minimum area. Otherwise retain the connected
   component as one candidate. Automatic area limits do not otherwise delete
   candidate masks.

When explicitly approved analysis rectangles overlap, the first accepted
candidate owns its pixels. A later overlapping candidate is skipped if less
than half of its original area remains available; otherwise only its available
pixels are retained.

## Candidate measurements

Every candidate is measured before quality acceptance:

- **Area:** labelled pixel count.
- **Circularity:** `min(1, 4*pi*area / Crofton_perimeter²)`.
- **Solidity:** area divided by convex-hull area, as calculated by
  `skimage.measure.regionprops`.
- **Local contrast:** the smoothed candidate mean minus the smoothed median in
  a 2--6 px annulus, divided by local noise. Pixels belonging to any candidate
  are removed from the annulus. The value is unavailable when no annulus pixels
  remain.
- **Intensity variation:** scaled MAD of the enhanced response inside the
  candidate divided by `max(abs(candidate median response), 0.1 * noise)`.

The recorded candidate data also include its analysis rectangle, centroid,
bounding box, peak response, split-group provenance, and stable integer ID.

## Automatic quality decision

Quality is evaluated separately inside each analysis rectangle.

1. Enabled absolute thresholds are applied first. By default, area outside
   8--150 px² is flagged. Optional absolute minimum circularity, minimum
   solidity, minimum local contrast, and maximum intensity variation are off by
   default.
2. Population filtering runs only when the rectangle contains at least five
   candidates, and a metric is evaluated only when it has at least five finite
   values.
3. The default robust baseline uses the metric median as its center and
   `1.4826 * MAD` as its spread. Advanced settings can instead use mean and
   sample standard deviation, or disable population filtering with **Manual
   thresholds only**.
4. Area is checked in both tails. Circularity, solidity, and local contrast are
   checked in the low tail. Intensity variation is checked in the high tail.
   The default boundary is three spreads from the center for every metric.
5. A candidate with any enabled quality flag is initially invalid. Automatic
   split provenance is not a quality failure, so otherwise valid split children
   remain accepted.

The software preserves two full-image masks: all detected candidates and the
currently accepted result. Filtering never destroys the candidate geometry.

## Stable pool and split identifiers

- IDs are assigned before quality filtering and remain available during review.
- For an automatic split, the largest child keeps the original parent ID and
  the other children receive new IDs.
- Pool-ID labels are placed at object centroids. Fewer than ten colliding labels
  are displaced with leader lines; larger collision groups are hidden until
  zoom makes them readable.
- After a reviewed merge, the connected component nearest the original group
  centroid keeps the parent ID. Any other disconnected valid component keeps a
  suitable child ID rather than being represented as a disconnected single
  pool.

## Review protocol

The two review decisions are deliberately independent.

### Valid/invalid review

- Enable **Review valid/invalid results**.
- The review overlay contains every active automatic candidate. Green means
  valid. Invalid objects use a reason-specific color; white edges indicate
  multiple quality reasons.
- Clicking a candidate toggles only its validity. In particular, making an
  invalid split child valid does not merge its split group.

### Automatic-split review

- Enable **Review automatic splits**.
- Magenta means the proposal is currently split; cyan means it is currently
  merged. White boundaries preserve the original proposed children in both
  states.
- Clicking any member toggles the entire proposal between split and merged.
- A merged result contains only children currently marked valid. Invalid
  children contribute no pixels.
- If all children are invalid, the group contributes no accepted object until
  at least one child is made valid.
- If excluded children disconnect the valid union, each connected component is
  kept as a separate accepted pool.
- Restoring the split restores every child's previous validity and child ID.

Only one review mode accepts clicks at a time. Starting either mode temporarily
turns off **Show accepted segmentation results**, turns on the relevant review
overlay, and preserves the user's previous overlay settings. Ending review
restores those settings exactly.

Validity and split changes are stored in the same per-recording undo/redo
history as mask corrections. A manual correction retires only the automatic
candidates or split proposals that it replaces, preventing a later review
click from overwriting a hand-drawn result.

## Final accepted result and invalidation

Only the current accepted label mask is used for pool-area analysis, background
selection, trace extraction, and scientific export. Candidate and review
overlays are diagnostic and do not add pixels to downstream analysis by being
visible.

Rerunning automatic segmentation recalculates candidates, quality flags, split
groups, accepted labels, and review decisions. Changing the effective peak,
analysis rectangles, motion correction, or segmentation parameters invalidates
dependent segmentation and downstream cached results.

## Validation policy

Synthetic known-answer tests cover close maxima, split provenance, reversible
quality filtering, excluded split children, disconnected merged components,
stable IDs, overlay state, and undo/redo. Supplied microscopy recordings are
used for regression and manual visual review. Example ImageJ annotations are
not an exhaustive ground truth and must not be used to label every unmatched
detection as a false positive.
