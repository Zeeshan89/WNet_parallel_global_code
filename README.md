# WNet storm detection

Detection of four storm types from reanalysis fields using a segmentation network,
together with the trackers that turn per timestep detections into storm tracks. The
four types are atmospheric rivers, tropical cyclones, extratropical cyclones and
mesoscale convective systems.

`METHODOLOGY.md` describes the method. This file describes the code.

Source files only. No data, model weights, results or figures are included.

## What changed in this revision

The ERA5 configuration now runs at the **native 0.25 degree grid, 721 x 1440**, instead
of the earlier 192 x 384. The network pools three times, so training drops the row at
-90 to give 720 x 1440 (`main_phase2.py --crop-rows 720`); inference instead reflect
pads 721 rows to 728 and crops back, so the exported files keep the full grid.

Three of the four training targets were rebuilt:

**Atmospheric rivers** are no longer a single global threshold on IVT. A cell qualifies
where IVT exceeds the 85th percentile *for that cell and that five month season*, with
an absolute floor of 175 kg m-1 s-1 tapering to 100 toward the poles, and the connected
object must then be at least 2000 km long, at least twice as long as it is wide, carry
moisture poleward with a coherent direction, and reach beyond 23.5 degrees latitude.
Objects that fail are thresholded again inside their own footprint at the 87.5th, 90th,
92.5th and 95th percentiles. This follows the Guan and Waliser design and fixes the
earlier behaviour where an AR vanished at the coastline.

**Tropical cyclones** using different reanalysis variables, like MSLP, vorticity, and windspeed. Validated against IBTrACS.

**Extratropical cyclones** require a closed pressure contour around the surface low, the
TempestExtremes criterion, instead of a local minimum inside a box.

**Mesoscale convective systems** are unchanged.

## How it works, end to end

The model reads six physical fields for each six hourly timestep: integrated vapour
transport, sea level pressure, relative vorticity at 850 hPa, wind speed at 850 hPa,
cloud top temperature and total column water vapour. It returns four probability maps,
one per storm type. A cell becomes a detection when its probability exceeds one half.

The target masks used for training are built from physical criteria and are kept
strictly separate from the predictors. The network never receives them as input, so it
cannot reach a low loss by copying an answer that was handed to it.

Detections are written as monthly files on the full grid. The trackers then read those
files, link detections across time into storm paths, and apply the physical checks such
as minimum lifetime and genesis latitude.

## Running the pipeline

Steps run in order; each writes what the next one reads.

| Step | Script | What it does |
|---|---|---|
| 1 | `precompute_tensors.py --split` | Builds the monthly cache from the raw NetCDF inputs. |
| 2 | `phase4_ivt_percentiles.py` | Per cell, per season IVT percentiles, the intensity term of the AR target. |
| 3 | `phase4_build_ar_label.py` | Builds the atmospheric river target from those percentiles plus the geometry and transport tests. |

| 4 | `phase4_build_etc_label.py` | Builds the closed contour extratropical cyclone target. |
| 5 | `build_fullres_cache.py` | Assembles the 0.25 degree cache: six predictors and the four targets, one file per month, checked against the earlier cache. |
| 6 | `compute_predictor_stats.py` | Derives the mean and standard deviation used to scale each predictor field. |
| 7 | `main_phase2.py` | Trains the network. |
| 8 | `phase4_export_detections.py` | Runs the trained model over the record and writes monthly detection files, one per storm type. |
| 9 | `export_hourly_fullres.py` | Hourly MCS detections, predictors assembled on the fly. |
| 10 | `phase4_run_trackers.py` | Runs the trackers on those files to produce storm tracks. |
| 11 | `phase4_add_mcs_ids.py` | Adds tracked MCS identifiers to the MCS detection files. |

`scripts/` holds one SLURM launcher per step as a worked example. They are written for
OLCF Frontier and carry site specific paths, module loads and account names, so treat
them as templates rather than portable scripts. Several Python files also contain
absolute paths to the machine the work was done on; these need changing for any other
system.

## The files

**Data preparation**

| File | Role |
|---|---|
| `dataset.py` | Builds the storm target masks and defines the contract that keeps predictors and targets separate. |
| `precompute_tensors.py` | Builds the monthly cache consumed by training and inference. |
| `build_fullres_cache.py` | Builds the 0.25 degree cache with the current target definitions, verifying each month against the earlier one. |
| `build_hourly_split_cache.py` | Hourly predictor cache, and the month bounds and time interpolation reused by the hourly export. |
| `cache_loader.py` | Reads a preprocessed month, with a fallback that rebuilds it from raw NetCDF. |
| `cache_to_netcdf.py` | Writes cache channels back out as NetCDF, one file per variable, in raw physical units. |
| `compute_predictor_stats.py` | Derives the scaling constants from the data. |
| `repair_split_month.py` | Rebuilds a single month whose cache is unreadable. |
| `distributed_dataset.py` | Splits the monthly files across processes. |
| `ibtracs_loader.py` | Reads the IBTrACS observed cyclone database. |
| `utils.py` | Shared helpers. |

**Target construction**

| File | Role |
|---|---|
| `ar_label_lib.py` | The atmospheric river definition: intensity, geometry, transport direction, iterative rethresholding. |
| `phase4_ivt_percentiles.py` | Per cell, per season IVT percentiles. |
| `phase4_build_ar_label.py` | Applies the definition and writes the target, native or into a cache. |
| `etc_closed_contour.py` | Cyclone centres as closed pressure lows, shared by the target and the tracker. |
| `phase4_build_etc_label.py` | Builds the extratropical cyclone target. |

| `phase4_tc_label_screen.py` | Screening rules used when building that target. |

**Model and training**

| File | Role |
|---|---|
| `WNet_model.py` | Network definition, including the multi label version. |
| `loss.py` | Loss functions: per class cross entropy, Tversky, and the clustering term. |
| `metrics.py` | Scoring, per class and aware of class imbalance. |
| `train.py` | Training loop, year based splits, validation. |
| `main_phase2.py` | Training entry point, including `--crop-rows` and per class checkpoint selection. |

**Inference and tracking**

| File | Role |
|---|---|
| `phase4_export_detections.py` | Writes the monthly detection files. Reads only the six predictor channels. |
| `export_hourly_fullres.py` | Hourly MCS detections at 0.25 degrees, predictors built per hour rather than cached. |
| `phase4_run_trackers.py` | Runs the trackers on the detection files. |
| `Hurricane_track.py` | Tropical cyclone tracker, including the track level physical checks. |
| `updated_monthly_ETC_Tracker.py` | Extratropical cyclone tracker. |
| `phase4_add_mcs_ids.py` | Tracks MCS and appends the identifiers to the detection files. |
| `post_processing.py` | Shared masking, labelling and tracking helpers. |

**Evaluation and figures**

| File | Role |
|---|---|
| `phase4_evaluate_tc.py` | Scores tropical cyclones against observations, per timestep. |
| `phase4_evaluate_tracks.py` | Scores whole tracks, which is the right level for judging track filters. |
| `plot_tracks_peryear.py` | Track plotting, and the IBTrACS reading and track matching used by the scorers. |
| `plot_phase2_history.py` | Training convergence figures. |
| `visualization.py` | Shared plotting helpers. |

## Three things to know when reading the output

**Detection files and track files are different products.** The monthly NetCDF files
written by `phase4_export_detections.py` are raw model output. A cell is marked purely
because the model's own probability exceeded the threshold; nothing is filtered, masked
or overwritten at that stage. The physical checks live in the trackers and are applied
afterwards. Storm counts from the two therefore differ, by design.

**Tracker thresholds are exposed as environment variables**, so criteria such as
minimum duration, genesis latitude, minimum path length and the ocean gates can be
changed and swept without re-running the model. The defaults reproduce the published
configuration.

**Each class can be exported from its own checkpoint.** `main_phase2.py --select-class`
keeps a separate best checkpoint per storm type, because the metric that selects a good
tropical cyclone model is not the one that selects a good atmospheric river model: tropical
cyclones are selected on recall at a precision floor, the other three on IoU. The export
runs one pass per class.

## Requirements

Python 3, with numpy, xarray, netCDF4, scipy, scikit image, matplotlib, cartopy and
PyTorch. Training and inference expect GPUs and use PyTorch distributed data parallel.
The trackers run on CPU.
