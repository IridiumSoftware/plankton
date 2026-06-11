# Changelog

All notable changes to **plankton** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions are git tags.

## [Unreleased]

### Added
- **Morphology atlas** (`--morphology`) — maps the dye-structure regimes
  (foam / network / spots / dispersed) across cohesion × dyeDecay, classifying
  each cell by contrast + Euler characteristic + largest-connected-component
  fraction (descriptors verified by `--morphtest`). Renders a montage of the
  actual structures (`study/morphology_atlas.py`). Finding: chemotaxis makes
  aggregate structure robust across the plane, while *closed-cell foam* is a
  narrow low-cohesion/low-dyeDecay band. See `docs/morphology.md`.
- **Ecology mode** (`e`) — replicator-mutator dynamics over the 8 cohorts, now in
  **both the 2D and 3D engines**: they become game-theory strategies whose
  frequencies evolve by the replicator equation under a payoff matrix (RPS /
  coexistence / dominance presets), with extinct strategies reseeded from the
  leader. Stage 1 is the global/well-mixed core, verified by `--ecologytest` and
  `--ecologysim` (which now exercises the 2D *and* 3D reallocation glue); stage 2
  will make the payoff spatial (RPS spiral waves). Cohort membership is now
  per-agent (`cohortBuffer`) in both engines, so the 2D capture format is at v2
  and the 3D format at v3 (older files still load — the partition is rebuilt).
- **In-app video & GIF recording** (closes #1). `v` records an H.264 mp4 (full
  resolution, streamed via AVAssetWriter); `g` records a downscaled animated GIF
  (ImageIO) — both in 2D and 3D, written to `captures/video/`. Frames are blitted
  off the live drawable and encoded on the GPU completion handler, so recording
  doesn't stall the render loop. Headless encoder check: `--rectest`.
- `ROADMAP.md` — possible future directions (richer brains, video/GIF export,
  scale, boundaries/obstacles, sound-reactive, VR) with design principles to
  preserve; linked from the README.

### Changed
- Headless study modes (`--sweep`, `--map`, `--map3`, `--sdscan`, `--bistab`,
  `--3dspec`) now write their CSVs (and the 3dspec `.bin` dumps) under `data/`
  instead of the repo root; the Python analysis scripts read from `data/` too.
  Keeps the project root clean.
- Moved the Python analysis scripts to `study/`. Each chdir's to the repo root
  on startup, so they resolve `data/`, `presets/`, and `figures/` correctly no
  matter where they're invoked from (e.g. `.venv/bin/python study/make_figures.py`).

## [0.1.0] — 2026-06-10

First public release. A Metal-native flow-field art engine and soft
fluid-dynamics visualizer for Apple Silicon: self-propelled agents with
evolvable 80-parameter Fourier brains forcing a faithful incompressible
Navier–Stokes fluid (real ν∇² viscosity, net-zero force dipoles, Hodge/Leray
projection), in both 2D and 3D.

### Added
- **2D engine** — agent-driven incompressible fluid rendered as a dye field,
  with a grouped, live slider panel (−/+ steppers on every slider).
- **3D engine** (`--3d`) — the model promoted to 3D (160³ grid, Monte-Carlo
  tangent-plane brain), rendered as a ray-marched volume.
- **Faithful fluid** — real ν∇² viscosity (`viscosity`), net-zero force-dipole
  forcing (`dipoleLen`), Jacobi-Leray projection. See `docs/faithful_fluid.md`.
- **Chemotaxis** (`cohesion`) — agents steer up the dye-density gradient and
  aggregate into membranes and cells (2D and 3D).
- **Breeding** — right-click to adopt + mutate the cohort under the cursor
  (2D) or click ray (3D); 8 cohorts; `mutationStrength` knob.
- **Capture** — `c`/`x` save/restore full creature state (`.fluo`/`.fluo3`,
  round-trips bit-for-bit); `j`/`k` record/replay a parameter path. The state
  is captured because the system is hysteretic. See `docs/capture.md`.
- **Research-viz** — vorticity / enstrophy / divergence views, live E(k)
  energy spectrum, and a fluid-only 3D vortex-tube view; `viewMode` /
  `colorMode` dropdowns.
- **`simSpeed`** — run N sim steps per rendered frame (fast-forward), step on
  some frames (slow motion), or 0 to pause.
- **Spectrum study** — headless surveys (`--sweep`, `--map`, …) + Python
  analysis showing the spectral exponent is forcing-controlled, not universal.
  See `docs/spectrum_study.md`.
- **Headless verification** — `--smoke`, `--simtest`, `--3dtest`,
  `--capturetest`, `--spectest`, `--vortprobe`.
- AGPL-3.0 license; minimal macOS menu bar (⌘Q).

### Notes
- Fork of [Fluoddity](https://github.com/aphid91/Fluoddity) (aphid91), built
  from the ideas rather than ported from its OpenGL/GLSL codebase.

[Unreleased]: https://github.com/IridiumSoftware/plankton/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/IridiumSoftware/plankton/releases/tag/v0.1.0
