# Capturing creatures & paths

*2026-06-05. Two ways to keep a creature you grew — because the system is **hysteretic**
(see `navier-stokes` SIM_SPEC.md AT-7): a creature is a function of its **history**, not
the parameter values, so you can't reproduce it by dialing the sliders back to the same
numbers. You have to save the **state**, or replay the **path**.*

## Keys

| key | action |
|---|---|
| **c** | **capture creature** — write the full state (every agent's position + heading, the velocity field, the dye field, all 8 cohort brains, and every parameter) to `captures/creatures/creature_NNN.fluo` |
| **x** | **restore** — load the captured state back (cycles through your captures on repeat presses); replays the exact creature, bit-for-bit |
| **j** | **record path** (toggle) — start recording from the current canvas; tune your way to a creature; press **j** again to stop and save `captures/paths/path_NNN.fluopath` |
| **k** | **replay last path** — restore the recorded start state and re-run the trajectory (applies each parameter change at the frame you made it), re-growing the creature the way you reached it |

(Existing keys unchanged: `r` re-roll brains, `[ ] - =` keyboard tuning, drag = stir,
right-click = breed.)

## Which to use

- **Capture (c/x)** — the reliable "I love this, save it" button. The state restores
  *exactly* (verified bit-for-bit by `--capturetest`), independent of the hysteresis. Use
  it the moment a creature appears.
- **Path (j/k)** — captures the *developmental recipe*: the start canvas + the timeline of
  slider moves. Replay re-grows the creature along the same path. (The GPU splat uses atomic
  adds, so replay lands in the *same attractor* but isn't bit-identical — which is the point:
  it reproduces the **growth**, not a frozen frame.) Use it to capture *how* you grew
  something, or to share a recipe.

## Files

```
captures/
  creatures/creature_001.fluo      # full-state snapshots (binary: magic "FLUO" + buffers + params)
  paths/path_001.fluopath          # start state + (frame, param, value) change list
  creatures3d/creature3d_001.fluo3 # same, 3D engine (magic "FLU3")
  paths3d/path3d_001.fluo3path     # same, 3D engine
```

All live under the run directory. `.fluo` is `~`(8·N² + 16·particleCount)` bytes — a few
MB to ~28 MB at the default 1024² / 2²⁰ agents. The dimensions are checked on load (a capture
only restores into a sim of the same grid + agent count).

The 3D format is versioned: **v2** stores all 8 cohort brains (since 3D breeding landed);
**v1** files (single brain) still load — the brain is replicated into every cohort, so a
breed on a restored v1 creature starts from 7 mutants of its original brain.

## Why bit-for-bit matters

The full-state capture is a byte copy of the GPU-shared buffers, so restore is exact — no
float drift. That's what makes it the dependable way to hold a creature that the hysteresis
(AT-7) would otherwise make irreproducible from parameters alone.
