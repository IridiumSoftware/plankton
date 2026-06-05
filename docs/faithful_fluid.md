# Faithful fluid retrofit — watching creatures on a real Navier–Stokes fluid

*2026-06-05. The 2D engine's fluid is now physically faithful: real ν∇² viscosity +
net-zero force-dipole forcing, on the existing incompressible (Hodge-projected) grid.
This is Phase 4b of the active-turbulence arc in the sibling `navier-stokes` repo.*

## Why

The companion study (`docs/spectrum_study.md`) proved the original engine is a
**forcing-controlled spectral dial, not turbulence** — because its fluid was
unphysical in two ways:

1. **Uniform drag.** The whole velocity buffer was multiplied by a constant each frame
   (`scale_buffer`). Real dissipation is `ν∇²` — `k²`-selective: it kills small scales
   and spares large ones, which is what *makes* an inertial range / cascade. Uniform
   drag damps every mode equally ⇒ no cascade.
2. **Momentum-monopole splat.** Each agent deposited a single force `f = v·forceGain`
   at its position — net momentum ≠ 0. A real active swimmer is *force-free*: it
   exerts a **dipole** (thrust forward, drag back) with **zero net momentum**.

The `navier-stokes` repo built the rigorous version from scratch and characterized it
(SIM_SPEC.md, entries **AT-1..6**):

- **AT-1/AT-2** — a faithful 2D vorticity solver (IF-RK4, exact ν∇²), validated to
  machine precision.
- **AT-2** — under forcing it produces a **universal −3 enstrophy cascade** (the dial
  never had a universal exponent).
- **AT-3** — net-zero dipole agents (momentum machine-zero vs the monopole's O(1)).
- **AT-4** — velocity-sensing agents alone do **not** cluster (active turbulence ≠
  creatures).
- **AT-5** — **chemotaxis** (density-aggregation) **does** cluster them, even on the
  faithful incompressible fluid ⇒ the creatures were chemotaxis-driven aggregation, not
  a fluid artifact.
- **AT-6** — the solver ported to GPU (MPSGraph), GPU≡CPU to ~6 digits, ~100× faster.

This retrofit brings those two fixes back into *this* interactive app so the creatures
can be **watched live** on a faithful fluid.

## What changed (3 files)

| file | change |
|---|---|
| `Shaders.swift` | + `diffuse_velocity` kernel (explicit ν∇² Laplacian, scale-selective); `splat` rewritten as a **net-zero dipole** (+f ahead, −f behind), dye still deposited at the agent center for chemotaxis |
| `Simulation.swift` | the step now does **advect → ν∇² diffuse → weak drag → dipole splat → project**; `dipoleLen` passed to splat, `viscosity` to diffuse |
| `Params.swift` | new live knobs **`viscosity`** (0–0.25, the ν∇² strength) and **`dipoleLen`** (0–6 cells, the dipole separation); `velDamp` is now a *weak* large-scale drag (raise → 1 to let ν∇² dominate) |

Already present and kept: **chemotaxis** (`cohesion`, the AT-5 ingredient) and
**incompressibility** (the Jacobi Leray projection).

## Watch / tune

```
swift run fluoddity-metal              # windowed — watch the creatures
swift run fluoddity-metal --simtest    # headless: faithful pipeline stable + projected (PASS)
```

Live knobs for the faithful feel: **`viscosity`** sets the small-scale dissipation
(the cascade); **`dipoleLen`** the swimmer dipole reach; **`cohesion`** the chemotaxis
strength (turn it up to make the creatures clump — the AT-5 effect); **`velDamp`** →
1.0 lets viscosity be the only dissipation. View modes 1/2 (vorticity / enstrophy) show
the fluid self-organizing into coherent vortices — the real 2D-NS behaviour.

## Faithful vs exact

`diffuse_velocity` is an explicit grid Laplacian (stable for `visc ≤ 0.25`) — a faithful
ν∇², grid-discretized for the live render loop. The *spectrally-exact* IF-RK4 version
(and its machine-precision validation) lives in `navier-stokes` `metal/active_turbulence_gpu.swift`
(AT-6). The physics is the same; this is the interactive, watchable form.
