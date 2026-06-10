# Roadmap

Possible future directions for **plankton**. Nothing here is committed or
scheduled — it's a record of where the project could go, with honest notes on
value and effort. Contributions and discussion welcome (open an issue).

## Design principles to preserve

These are intentional and should survive any extension:

- **Directed (aesthetic) evolution is the interaction model.** You breed by
  *selecting* what you like (right-click), not by specifying a fitness
  function. This keeps it fun, controllable, and exploratory — a creature zoo,
  not an optimizer. Any added automatic search (GA, novelty search) should be
  an *opt-in mode*, never a replacement for the human-in-the-loop default.
- **Faithful fluid first.** The fluid stays a real incompressible
  Navier–Stokes flow (ν∇² viscosity, net-zero dipole forcing, Hodge/Leray
  projection). Features shouldn't quietly trade physical faithfulness for
  effect.
- **Verify headlessly.** Every engine change keeps a `--*test` that proves the
  compute path without needing the window.

## Directions

### Brain & evolution
The 80-parameter symmetric Fourier brain works well for *smooth* behaviors,
but it's a relatively low-dimensional parametric search. Richer brains are a
natural extension:
- **Neural-net / CPPN brains** — higher capacity, more behavioral diversity;
  the sense→act interface already exists, so this is mostly swapping the
  `fourier()` evaluation for a small MLP/CPPN and widening the rule buffer.
- **Other ALife controllers** — recurrent/stateful brains (memory), or
  GRN-style update rules.
- **Optional automatic search modes** — novelty search or a fitness-driven GA
  as a *toggle* alongside directed evolution (see the design principle above).
- Keep the mirror-symmetry trick (no built-in handedness) for whatever brain
  replaces the Fourier sum.

### Sharing: video / GIF export
No built-in capture-to-video yet — currently you screenshot, or capture state
(`c`) and re-shoot. An in-app **record → mp4/GIF** (offscreen render loop →
frame encode via AVFoundation, or PNG frames + an ffmpeg note) would be
**high-value, modest-effort**, and the single biggest lever for sharing what
the engine produces. Strong candidate for the next feature.

### Scale
- **Larger grids / more agents.** `fieldDim` and `particleCount` are already
  parameters; this is mainly perf tuning (threadgroup sizes, the Jacobi
  iteration count, ray-march step count) rather than new architecture.
- **Multi-species interaction.** 3D already runs 8 cohorts; "species" that
  sense/affect each other differently (predator/prey, distinct dye channels,
  cross-cohort chemotaxis) would be a deeper change to the move/splat kernels.

### Environment & interaction
- **Boundaries / obstacles.** Walls and solid bodies would require enforcing
  no-slip / no-penetration in the projection solve (masked cells, Neumann
  pressure BCs) — touches the core fluid step, medium effort, high expressive
  payoff.
- **Sound-reactive mode.** Drive forcing / params from live audio (FFT bands →
  forceGain, cohesion, injection) — small-to-medium, great for installations.

### Immersion
- **VR / immersive view.** A stereoscopic / head-tracked volume view
  (visionOS or OpenXR). The 3D ray-march is the natural basis, but this is a
  large lift (new render path, input model, performance budget).

## Rough priority

A suggested ordering by value-to-effort, not a commitment:

1. **Video / GIF export** — highest sharing value, modest effort.
2. **Larger grids / more agents** — mostly perf tuning, unlocks richer scenes.
3. **Sound-reactive mode** — small, high-delight for installations.
4. **Boundaries / obstacles** — medium, touches the solver, big expressive win.
5. **NN / CPPN brains** — opens the behavior space; clean interface already.
6. **Multi-species interaction** — deeper kernel work.
7. **VR / immersive view** — largest lift; do once the rest is mature.
