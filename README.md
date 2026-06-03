# fluoddity-metal

A Metal-native flow-field art engine and soft fluid-dynamics visualizer for
Apple Silicon — a fork of [Fluoddity](https://github.com/aphid91/Fluoddity) (by
aphid91), rebuilt from the ideas up in Swift + Metal rather than ported from its
OpenGL/GLSL-compute codebase. Both a **2D** engine and a **3D** engine.

## Idea

Particles (agents) never interact directly; they sense and deposit into a shared
**velocity field**. Rather than letting that field merely decay and diffuse
(Physarum-style), we evolve it as a real incompressible flow —
**advect → project-to-divergence-free** (the Stable-Fluids loop). The agents are
the *forcing*; the projection is a discrete **Hodge/Helmholtz decomposition**
(a Jacobi pressure-Poisson solve). The result reads as genuine fluid motion —
vortices, advected filaments, shear — without claiming to solve Navier–Stokes
from boundary conditions: an incompressible-field solver driven by agent forcing.

Each agent steers through an **80-parameter symmetric Fourier "brain"**
(10 Fourier centers, evaluated twice mirror-averaged so there's no
clockwise/counter-clockwise bias), plus a **chemotaxis** term that lets them
aggregate into membranes and cells. You **breed** behaviors by re-rolling and
selecting (right-click in 2D), and the same engine doubles as a **soft
visualizer of the underlying fluid** (vorticity, enstrophy, divergence, energy).

## Build & run

Requires macOS on Apple Silicon, the Swift toolchain (**Command Line Tools is
enough — no full Xcode**), and Metal. Shaders are compiled at runtime.

```sh
swift run fluoddity-metal            # 2D engine (the main app)
swift run fluoddity-metal --3d       # 3D engine (orbital, volumetric)

# headless verification (no window — runs the compute path + checks):
swift run fluoddity-metal --smoke    # toolchain / atomic_float
swift run fluoddity-metal --simtest  # 2D fluid incompressibility + stability
swift run fluoddity-metal --3dtest   # 3D fluid incompressibility + stability
```

## 2D engine

An agent-driven incompressible 2D fluid rendered as a colored dye field with the
agents overlaid. A live, grouped slider panel (top-left) tunes everything.

- **Mouse drag** — stir the fluid and inject dye.
- **Right-click** — adopt + mutate the cohort under the cursor (directed
  evolution; 8 cohorts, tinted by color).
- **`r`** — re-roll all brains (or use the **Brain** button).
- **Save / Load / Reset** buttons — presets (params + brains) to `presets/*.json`.
- **Diag** toggle — research mode (HUD + plot + field calcs) vs art mode (perf).
- **`viewMode`** slider — dye art / **vorticity** ω / **enstrophy** |ω|² /
  **divergence** field. With the **E / Z / |ω|ₘₐₓ / div** HUD and a scrolling
  **E/Z time-series plot** (the research-viz layer; ties to the
  `navier-stokes` program — ω is exactly that 2D solver's state variable).

## 3D engine (`--3d`)

The whole model promoted to 3D, the owner's `3dMC` idea realized: a 128³
incompressible fluid (6-neighbour Hodge projection) forced by agents whose
**Monte-Carlo tangent-plane brain** samples random planes containing their
velocity, runs the 2D symmetric brain per plane, and integrates the steering into
3D. Rendered as a **ray-marched volume** of the agents' dye density.

- **Drag** to orbit · **scroll** to zoom.
- **`r`** — re-roll the 3D brain.
- **`[` / `]`** — dim / brighten the volume density.

## How it's built

Pure SwiftPM (no Xcode project), AppKit + Metal, runtime-compiled `.metal`
sources held as Swift strings. No external dependencies. Every step is verified
headlessly (the compute path + incompressibility checks) since the rendered
window can only be eyeballed; see the `--*test` flags above.

---

Fork of [aphid91/Fluoddity](https://github.com/aphid91/Fluoddity) — built from
the ideas, idea-sharing collaboration. Made with
[Claude Code](https://claude.com/claude-code).
