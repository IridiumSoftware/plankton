# fluoddity-metal

A Metal-native flow-field art engine for Apple Silicon — a fork of
[Fluoddity](https://github.com/aphid91/Fluoddity) (by aphid91), rebuilt from the
ideas up rather than ported from its OpenGL/GLSL-compute codebase.

## Idea

Particles never interact directly; they sense and deposit into a shared
**velocity field**. Rather than letting that field merely decay and diffuse
(Physarum-style), we evolve it as a real incompressible flow —
**advect → diffuse → project-to-divergence-free** (the Stable-Fluids loop). The
agents are the *forcing*; the projection step is a discrete Hodge/Helmholtz
decomposition. The result reads as genuine fluid motion (vortices, advected
filaments, shear) without claiming to solve Navier–Stokes from boundary
conditions — an incompressible-field solver driven by agent forcing.

2D first (the inverse-cascade regime: coherent vortices, stable, paintable).
3D later (Monte-Carlo tangent-plane physics + volumetric rendering).

## Build & run

Requires macOS on Apple Silicon, the Swift toolchain (**Command Line Tools is
enough — no full Xcode**), and Metal. Shaders are compiled at runtime.

```sh
swift run fluoddity-metal           # open the window
swift run fluoddity-metal --smoke   # headless toolchain check
```

## Status

- ✅ **Toolchain validated** — SwiftPM + Metal, runtime-compiled shaders, compute
  dispatch, and `atomic_float` (the splat primitive), all green on Apple Silicon
  without Xcode.
- 🛠 **Now** — windowed render loop (AppKit window + MTKView).
- ⏭ **Next** — particle compute (sense → move → atomic-splat) + flow field →
  advection → projection.

---

Built with [Claude Code](https://claude.com/claude-code).
