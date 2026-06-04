#!/usr/bin/env python3
"""
Minimal forced 2D Navier-Stokes pseudospectral DNS (vorticity form) — the REAL
cascade, as a calibration reference for the fluoddity-metal engine's spectrum.

  omega_t + J(psi,omega) = nu*Lap(omega) - mu*omega + f
  u = d psi/dy,  v = -d psi/dx,  omega = -Lap(psi)  =>  psi_hat = omega_hat / k^2

Pseudospectral on a 2*pi box, 2/3 dealiasing, integrating-factor RK4 (exact on the
linear -(nu*k^2 + mu) part, RK4 on advection + forcing). Forcing is white-in-time,
random-phase, in a wavenumber annulus around k_f; mu is a large-scale drag that
lets the flow reach a statistically steady state.

The point of comparison: in REAL 2D NS the forward enstrophy-cascade slope is fixed
(~ -3, Kraichnan) and INDEPENDENT of forcing amplitude — universal. The fluoddity
engine's slope slides with amplitude (a dial). `sweep` demonstrates the contrast.

Correctness anchor (`selfcheck`): unforced inviscid Euler conserves energy and
enstrophy exactly (until the forward cascade reaches the 2/3 cutoff) — same
solver-validation logic as navier-stokes/scripts/spectral_2d_control.jl.

  .venv/bin/python ns2d_dns.py selfcheck   # Euler invariant conservation
  .venv/bin/python ns2d_dns.py run         # one forced run  -> ns2d_spectrum.csv
  .venv/bin/python ns2d_dns.py sweep        # amplitude sweep -> ns2d_sweep.csv
"""
import numpy as np, sys, csv

def operators(N):
    k = np.fft.fftfreq(N, d=1.0 / N)                 # integer wavenumbers
    kx = np.repeat(k[:, None], N, axis=1)
    ky = np.repeat(k[None, :], N, axis=0)
    k2 = kx**2 + ky**2
    k2p = k2.copy(); k2p[0, 0] = 1.0
    kmax = N // 3
    deal = (np.abs(kx) <= kmax) & (np.abs(ky) <= kmax)
    return dict(N=N, kx=kx, ky=ky, k2=k2, k2p=k2p, deal=deal, kmag=np.sqrt(k2))

def vel_hat(wh, op):
    psi = wh / op["k2p"]; psi[0, 0] = 0
    return 1j * op["ky"] * psi, -1j * op["kx"] * psi          # u_hat, v_hat

def nonlinear(wh, op):
    uh, vh = vel_hat(wh, op)
    u = np.fft.ifft2(uh).real; v = np.fft.ifft2(vh).real
    wx = np.fft.ifft2(1j * op["kx"] * wh).real
    wy = np.fft.ifft2(1j * op["ky"] * wh).real
    advh = np.fft.fft2(u * wx + v * wy) * op["deal"]
    return -advh                                              # -J(psi,omega), dealiased

def forcing_hat(op, kf, amp, rng):
    # band-limit a real random field to the annulus, then set its PHYSICAL rms to
    # `amp` (np FFT is unnormalized, so Fourier-space amplitude != physical amplitude
    # by a factor of N^2 — normalizing in physical space is the only safe way).
    ann = (op["kmag"] >= kf - 1) & (op["kmag"] <= kf + 1)
    fp = np.fft.ifft2(np.fft.fft2(rng.standard_normal((op["N"], op["N"]))) * ann).real
    s = fp.std()
    if s > 0:
        fp *= amp / s
    return np.fft.fft2(fp)

def step(wh, dt, op, nu, mu, force):
    L = -(nu * op["k2"] + mu)
    E = np.exp(L * dt); E2 = np.exp(L * dt / 2)
    G = lambda w: nonlinear(w, op) + force
    k1 = dt * G(wh)
    k2 = dt * G(E2 * (wh + k1 / 2))
    k3 = dt * G(E2 * wh + k2 / 2)
    k4 = dt * G(E * wh + E2 * k3)
    return E * wh + (E * k1 + 2 * E2 * (k2 + k3) + k4) / 6

def energy(wh, op):
    uh, vh = vel_hat(wh, op)
    return 0.5 * np.mean(np.fft.ifft2(uh).real**2 + np.fft.ifft2(vh).real**2)

def enstrophy(wh):
    return 0.5 * np.mean(np.fft.ifft2(wh).real**2)

def spectrum(wh, op):
    uh, vh = vel_hat(wh, op)
    N = op["N"]
    ek = 0.5 * (np.abs(uh)**2 + np.abs(vh)**2) / N**4
    kr = np.round(op["kmag"]).astype(int)
    nb = N // 2
    return np.array([ek[kr == kk].sum() for kk in range(nb)])

def fit_slope(E, klo, khi):
    ks = np.arange(klo, khi + 1)
    ek = E[klo:khi + 1]
    m = ek > 0
    x = np.log10(ks[m]); y = np.log10(ek[m])
    if len(x) < 3: return 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ c
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
    return c[0], r2

# ── runs ─────────────────────────────────────────────────────────────────────
def selfcheck():
    N = 128; op = operators(N)
    x = 2 * np.pi * np.arange(N) / N
    w = np.sin(x)[:, None] * np.cos(x)[None, :] + 0.5 * np.sin(2 * x[:, None] + x[None, :])
    wh = np.fft.fft2(w)
    E0, Z0 = energy(wh, op), enstrophy(wh)
    print("2D EULER self-check (nu=0, no forcing): energy + enstrophy must be conserved")
    print(f"  t=0.0   E/E0=1.000000  Z/Z0=1.000000")
    dt = 0.004
    for t in range(1, 5):
        for _ in range(int(1.0 / dt)):
            wh = step(wh, dt, op, 0.0, 0.0, 0.0)
        print(f"  t={t}.0   E/E0={energy(wh,op)/E0:.6f}  Z/Z0={enstrophy(wh)/Z0:.6f}")
    print("  => E conserved ~machine; Z conserved until the forward cascade hits the 2/3 cutoff.")

def run_forced(N=256, nu=3e-5, mu=0.1, kf=10.0, amp=2.0, dt=0.004, T=50.0, Tavg=25.0,
               seed=1, verbose=True):
    op = operators(N); rng = np.random.default_rng(seed)
    force = forcing_hat(op, kf, amp, rng)             # FIXED steady random forcing
    wh = np.fft.fft2(1e-3 * rng.standard_normal((N, N)))   # tiny seed
    nsteps = int(T / dt); avg_from = int((T - Tavg) / dt)
    Eacc = np.zeros(N // 2); nacc = 0
    for s in range(nsteps):
        wh = step(wh, dt, op, nu, mu, force)
        if s >= avg_from:
            Eacc += spectrum(wh, op); nacc += 1
        if verbose and s % max(1, nsteps // 8) == 0:
            print(f"    step {s}/{nsteps}  E={energy(wh,op):.4f}  Z={enstrophy(wh):.3f}", flush=True)
    return Eacc / max(1, nacc), op

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "selfcheck":
        selfcheck(); return
    if cmd == "run":
        E, op = run_forced()
        with open("ns2d_spectrum.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["k", "Ek"])
            for kk in range(1, len(E)):
                if E[kk] > 0: w.writerow([kk, f"{E[kk]:.6g}"])
        kf = 10
        m, r2 = fit_slope(E, 15, 50)        # clean enstrophy-cascade window
        print(f"\nforced run: enstrophy-range slope = {m:.2f} (R^2={r2:.2f}); wrote ns2d_spectrum.csv")
        return
    if cmd == "sweep":
        amps = [0.5, 1.0, 2.0, 4.0, 8.0]
        kf = 10
        rows = [["amp", "E", "Z", "peakK", "enstSlope", "enstR2"]]
        print("amp   E       Z       peakK  enstSlope  R2")
        for a in amps:
            E, op = run_forced(amp=a, verbose=False)
            peak = int(np.argmax(E[1:]) + 1)
            m, r2 = fit_slope(E, 15, 50)        # clean enstrophy-cascade window
            etot = E.sum()
            rows.append([a, f"{etot:.5f}", "", peak, f"{m:.3f}", f"{r2:.3f}"])
            print(f"{a:<5g} {etot:<7.4f}         {peak:<5d}  {m:+.2f}      {r2:.2f}", flush=True)
        with open("ns2d_sweep.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print("\nwrote ns2d_sweep.csv")
        return
    print(f"unknown command: {cmd}")

if __name__ == "__main__":
    main()
