#!/usr/bin/env python
"""Repeat the rotation figures of search_spectra_rotation_orbit.ipynb.

Which asteroids and which figures to write is set in select_provid_all(),
select_provid_list(), select_provid_rotation(), or select_provid_medellin().
The active one is the call at the start of main().

Writes, into figall/, as WebP at half the previous resolution:
  <provid>_lightcurve_allnights_H.webp
  <provid>_periodogram.webp
  <provid>_phase_allbands.webp
  <provid>_timemodel_H.webp
  <provid>_phase_model_all_periods.webp
  <provid>_phase_model.png
  <provid>_phase_spheres_fit.png
  <provid>_phase_ellipsoid_fit.png
  period_compare_lsm.webp
  period_compare_fourier.webp
"""

import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.timeseries import LombScargle
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from util import ALL_BANDS, BANDS, band_colors

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figcomp"
ROTATION_CATALOG = DATA_DIR / "asteroids-with-rotation.csv"
MEDELLIN_CATALOG = DATA_DIR / "medellin-vro-catalogue.csv"
DESIGNATIONS = DATA_DIR / "mpc_designations.csv"
OBS_FILE = DATA_DIR / "obs_sbn_X05_full.parquet"

BAND = "Lr"
PROP = "H"
PERIODS_H = (2 / 60, 24)  # photometric periods searched, in hours
N_FREQ = 20000
MIN_N_OBS = 200
DPI = 150  # half of the previous 300 dpi
WEBP_QUALITY = 80
ASTEROID_PROVID = re.compile(r"^\d{4} [A-Z]{2}\d+[A-Z]?$")

# Per-asteroid figure names used in main().
FIGURES = ("lightcurve", "periodogram", "phase", "phase_model", "shape_fit", "timemodel", "literature", "compare")

def set_fig_dir(fig_dir):
    """Send this run's figures to fig_dir. A relative path is taken from ROOT."""
    global FIG_DIR
    fig_dir = Path(fig_dir)
    if not fig_dir.is_absolute():
        fig_dir = ROOT / fig_dir
    FIG_DIR = fig_dir
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures: {FIG_DIR}")
    return FIG_DIR


def slug(provid):
    return provid.replace(" ", "_")


def output_path(kind, provid):
    stem = {
        "lightcurve": "lightcurve_allnights_H",
        "periodogram": "periodogram",
        "phase": "phase_allbands",
        "timemodel": "timemodel_H",
        "literature": "phase_model_all_periods",
        "phase_model": "phase_model",
        "spheres": "phase_spheres_fit",
        "ellipsoid": "phase_ellipsoid_fit",
    }[kind]
    suffix = ".png" if kind in {"phase_model", "spheres", "ellipsoid"} else ".webp"
    return FIG_DIR / f"{slug(provid)}_{stem}{suffix}"


def save_figure(fig, path):
    """Write a WebP at DPI. The path suffix is forced to .webp."""
    path = Path(path).with_suffix(".webp")
    fig.savefig(path, dpi=DPI, pil_kwargs={"quality": WEBP_QUALITY, "method": 6})
    plt.close(fig)
    return path


def format_elapsed(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def check_figures(figures, where):
    unknown = set(figures) - set(FIGURES)
    if unknown:
        names = ", ".join(sorted(unknown))
        allowed = ", ".join(FIGURES)
        raise SystemExit(f"Unknown figure(s) in {where}: {names}. Use: {allowed}")


def load_df_set():
    """Photometry table, filtered as in the notebook (df_set)."""
    columns = ["obstime", "permid", "provid", "ra", "dec", "band", "mag", "rmsmag"]
    df_full = pd.read_parquet(OBS_FILE, columns=columns)
    print(f"Original number of observations: {len(df_full)}")

    df_set = df_full[~(df_full["provid"].isna() & df_full["permid"].isna())].reset_index(drop=True)
    print(f"Number of observations with provid or permid: {len(df_set)}")

    df_set = df_set[
        df_set["provid"].apply(lambda pid: bool(ASTEROID_PROVID.match(str(pid))))
    ].reset_index(drop=True)
    n_provid = df_set["provid"].dropna().nunique()
    print(f"Number of observations with asteroid provid: {len(df_set)}")
    print(f"Number of objects with provid: {n_provid}")

    df_set["band"] = df_set["band"].apply(
        lambda b: f"L{b}" if pd.notnull(b) and len(str(b)) == 1 else b
    )
    df_set = df_set.sort_values(
        by=["provid", "band", "obstime"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    df_set[["ra", "dec", "mag", "rmsmag"]] = df_set[["ra", "dec", "mag", "rmsmag"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df_set["obstime"] = pd.to_datetime(df_set["obstime"])
    df_set["night"] = df_set["obstime"].dt.floor("D")
    return df_set


def observation_counts(df_set):
    """df_obs from the notebook: number of observations per asteroid provid."""
    provids = df_set["provid"].dropna().unique()
    df_obs = (
        df_set[df_set["provid"].isin(provids)]
        .groupby("provid")
        .size()
        .reset_index(name="n_obs")
        .sort_values(by="n_obs", ascending=False)
    )
    return df_obs.reset_index(drop=True)


def load_rotation_catalog():
    """df_rot from the notebook: asteroids-with-rotation.csv."""
    df_rot = pd.read_csv(ROTATION_CATALOG)
    df_rot["provid"] = df_rot["provid"].dropna().astype(str).str.strip()
    df_rot = df_rot[df_rot["provid"] != ""].drop_duplicates("provid")
    return df_rot.reset_index(drop=True)


def load_mpc_designations():
    """provid -> mpc_designation, from data/mpc_designations.csv."""
    df = pd.read_csv(DESIGNATIONS)
    df["provid"] = df["provid"].astype(str).str.strip()
    return df.drop_duplicates("provid").set_index("provid")["mpc_designation"]


def load_geometry(designation):
    if designation is None or pd.isna(designation) or str(designation).strip() == "":
        return pd.DataFrame(columns=["obstime", "mag", "H"])
    table = pq.read_table(
        DATA_DIR / "obs_final_elements_geometry_H.parquet",
        columns=["obstime", "mag", "H"],
        filters=[("mpc_designation", "==", str(designation))],
    )
    df = table.to_pandas()
    df["obstime"] = pd.to_datetime(df["obstime"])
    return df


def prepare_object(df_set, provid, designations):
    df_obj = df_set[df_set["provid"] == provid].copy()
    if df_obj.empty:
        return df_obj
    df_obj = df_obj.sort_values(["band", "obstime"]).reset_index(drop=True)
    t0 = df_obj["obstime"].iloc[0]
    df_obj["t_days"] = (df_obj["obstime"] - t0).dt.total_seconds() / 86400.0
    designation = designations.get(provid, provid)
    df_spy = load_geometry(designation)
    df_obj = pd.merge(df_obj, df_spy, on="obstime", how="left", suffixes=("", "_spy"))
    return df_obj


def plot_lightcurve_h(df_obj, provid):
    path = output_path("lightcurve", provid)
    if path.exists():
        return path
    nights = df_obj.night.unique()
    n_nights = len(nights)
    fig, axs = plt.subplots(n_nights + 1, 1, figsize=(10, 3 * n_nights))
    if n_nights + 1 == 1:
        axs = [axs]

    ax = axs[0]
    for b in BANDS:
        sb = df_obj[df_obj["band"] == b]
        if len(sb) > 0:
            ax.errorbar(
                sb["obstime"], sb["H"], yerr=sb["rmsmag"], fmt="o",
                label=f"{b} (N={len(sb)})", color=band_colors.get(b, "k"),
                markersize=4, alpha=0.85,
            )
    ax.set_xlabel("Observation Time")
    ax.set_ylabel("H")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Light Curve of Asteroid {provid} (all Nights)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=True)
    ax.grid(True, alpha=0.3)

    for i, night in enumerate(nights):
        night_str = pd.to_datetime(night).strftime("%Y-%m-%d")
        ax = axs[i + 1]
        t_night = df_obj[df_obj["night"] == night]
        t_night_0 = t_night["obstime"].min()
        hours = (t_night["obstime"] - t_night_0).dt.total_seconds() / 3600.0
        for b in ALL_BANDS:
            if b == "Lu":
                continue
            sel = t_night["band"] == b
            if not sel.any():
                continue
            ax.errorbar(
                hours[sel], t_night.loc[sel, "H"], yerr=t_night.loc[sel, "rmsmag"],
                fmt="o", label=f"{b} (N={sel.sum()})", color=band_colors.get(b, "k"),
                markersize=4, alpha=0.85,
            )
        ax.set_title(f"Night {night_str}")
        ax.set_xlabel("Time since start [h]")
        ax.set_ylabel("H")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=True)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return save_figure(fig, output_path("lightcurve", provid))


def lightcurve_amplitudes(ls_band, t_band, y_band, frequency):
    """Peak-to-peak amplitudes of the periodogram model and of the photometry.

    amp_lsm is max(model) - min(model) of ls_band.model over one cycle at
    `frequency`. amp_photo is the width of y_band between the 5th and 95th
    percentiles.
    """
    t0 = float(np.min(t_band))
    phase = np.linspace(0, 1, 400, endpoint=False)
    y_model = ls_band.model(t0 + phase / frequency, frequency)
    amp_lsm = float(np.ptp(y_model))
    amp_photo = float(np.percentile(y_band, 95) - np.percentile(y_band, 5))
    return amp_lsm, amp_photo


def periodogram(df_obj, provid, save=True):
    df_band = df_obj[df_obj["band"] == BAND].dropna(subset=[PROP]).copy()
    if len(df_band) < 5:
        print(f"  skip periodogram for {provid}: {len(df_band)} {BAND} points with {PROP}")
        return None

    t_band = df_band["t_days"].to_numpy()
    y_band = df_band[PROP].to_numpy()
    dy_band = df_band["rmsmag"].fillna(0.01).to_numpy()
    freq_grid = np.linspace(24 / PERIODS_H[1], 24 / PERIODS_H[0], N_FREQ)
    ls_band = LombScargle(t_band, y_band, dy_band)
    power = ls_band.power(freq_grid)

    main_peak_power = np.max(power)
    best_freq = freq_grid[np.argmax(power)]
    best_phot_period = 24.0 / best_freq
    best_rot_period = 2.0 * best_phot_period

    peaks, _ = find_peaks(power)
    high_power_peaks = np.intersect1d(peaks, np.where(power >= 0.8 * main_peak_power)[0])
    peak_data = []
    for idx in high_power_peaks:
        freq = freq_grid[idx]
        peak_data.append({
            "frequency": freq,
            "rotational_period_h": 2.0 * 24.0 / freq,
            "relative_power": power[idx] / main_peak_power,
        })
    if not peak_data:
        peak_data.append({
            "frequency": best_freq,
            "rotational_period_h": best_rot_period,
            "relative_power": 1.0,
        })
    df_peaks = pd.DataFrame(peak_data).sort_values("relative_power", ascending=False).reset_index(drop=True)
    all_rot_periods = df_peaks["rotational_period_h"].to_numpy()

    path = output_path("periodogram", provid)
    if save and not path.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        period_grid_hours = (24.0 / freq_grid) * 2.0
        ax.plot(period_grid_hours, power, color="b", lw=1.5)
        ax.axvline(best_rot_period, color="r", linestyle="--", lw=2, label=f"Peak Period = {best_rot_period:.2f} h")
        for period in all_rot_periods:
            ax.axvline(period, color="g", linestyle=":", alpha=1, lw=1)
        ax.set_xlabel("Rotational Period [hours]")
        ax.set_ylabel("Spectral Power")
        ax.set_title(f"Lomb-Scargle Periodogram for Asteroid {provid} ({BAND} Band, property={PROP})")
        ax.set_xlim(0, 10)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = save_figure(fig, path)
    elif not save or not path.exists():
        path = None

    freq_sel = df_peaks.iloc[0]["frequency"]
    per_sel = df_peaks.iloc[0]["rotational_period_h"]
    amp_lsm, amp_photo = lightcurve_amplitudes(ls_band, t_band, y_band, freq_sel)
    print(
        f"  {provid}: P_rot = {per_sel:.2f} h ({BAND}, {PROP}), "
        f"amp_lsm = {amp_lsm:.3f}, amp_photo = {amp_photo:.3f}"
    )
    return {
        "df_band": df_band,
        "t_band": t_band,
        "y_band": y_band,
        "dy_band": dy_band,
        "ls_band": ls_band,
        "best_freq": best_freq,
        "best_phot_period": best_phot_period,
        "best_rot_period": best_rot_period,
        "freq_sel": freq_sel,
        "per_sel": per_sel,
        "amp_lsm": amp_lsm,
        "amp_photo": amp_photo,
        "df_peaks": df_peaks,
        "path": path,
    }

def plot_phase_allbands(df_obj, provid, fit):
    path = output_path("phase", provid)
    if path.exists():
        return path
    t0 = fit["t_band"].min()
    freq_sel = fit["freq_sel"]
    per_sel = fit["per_sel"]
    present = [b for b in ALL_BANDS if (df_obj["band"] == b).any()]
    m_max = {b: df_obj.loc[df_obj["band"] == b, PROP].min() for b in present}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for b in present:
        sb = df_obj[df_obj["band"] == b]
        phase_b = np.mod((sb["t_days"].to_numpy() - t0) * freq_sel / 2.0, 1.0)
        dy = sb["rmsmag"].fillna(0.01)
        marker = "o" if b.startswith("L") else "s"
        m = m_max[b]
        style = dict(
            fmt=marker, ms=4, alpha=0.8, color=band_colors.get(b, "k"),
            label=f"{b} (N={len(sb)}, m_max={m:.2f})",
        )
        axes[0].errorbar(phase_b, sb[PROP], yerr=dy, **style)
        axes[1].errorbar(phase_b, sb[PROP] - m, yerr=dy, **style)

    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Observed")
    axes[1].set_ylabel(r"$m - m_{\mathrm{max}}$")
    axes[1].set_title("Normalized to the brightest point of each filter")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Phase at the rotation period")
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.suptitle(f"Phase curve of {provid}, $P_{{rot}}$ = {per_sel:.2f} h")
    fig.tight_layout()
    return save_figure(fig, output_path("phase", provid))


def plot_timemodel_h(provid, fit):
    path = output_path("timemodel", provid)
    if path.exists():
        return path
    df_band = fit["df_band"]
    nights_band = df_band["night"].drop_duplicates().sort_values()
    n_nights_band = len(nights_band)
    fig, axs = plt.subplots(n_nights_band, 1, figsize=(10, 2.6 * n_nights_band), sharey=True)
    if n_nights_band == 1:
        axs = [axs]

    for ax, night in zip(axs, nights_band):
        sel = df_band["night"] == night
        t_night = df_band.loc[sel, "t_days"].to_numpy()
        y_night = df_band.loc[sel, PROP].to_numpy()
        dy_night = df_band.loc[sel, "rmsmag"].fillna(0.01).to_numpy()
        t_left, t_right = t_night.min(), t_night.max()
        pad = max(0.01, 0.05 * (t_right - t_left))
        t_grid = np.linspace(t_left - pad, t_right + pad, 400)
        y_grid = fit["ls_band"].model(t_grid, fit["best_freq"])
        hours = (t_night - t_left) * 24.0
        hours_grid = (t_grid - t_left) * 24.0
        ax.errorbar(
            hours, y_night, yerr=dy_night, fmt="o", ms=4, alpha=0.85,
            color="0.2", ecolor="0.6", label="Observed magnitude", zorder=3,
        )
        ax.plot(
            hours_grid, y_grid, color="C3", lw=1.8,
            label=f"Periodogram model, P = {fit['best_phot_period']:.2f} h", zorder=2,
        )
        ax.set_title(pd.to_datetime(night).strftime("%Y-%m-%d"))
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)

    axs[0].invert_yaxis()
    axs[0].legend(loc="best", frameon=True)
    axs[-1].set_xlabel("Hours since the first observation of the night")
    fig.suptitle(f"{provid}: observed light curve and periodogram model")
    fig.tight_layout()
    return save_figure(fig, output_path("timemodel", provid))


def literature_periods(df_rot, provid):
    """Fourier and LSM periods from df_rot, in the notebook's panel order."""
    row = df_rot.loc[df_rot["provid"] == provid]
    if row.empty:
        return {}
    periods = {}
    for column, name in (
        ("period_fourier_hr", "Literature (Fourier)"),
        ("period_lsm_hr", "Literature (Lomb-Scargle)"),
    ):
        value = pd.to_numeric(row.iloc[0][column], errors="coerce")
        if np.isfinite(value) and value > 0:
            periods[name] = float(value)
    return periods


def Lr_phase_curve(provid, fit):
    """Folded Lr photometry and the periodogram model, as in shape_fit.ipynb."""
    path = output_path("phase_model", provid)
    if path.exists():
        return path

    t_band = fit["t_band"]
    y_band = fit["y_band"]
    dy_band = fit["dy_band"]
    freq_sel = fit["freq_sel"]
    ls_band = fit["ls_band"]
    per_rot = fit["per_sel"]
    t0 = t_band.min()
    phase_rot = np.mod((t_band - t0) * freq_sel / 2.0, 1.0)
    phase_rot_grid = np.linspace(0, 1, 400)
    y_model_rot = ls_band.model(t0 + phase_rot_grid * 2.0 / freq_sel, freq_sel)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    curve_color = band_colors.get(BAND, "k")
    ax.errorbar(
        phase_rot, y_band, yerr=dy_band, fmt="o", ms=4, alpha=0.75,
        color=curve_color, ecolor="0.6", label="Magnitude",
    )
    ax.plot(
        phase_rot_grid, y_model_rot, color=curve_color, lw=2,
        label=rf"Periodogram model, $P_{{rot}}$ = {per_rot:.2f} h",
    )
    ax.hlines(
        [y_model_rot.min(), y_model_rot.max()], 0, 1,
        colors=curve_color, linestyles="--", alpha=0.5, label="Model min/max",
    )
    ax.hlines(
        [np.percentile(y_band, 1), np.percentile(y_band, 99)], 0, 1,
        colors="grey", linestyles=":", alpha=0.7, label="Photometry 1%/99%",
    )
    ax.set_xlabel("Phase at the rotation period ($2P$)")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    ax.set_ylabel("Magnitude")
    ax.set_title(
        f"Folded light curve of {provid} ({PROP},{BAND}), "
        rf"$P_{{rot}}$ = {per_rot:.2f} h"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_phase_model_all_periods(provid, fit, lit_periods):
    """Folded light curve at our period and at the literature periods."""
    path = output_path("literature", provid)
    if path.exists():
        return path
    periods = {"VRO": fit["best_rot_period"], **lit_periods}
    t_band = fit["t_band"]
    y_band = fit["y_band"]
    dy_band = fit["dy_band"]
    ls_band = fit["ls_band"]
    t0 = t_band.min()
    curve_color = band_colors.get(BAND, "k")

    fig, axs = plt.subplots(len(periods), 1, figsize=(10, 4.5 * len(periods)), squeeze=False)
    for ax, (name, per) in zip(axs.ravel(), periods.items()):
        freq_sel = 24.0 / (per / 2.0)
        phase_rot = np.mod((t_band - t0) * freq_sel / 2.0, 1.0)
        phase_rot_grid = np.linspace(0, 1, 400)
        y_model_rot = ls_band.model(t0 + phase_rot_grid * 2.0 / freq_sel, freq_sel)
        ax.errorbar(
            phase_rot, y_band, yerr=dy_band, fmt="o", ms=4, alpha=0.75,
            color=curve_color, ecolor="0.6", label="Magnitude",
        )
        ax.plot(
            phase_rot_grid, y_model_rot, color=curve_color, lw=2,
            label=rf"Periodogram model, $P_{{rot}}$ = {per:.2f} h",
        )
        ax.set_xlabel("Phase at the rotation period ($2P$)")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True)
        ax.set_ylabel("Magnitude")
        ax.set_title(
            f"Folded light curve of {provid} ({PROP},{BAND}), "
            rf"$P_{{rot}}$ = {per:.2f} h [{name}]"
        )
    fig.tight_layout()
    return save_figure(fig, output_path("literature", provid))


def plot_period_comparison(df_rot, measured):
    """Our Lomb-Scargle rotation period against the two literature periods."""
    specs = [
        ("period_lsm_hr", "LSM", "period_compare_lsm.webp"),
        ("period_fourier_hr", "Fourier", "period_compare_fourier.webp"),
    ]
    literature = df_rot.set_index("provid")
    paths = []
    for column, label, filename in specs:
        rows = []
        for provid, p_own in measured.items():
            if provid not in literature.index or column not in literature.columns:
                continue
            p_lit = pd.to_numeric(literature.at[provid, column], errors="coerce")
            if not np.isfinite(p_lit) or p_lit <= 0 or not np.isfinite(p_own) or p_own <= 0:
                continue
            rows.append((provid, float(p_lit), float(p_own)))
        if not rows:
            print(f"  no pairs for the {label} comparison")
            continue
        path = FIG_DIR / filename
        if path.exists():
            print(f"  {path.name} already exists")
            paths.append(path)
            continue

        provids, x, y = zip(*rows)
        x = np.asarray(x)
        y = np.asarray(y)
        lo = min(x.min(), y.min()) * 0.7
        hi = max(x.max(), y.max()) * 1.4

        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax.scatter(x, y, s=28, color="#1f4e79", zorder=3)
        for provid, xi, yi in rows:
            ax.annotate(provid, (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=6, alpha=0.85)
        ax.plot([lo, hi], [lo, hi], color="0.45", lw=1, label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Literature period, {label} [h]")
        ax.set_ylabel("Lomb-Scargle period from our photometry [h]")
        ax.set_title(f"Rotation period: our photometry vs literature ({label})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", frameon=True)
        fig.tight_layout()
        path = save_figure(fig, path)
        paths.append(path)
        print(f"  {path.name} ({len(rows)} asteroids)")
    return paths


def touching_spheres_area(theta, q, phi):
    """Projected area of two touching spheres (r1 = 1, r2 = q)."""
    r1, r2 = 1.0, q
    d = (r1 + r2) * np.sqrt(np.cos(theta)**2 + np.sin(theta)**2 * np.cos(phi)**2)
    d = np.maximum(d, 1e-9)
    xi1 = np.arccos(np.clip((d**2 + r1**2 - r2**2) / (2 * d * r1), -1, 1))
    xi2 = np.arccos(np.clip((d**2 + r2**2 - r1**2) / (2 * d * r2), -1, 1))
    return (r1**2 * (np.pi - xi1 + np.sin(xi1) * np.cos(xi1))
            + r2**2 * (np.pi - xi2 + np.sin(xi2) * np.cos(xi2)))


def touching_spheres_mag(phase, m_sep, q, cos2phi, theta0):
    """Magnitude versus rotational phase. m_sep is the fully separated magnitude."""
    phi = np.arccos(np.sqrt(np.clip(cos2phi, 0, 1)))
    area = touching_spheres_area(2 * np.pi * phase + theta0, q, phi)
    return m_sep - 2.5 * np.log10(area / (np.pi * (1.0 + q**2)))


def ellipsoid_area(theta, b, c, phi, a=1.0):
    """Projected area of a triaxial ellipsoid rotating about c."""
    sphi, cphi = np.sin(phi), np.cos(phi)
    return np.pi * np.sqrt((a * c * np.cos(theta) * sphi)**2
                           + (b * c * np.sin(theta) * sphi)**2
                           + (a * b * cphi)**2)


def ellipsoid_mag(phase, m_ab, b, c_over_b, cos2phi, theta0):
    """Magnitude versus rotational phase. m_ab is the a-b face, with a = 1."""
    phi = np.arccos(np.sqrt(np.clip(cos2phi, 0, 1)))
    area = ellipsoid_area(2 * np.pi * phase + theta0, b, c_over_b * b, phi)
    return m_ab - 2.5 * np.log10(area / (np.pi * b))


def _rotational_phase(t, t0, freq):
    return np.mod((t - t0) * freq / 2.0, 1.0)


def _best_least_squares(residuals, lower, upper, starts):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    best = None
    for p0 in starts:
        p0 = np.clip(np.asarray(p0, dtype=float), lower + 1e-10, upper - 1e-10)
        sol = least_squares(residuals, p0, bounds=(lower, upper))
        if best is None or sol.cost < best.cost:
            best = sol
    return best


def _fit_scale(cost, jac, n_data, n_param):
    ndof = n_data - n_param
    chi_r = 2 * cost / ndof if ndof > 0 else np.nan
    cov = np.linalg.pinv(jac.T @ jac) * chi_r
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    return chi_r, err


def _fit_touching_spheres(t, y, dy, freq0, t0):
    def residuals(p):
        m_sep, q, cos2phi, theta0, freq = p
        model = touching_spheres_mag(_rotational_phase(t, t0, freq), m_sep, q, cos2phi, theta0)
        return (y - model) / dy

    lower = [y.min() - 2, 0.05, 0.0, -2 * np.pi, freq0 * 0.98]
    upper = [y.max() + 2, 20.0, 1.0, 4 * np.pi, freq0 * 1.02]
    rng = np.random.default_rng(42)
    starts = [
        [np.median(y) + rng.normal(0, 0.3), rng.uniform(0.2, 3.0), rng.uniform(0, 1),
         rng.uniform(0, 2 * np.pi), freq0 * (1 + rng.normal(0, 5e-4))]
        for _ in range(100)
    ]
    best = _best_least_squares(residuals, lower, upper, starts)
    m_sep, q_fit, cos2phi, theta0, freq = best.x
    chi_r, err = _fit_scale(best.cost, best.jac, len(t), len(best.x))
    q_can = min(q_fit, 1 / q_fit)
    q_err = err[1] * (1 if q_fit <= 1 else 1 / q_fit**2)
    phi = np.arccos(np.sqrt(np.clip(cos2phi, 0, 1)))
    phi_err = err[2] / max(2 * np.sqrt(max(cos2phi, 0)) * np.sin(phi), 1e-3)
    phi_is_limit = cos2phi < 2 * err[2]
    ph_grid = np.linspace(0, 1, 2000)
    m_grid = touching_spheres_mag(ph_grid, m_sep, q_fit, cos2phi, theta0)
    m_bright, m_faint = float(m_grid.min()), float(m_grid.max())
    return {
        "m_sep": float(m_sep),
        "q_fit": float(q_fit),
        "q_sph": float(q_can),
        "q_sph_err": float(q_err),
        "cos2phi": float(cos2phi),
        "theta0": float(theta0),
        "freq": float(freq),
        "chi_r_sph": float(chi_r),
        "phi_sph_deg": float(np.rad2deg(phi)),
        "phi_sph_err_deg": np.nan if phi_is_limit else float(np.rad2deg(phi_err)),
        "amp_sph": m_faint - m_bright,
        "m_mid_sph": 0.5 * (m_bright + m_faint),
        "m_bright": m_bright,
        "m_faint": m_faint,
        "p_rot_sph_h": float(48 / freq),
        "ph_grid": ph_grid,
        "m_grid": m_grid,
        "rms": float(np.std(best.fun * dy)),
    }


def _fit_ellipsoid(t, y, dy, freq0, t0):
    def residuals(p):
        m_ab, b, theta0, freq = p
        model = ellipsoid_mag(_rotational_phase(t, t0, freq), m_ab, b, 1.0, 0.0, theta0)
        return (y - model) / dy

    lower = [y.min() - 2, 0.05, -2 * np.pi, freq0 * 0.98]
    upper = [y.max() + 2, 1.0, 4 * np.pi, freq0 * 1.02]
    rng = np.random.default_rng(7)
    starts = [
        [np.median(y) + rng.normal(0, 0.2), rng.uniform(0.3, 0.95),
         rng.uniform(0, 2 * np.pi), freq0 * (1 + rng.normal(0, 5e-4))]
        for _ in range(60)
    ]
    best = _best_least_squares(residuals, lower, upper, starts)
    m_ab, b, theta0, freq = best.x
    chi_r, err = _fit_scale(best.cost, best.jac, len(t), len(best.x))
    ph_grid = np.linspace(0, 1, 2000)
    m_grid = ellipsoid_mag(ph_grid, m_ab, b, 1.0, 0.0, theta0)
    m_bright, m_faint = float(m_grid.min()), float(m_grid.max())
    return {
        "m_ab": float(m_ab),
        "b_over_a": float(b),
        "b_over_a_err": float(err[1]),
        "theta0": float(theta0),
        "freq": float(freq),
        "chi_r_ell": float(chi_r),
        "amp_ell": m_faint - m_bright,
        "m_mid_ell": 0.5 * (m_bright + m_faint),
        "p_rot_ell_h": float(48 / freq),
        "ph_grid": ph_grid,
        "m_grid": m_grid,
        "rms": float(np.std(best.fun * dy)),
    }


def _plot_spheres_fit(provid, t, y, dy, t0, spheres):
    path = output_path("spheres", provid)
    if path.exists():
        return path
    curve_color = band_colors.get(BAND, "k")
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ph_data = _rotational_phase(t, t0, spheres["freq"])
    ax.errorbar(ph_data, y, yerr=dy, fmt="o", ms=4, alpha=0.6, color=curve_color, ecolor="0.7", label="Magnitude")
    ax.plot(
        spheres["ph_grid"], spheres["m_grid"], "k", lw=2.5,
        label=fr"Touching spheres: $q$={spheres['q_sph']:.2f}, "
              fr"$\phi$={spheres['phi_sph_deg']:.0f}$^\circ$, $\Delta m$={spheres['amp_sph']:.2f}",
    )
    ax.axhline(spheres["m_mid_sph"], color="C3", ls="--", lw=1.2, label=f"Midpoint = {spheres['m_mid_sph']:.3f}")
    ax.axhline(spheres["m_bright"], color="0.5", ls=":", lw=1)
    ax.axhline(spheres["m_faint"], color="0.5", ls=":", lw=1)
    ax.set_ylabel("Magnitude")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title(
        f"Touching-spheres fit to {provid} ({PROP},{BAND}), "
        rf"$P_{{rot}}$ = {spheres['p_rot_sph_h']:.3f} h, $\chi^2_\nu$ = {spheres['chi_r_sph']:.1f}"
    )
    _draw_sphere_inset(ax, spheres["q_sph"])
    model = touching_spheres_mag(ph_data, spheres["m_sep"], spheres["q_fit"], spheres["cos2phi"], spheres["theta0"])
    axr.errorbar(ph_data, y - model, yerr=dy, fmt="o", ms=3, alpha=0.6, color=curve_color, ecolor="0.7")
    axr.axhline(0, color="k", lw=1)
    axr.set_ylabel("Data - model")
    axr.set_xlabel("Phase at the rotation period ($2P$)")
    axr.invert_yaxis()
    axr.set_xlim(0, 1)
    axr.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _draw_sphere_inset(ax, q_can):
    """Oblique view of the two fitted lobes, with radii in the ratio q."""
    alpha, beta = np.deg2rad(18), np.deg2rad(12)
    r1_s, r2_s = 1.0, q_can
    view = np.array([np.sin(alpha), np.cos(alpha) * np.cos(beta), np.sin(beta)])
    view /= np.linalg.norm(view)
    right = np.cross(np.array([0.0, 0.0, 1.0]), -view)
    right /= np.linalg.norm(right)
    up = np.cross(-view, right)

    def sphere_face(color, n=200):
        g = np.linspace(-1, 1, n)
        u, v = np.meshgrid(g, g)
        rho2 = u**2 + v**2
        nz = np.sqrt(np.clip(1 - rho2, 0, None))
        light = np.array([-0.30, 0.48, 0.82])
        light /= np.linalg.norm(light)
        shade = np.clip(u * light[0] + v * light[1] + nz * light[2], 0, 1)
        rgb = np.array(plt.matplotlib.colors.to_rgb(color))
        img = np.zeros((n, n, 4))
        intensity = (0.20 + 0.80 * shade) * (0.45 + 0.55 * nz)
        img[..., :3] = np.clip(rgb * intensity[..., None] + 0.40 * (shade ** 24)[..., None], 0, 1)
        img[..., 3] = np.clip((1 - rho2) * (n / 3.5), 0, 1)
        return img

    spheres = [
        {"center": np.array([-r2_s, 0.0, 0.0]), "radius": r1_s, "label": r"$r_1$"},
        {"center": np.array([r1_s, 0.0, 0.0]), "radius": r2_s, "label": r"$r_2$"},
    ]
    spheres.sort(key=lambda s: np.dot(s["center"], view))
    axins = ax.inset_axes([0.705, 0.025, 0.28, 0.345])
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.45")
    axins.set_facecolor("white")
    placed = []
    for s in spheres:
        p = np.array([np.dot(s["center"], right), np.dot(s["center"], up)])
        r = s["radius"]
        axins.imshow(
            sphere_face("#8a8a8a"), origin="lower", aspect="equal", interpolation="bilinear",
            extent=[p[0] - r, p[0] + r, p[1] - r, p[1] + r], zorder=2,
        )
        axins.add_patch(plt.Circle((p[0], p[1]), r, fill=False, ec="0.12", lw=0.6, zorder=3))
        placed.append((p, r, s["label"]))
    y_lab = min(p[1] - r for p, r, _ in placed) - 0.05
    for p, r, lab in placed:
        axins.text(p[0], y_lab, lab, ha="center", va="top", fontsize=8, color="0.1")
    axins.text(
        np.mean([p[0] for p, _, _ in placed]), y_lab - 0.42,
        fr"$q = r_2/r_1 = {q_can:.2f}$", ha="center", va="top", fontsize=7.5, color="0.15",
    )
    xs = np.concatenate([[p[0] - r, p[0] + r] for p, r, _ in placed])
    ys = np.concatenate([[p[1] - r, p[1] + r] for p, r, _ in placed])
    axins.set_xlim(xs.min() - 0.28, xs.max() + 0.28)
    axins.set_ylim(y_lab - 0.85, ys.max() + 0.22)
    axins.set_aspect("equal", adjustable="box")


def _plot_ellipsoid_fit(provid, t, y, dy, t0, spheres, ellipsoid):
    path = output_path("ellipsoid", provid)
    if path.exists():
        return path
    curve_color = band_colors.get(BAND, "k")
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ph_data = _rotational_phase(t, t0, ellipsoid["freq"])
    ax.errorbar(ph_data, y, yerr=dy, fmt="o", ms=4, alpha=0.6, color=curve_color, ecolor="0.7", label="Magnitude")
    ax.plot(
        ellipsoid["ph_grid"], ellipsoid["m_grid"], "k", lw=2.5,
        label=fr"Ellipsoid, equator-on $b/a={ellipsoid['b_over_a']:.2f}$, $\Delta m={ellipsoid['amp_ell']:.2f}$",
    )
    t_grid = t0 + ellipsoid["ph_grid"] * 2.0 / ellipsoid["freq"]
    m_sph = touching_spheres_mag(
        _rotational_phase(t_grid, t0, spheres["freq"]),
        spheres["m_sep"], spheres["q_fit"], spheres["cos2phi"], spheres["theta0"],
    )
    ax.plot(ellipsoid["ph_grid"], m_sph, color="0.45", ls="--", lw=1.4, label=fr"Touching spheres, $q={spheres['q_sph']:.2f}$")
    ax.axhline(ellipsoid["m_mid_ell"], color="C3", ls="--", lw=1.2, label=f"Midpoint = {ellipsoid['m_mid_ell']:.3f}")
    ax.set_ylabel("Magnitude")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title(
        f"Triaxial-ellipsoid fit to {provid} ({PROP},{BAND}), "
        rf"$P_{{rot}}$ = {ellipsoid['p_rot_ell_h']:.3f} h, "
        rf"$\chi^2_\nu$ = {ellipsoid['chi_r_ell']:.1f} (spheres {spheres['chi_r_sph']:.1f})"
    )
    model = ellipsoid_mag(ph_data, ellipsoid["m_ab"], ellipsoid["b_over_a"], 1.0, 0.0, ellipsoid["theta0"])
    axr.errorbar(ph_data, y - model, yerr=dy, fmt="o", ms=3, alpha=0.6, color=curve_color, ecolor="0.7")
    axr.axhline(0, color="k", lw=1)
    axr.set_ylabel("Data - model")
    axr.set_xlabel("Phase at the rotation period ($2P$)")
    axr.invert_yaxis()
    axr.set_xlim(0, 1)
    axr.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


SHAPE_COLUMNS = [
    "chi_r_sph", "q_sph", "q_sph_err", "phi_sph_deg", "phi_sph_err_deg",
    "amp_sph", "m_mid_sph", "m_sep", "p_rot_sph_h",
    "chi_r_ell", "b_over_a", "b_over_a_err", "amp_ell", "m_mid_ell", "m_ab", "p_rot_ell_h",
]


def shape_fit(provid, fit):
    """Touching-spheres and equator-on ellipsoid fits of the folded Lr light curve.

    Writes phase_spheres_fit and phase_ellipsoid_fit, and returns the columns
    stored in asteroid_analysis_results.csv, including chi_r_sph and chi_r_ell.
    """
    t = np.asarray(fit["t_band"], dtype=float)
    y = np.asarray(fit["y_band"], dtype=float)
    dy = np.maximum(np.asarray(fit["dy_band"], dtype=float), 1e-4)
    freq0 = float(fit["freq_sel"])
    t0 = float(np.min(t))
    spheres = _fit_touching_spheres(t, y, dy, freq0, t0)
    ellipsoid = _fit_ellipsoid(t, y, dy, freq0, t0)
    paths = [
        _plot_spheres_fit(provid, t, y, dy, t0, spheres),
        _plot_ellipsoid_fit(provid, t, y, dy, t0, spheres, ellipsoid),
    ]
    print(
        f"  shape: chi_r_sph = {spheres['chi_r_sph']:.2f}, q = {spheres['q_sph']:.3f}; "
        f"chi_r_ell = {ellipsoid['chi_r_ell']:.2f}, b/a = {ellipsoid['b_over_a']:.3f}"
    )
    row = {column: spheres[column] for column in SHAPE_COLUMNS if column in spheres}
    row.update({column: ellipsoid[column] for column in SHAPE_COLUMNS if column in ellipsoid})
    return row, paths


def _figure_paths(kind, provid):
    if kind == "shape_fit":
        return [output_path("spheres", provid), output_path("ellipsoid", provid)]
    return [output_path(kind, provid)]


def process(df_set, provid, df_rot, figures, designations):
    print(f"{provid}")
    kinds = [kind for kind in ("lightcurve", "periodogram", "phase", "phase_model", "shape_fit", "timemodel", "literature") if kind in figures]
    expected = [path for kind in kinds for path in _figure_paths(kind, provid)]
    if kinds and all(path.exists() for path in expected):
        print("  figures already exist")
        return "skip"
    df_obj = prepare_object(df_set, provid, designations)
    if df_obj.empty:
        print("  no observations")
        return None
    if df_obj[PROP].notna().sum() == 0:
        print(f"  no {PROP} values after merging the geometry table")
        return None

    written = []
    if "lightcurve" in figures:
        written.append(plot_lightcurve_h(df_obj, provid))

    fit = periodogram(df_obj, provid, save="periodogram" in figures)
    if fit is not None:
        if fit["path"] is not None:
            written.append(fit["path"])
        if "phase" in figures:
            written.append(plot_phase_allbands(df_obj, provid, fit))
        if "phase_model" in figures:
            written.append(Lr_phase_curve(provid, fit))
        if "shape_fit" in figures:
            shape, paths = shape_fit(provid, fit)
            fit["shape"] = shape
            written.extend(paths)
        if "timemodel" in figures:
            written.append(plot_timemodel_h(provid, fit))
        if "literature" in figures:
            written.append(plot_phase_model_all_periods(provid, fit, literature_periods(df_rot, provid)))
    for path in written:
        print(f"  {path.name}")
    return fit


def write_analysis_results(rows, replaced):
    """One row per asteroid in asteroid_analysis_results.csv.

    periods_h lists every period at or above 80% of the main peak, strongest
    first, separated by commas. The shape columns are the touching-spheres and
    equator-on ellipsoid fits, including chi_r_sph and chi_r_ell. Provids whose
    figures were already on disk keep the row already in the file.
    """
    columns = ["provid", "periods_h", "amp_lsm", "amp_photo", *SHAPE_COLUMNS]
    new = pd.DataFrame(rows, columns=columns)
    path = FIG_DIR / "asteroid_analysis_results.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["provid"].isin(replaced)]
        catalogue = pd.concat([old, new], ignore_index=True)
    else:
        catalogue = new
    catalogue.to_csv(path, index=False)
    print(f"{path.name} ({len(catalogue)} asteroids)")
    return path


def _selection_tables(provids, figures):
    """Photometry and observation counts for an explicit provid list."""
    check_figures(figures, "figures")

    wanted = list(dict.fromkeys(str(provid).strip() for provid in provids if str(provid).strip()))
    df_rot = load_rotation_catalog()
    df_set = load_df_set()
    present = set(df_set["provid"].dropna().astype(str))
    missing = [provid for provid in wanted if provid not in present]
    if missing:
        print(f"No observations for: {', '.join(missing)}")
    wanted = [provid for provid in wanted if provid in present]
    if not wanted:
        raise SystemExit("No selected asteroids have observations.")

    df_set = df_set[df_set["provid"].isin(wanted)].copy()
    counts = observation_counts(df_set).set_index("provid")["n_obs"]
    selected = pd.DataFrame({
        "provid": wanted,
        "n_obs": [int(counts[provid]) for provid in wanted],
    })
    print(f"{len(selected)} asteroids selected")
    return selected, df_set, df_rot, figures


def select_provid_all():
    """Asteroids with more than MIN_N_OBS observations, and the figures to write."""
    fig_dir = ROOT / "figcomp"
    set_fig_dir(fig_dir)
    # Figures for every asteroid. Remove a name to skip that plot.
    figures = {
        #"lightcurve",   # lightcurve_allnights_H: H vs time, all nights and each night
        "periodogram",  # periodogram: Lomb-Scargle power vs rotational period
        "phase",        # phase_allbands: phase curve in every band, raw and normalized
        "phase_model",  # phase_model: folded Lr photometry and the periodogram model
        "shape_fit",    # touching-spheres and ellipsoid fits
        #"timemodel",    # timemodel_H: nightly light curve with the periodogram model
        #"literature",   # phase_model_all_periods: folded curve at our period and the literature periods
        #"compare",      # period_compare_lsm and period_compare_fourier: our period vs literature, once for the sample
    }
    check_figures(figures, "figures")

    df_rot = load_rotation_catalog()
    df_set = load_df_set()
    df_obs = observation_counts(df_set)
    selected = df_obs[df_obs["n_obs"] > MIN_N_OBS].reset_index(drop=True)
    print(f"{len(selected)} asteroids with more than {MIN_N_OBS} observations")
    if selected.empty:
        raise SystemExit(f"No asteroids with more than {MIN_N_OBS} observations.")

    df_set = df_set[df_set["provid"].isin(selected["provid"])].copy()
    return selected, df_set, df_rot, figures

def select_provid_list():
    """An explicit provid list, and the figures to write."""
    fig_dir = ROOT / "figures/figures_list"
    set_fig_dir(fig_dir)
    figures = {
        "lightcurve",   # lightcurve_allnights_H: H vs time, all nights and each night
        "periodogram",  # periodogram: Lomb-Scargle power vs rotational period
        "phase",        # phase_allbands: phase curve in every band, raw and normalized
        "phase_model",  # phase_model: folded Lr photometry and the periodogram model
        "shape_fit",    # touching-spheres and ellipsoid fits
        "timemodel",    # timemodel_H: nightly light curve with the periodogram model
        #"literature",   # phase_model_all_periods: folded curve at our period and the literature periods
        #"compare",      # period_compare_lsm and period_compare_fourier: our period vs literature, once for the sample
    }
    provids = [
        "2025 MM82",
        "2025 MA19",
    ]
    return _selection_tables(provids, figures)


def select_provid_rotation():
    """Provids in asteroids-with-rotation.csv, and the figures to write."""
    fig_dir = ROOT / "figures/figures_rotation"
    set_fig_dir(fig_dir)
    figures = {
        #"lightcurve",   # lightcurve_allnights_H: H vs time, all nights and each night
        "periodogram",  # periodogram: Lomb-Scargle power vs rotational period
        "phase",        # phase_allbands: phase curve in every band, raw and normalized
        "phase_model",  # phase_model: folded Lr photometry and the periodogram model
        "shape_fit",    # touching-spheres and ellipsoid fits
        #"timemodel",    # timemodel_H: nightly light curve with the periodogram model
        "literature",   # phase_model_all_periods: folded curve at our period and the literature periods
        "compare",      # period_compare_lsm and period_compare_fourier: our period vs literature, once for the sample
    }
    df_rot = load_rotation_catalog()
    return _selection_tables(df_rot["provid"].tolist(), figures)

def select_provid_medellin():
    """Provids in medellin-vro-catalogue.csv, and the figures to write."""
    fig_dir = ROOT / "figures/figures_medellin"
    set_fig_dir(fig_dir)
    figures = {
        #"lightcurve",   # lightcurve_allnights_H: H vs time, all nights and each night
        "periodogram",  # periodogram: Lomb-Scargle power vs rotational period
        "phase",        # phase_allbands: phase curve in every band, raw and normalized
        "phase_model",  # phase_model: folded Lr photometry and the periodogram model
        "shape_fit",    # touching-spheres and ellipsoid fits
        #"timemodel",    # timemodel_H: nightly light curve with the periodogram model
        "literature",   # phase_model_all_periods: folded curve at our period and the literature periods
        #"compare",      # period_compare_lsm and period_compare_fourier: our period vs literature, once for the sample
    }
    catalogue = pd.read_csv(MEDELLIN_CATALOG)
    provids = catalogue["provid"].dropna().astype(str).str.strip().tolist()
    return _selection_tables(provids, figures)

def main():
    t_start = time.perf_counter()

    # select_provid_all: more than MIN_N_OBS observations
    # select_provid_list: the list written inside that function
    # select_provid_rotation: asteroids-with-rotation.csv
    # select_provid_medellin: medellin-vro-catalogue.csv
    # selected, df_set, df_rot, figures = select_provid_all()
    # selected, df_set, df_rot, figures = select_provid_list()
    # selected, df_set, df_rot, figures = select_provid_rotation()
    selected, df_set, df_rot, figures = select_provid_medellin()
    designations = load_mpc_designations()

    measured = {}
    peak_rows = []
    replaced = set()
    compare = "compare" in figures
    n_asteroids = len(selected)
    total_obs = int(selected["n_obs"].sum())
    t_loop = time.perf_counter()
    done_obs = 0
    for i, (provid, n_obs) in enumerate(zip(selected["provid"], selected["n_obs"]), start=1):
        print(f"N = {int(n_obs)}")
        fit = process(df_set, provid, df_rot, figures, designations)
        if fit == "skip":
            pass
        else:
            replaced.add(provid)
            if fit is not None:
                measured[provid] = fit["per_sel"]
                periods_h = ",".join(
                    f"{period:.6f}" for period in fit["df_peaks"]["rotational_period_h"]
                )
                row = {
                    "provid": provid,
                    "periods_h": periods_h,
                    "amp_lsm": fit["amp_lsm"],
                    "amp_photo": fit["amp_photo"],
                }
                row.update(fit.get("shape", {}))
                peak_rows.append(row)
        done_obs += int(n_obs)
        elapsed = time.perf_counter() - t_start
        # Scale the asteroid loop by observations; the catalog load is paid once.
        estimated_total = (t_loop - t_start) + (time.perf_counter() - t_loop) * (total_obs / done_obs)
        print(
            f"  elapsed: {format_elapsed(elapsed)} ({i}/{n_asteroids}), "
            f"estimated total: {format_elapsed(estimated_total)}"
        )

    write_analysis_results(peak_rows, replaced)

    if compare:
        print("Period comparison with the literature")
        plot_period_comparison(df_rot, measured)
        print(f"  elapsed: {format_elapsed(time.perf_counter() - t_start)}")

if __name__ == "__main__":
    main()
