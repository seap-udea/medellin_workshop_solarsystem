# Medellin VRO Catalogue

## Vera Rubin Observatory Catalogue of the Photometric and Dynamical properties of asteroids

![Medellin VRO Catalogue team](figures/Medellin-VROC-team.webp)

The catalogue was built by Team 5 of the [Open Tools, Open Sky](https://plazasmalagon.com/lsst-science-public-rubin-data-products-hackathon-UdeA-2026/) workshop (Medellín, 13–15 October 2026), mentored by Pedro Bernardinelli.

- Pedro Bernardinelli, mentor
- Jorge Iván Zuluaga Callejas, Universidad de Antioquia, Colombia
- Juanita Andrea Agudelo, Universidad de Antioquia, Colombia
- Angie Melissa Gutiérrez Estupiñán, Universidad Industrial de Santander, Colombia
- Nicolás Pan, Universidad de la República, Uruguay
- Valeria Judith Burgoa Camacho, Universidad Mayor de San Andrés, Bolivia

The Medellin VRO Catalogue collects 508 asteroids measured in the public Rubin First Look data during the Medellín workshop. Each row is one asteroid. It brings together the provisional designation, the orbit at the first observation, mean absolute magnitudes and colors, a Lomb–Scargle rotation period and its amplitude, two shape fits of the folded light curve, and the physical properties reported by the JPL Small-Body Database. A missing value is NaN. Magnitudes in the rotation and shape columns are absolute magnitudes in the LSST r band.

The table is `zuluaga_dev/medellin-vroc.csv`.

```python
import pandas as pd

catalogue = pd.read_csv("zuluaga_dev/medellin-vroc.csv")
catalogue.head()
```

## Columns

### Identification

| Column | Unit | Meaning |
|---|---|---|
| `provid` | | Provisional MPC designation, `YYYY LLnn`. |
| `permid` | | Permanent number: the integer in parentheses in `mpc_designation`. NaN when the asteroid is not numbered. |
| `mpc_designation` | | MPC designation used for the geometry. Numbered objects look like `(4273) Dunhuang`; unnumbered ones keep the provisional designation. |

### Orbit

Elements at `epoch_tdb`, taken from the geometry table at the first observation.

| Column | Unit | Meaning |
|---|---|---|
| `epoch_tdb` | TDB | Epoch of the elements, equal to the time of the first observation. |
| `a` | au | Semimajor axis. |
| `e` | | Eccentricity. |
| `q` | au | Perihelion distance. |
| `incl` | deg | Inclination to the ecliptic. |
| `node` | deg | Longitude of the ascending node. |
| `peri` | deg | Argument of perihelion. |
| `M` | deg | Mean anomaly. |

### Photometry

Mean absolute magnitudes and colors.

| Column | Unit | Meaning |
|---|---|---|
| `H_u`, `H_g`, `H_r`, `H_i`, `H_z`, `H_y` | mag | Mean absolute magnitude in that LSST band. |
| `N_u`, `N_g`, `N_r`, `N_i`, `N_z`, `N_y` | | Number of observations in that mean. |
| `u-g`, `g-r`, `r-i`, `i-z`, `z-y` | mag | Color, the difference of the two absolute magnitudes. |

### Rotation

Lomb–Scargle fit of the absolute magnitude in the r band. Periods at or above 80% of the main peak are kept, strongest first.

| Column | Unit | Meaning |
|---|---|---|
| `periods_h` | h | Those rotation periods, separated by commas. |
| `period_best` | h | First value of `periods_h`. |
| `amp_lsm` | mag | Peak-to-peak amplitude of the periodogram model over one cycle. |
| `amp_photo` | mag | Width of the photometry between the 5th and 95th percentiles. |

### Shape: touching spheres

Two spheres in contact, fitted to the folded r-band light curve. `q_sph` is reported with the smaller sphere second, so it is at most 1.

| Column | Unit | Meaning |
|---|---|---|
| `q_sph` | | Radius ratio `r2/r1`. |
| `q_sph_err` | | Uncertainty of `q_sph`. |
| `phi_sph_deg` | deg | Angle between the spin axis and the line of sight. |
| `phi_sph_err_deg` | deg | Symmetric uncertainty of that angle. NaN when the fit only gives a lower limit. |
| `chi_r_sph` | | Reduced chi-squared of the fit. |
| `amp_sph` | mag | Peak-to-peak amplitude of the model. |
| `m_mid_sph` | mag | Midpoint of the model curve, halfway between its brightest and faintest magnitudes. |
| `m_sep` | mag | Magnitude when the two disks are fully separated. |
| `p_rot_sph_h` | h | Rotation period refined by the fit. |

### Shape: ellipsoid

Equator-on triaxial ellipsoid (`phi = 90°`) rotating about its short axis. The long axis is `a = 1`.

| Column | Unit | Meaning |
|---|---|---|
| `b_over_a` | | Axis ratio `b/a`. |
| `b_over_a_err` | | Uncertainty of `b/a`. |
| `amp_ell` | mag | Peak-to-peak amplitude of the model. |
| `m_mid_ell` | mag | Midpoint of the model curve. |
| `m_ab` | mag | Magnitude of the `a`–`b` face, the brightest aspect when seen equator-on. |
| `p_rot_ell_h` | h | Rotation period refined by the fit. |
| `chi_r_ell` | | Reduced chi-squared of the fit. |

### SBDB physical parameters

From the JPL Small-Body Database. The number is stored without its unit. Each quantity has an `_err` column; that uncertainty is NaN when SBDB does not give a numeric error. Taxonomic classes have no numeric error.

| Column | Unit | Meaning |
|---|---|---|
| `H_sbdb` | mag | Absolute magnitude. |
| `albedo_sbdb` | | Geometric albedo. |
| `diameter_sbdb` | km | Diameter. |
| `extent_sbdb` | km | Axis extents, kept as text `a x b` or `a x b x c`. |
| `pole_sbdb` | deg | Ecliptic pole, kept as text `longitude/latitude`. |
| `density_sbdb` | g cm⁻³ | Bulk density. |
| `rot_per_sbdb` | h | Rotation period in the literature. |
| `GM_sbdb` | km³ s⁻² | Gravitational parameter. |
| `BV_sbdb` | mag | Color B−V. |
| `UB_sbdb` | mag | Color U−B. |
| `spec_B_sbdb` | | Bus taxonomic class. |
| `spec_T_sbdb` | | Tholen taxonomic class. |

### Dynamics

Reserved for a mean-motion resonance. They are not computed yet, so every value is NaN.

| Column | Unit | Meaning |
|---|---|---|
| `res_body` | | Planet the asteroid is resonant with. |
| `res_flag` | | Whether a resonance is identified. |
| `res_type` | | Kind of resonance. |
| `res_kp` | | One integer of the resonance. |
| `res_k` | | The other integer of the resonance. |
| `res_orbit` | | Description of the resonant orbit. |

## Suggested papers

- LSST predictions for many small body populations in [Kurlander et al (2025)](https://iopscience.iop.org/article/10.3847/1538-3881/add685)
- LSST Predictions [AAS Focus issue](https://iopscience.iop.org/collections/PSJ-250528-01)
- Analysis of Rubin First Look data by [Greenstreet et al (2026)](https://doi.org/10.3847/2041-8213/ae2a30)

## Suggested software

- [Sorcha](https://github.com/dirac-institute/sorcha)
- [Layup](https://github.com/Smithsonian/layup)

## Data access

- [B612 Foundation mirror of LSST data on the MPC](https://b612.ai/rubin-mpc-downloads/)
- Individual objects in the [Minor Planet Center](https://www.minorplanetcenter.net)

## Gallery

Light curves and phase curves for the catalogue are in the [SEAP-UdeA gallery](https://seap-udea.github.io/gallery/?repo=medellin_workshop_solarsystem).
