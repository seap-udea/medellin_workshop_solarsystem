import textwrap
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import gzip
import os
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import pandas as pd

DEG = np.pi / 180.0
RAD = 1.0 / DEG

wavelengths = {'Lg': 483, 'Lr': 622, 'Li': 755, 'Lz': 869, 'Ly': 971}
solar_color = {'Lg': 0.45, 'Lr': 0.00, 'Li': -0.12, 'Lz': -0.16, 'Ly': -0.18}
band_colors = {'Lu': '#0000ff', 'Lg': '#00bfbf', 'Lr': '#008000', 'Li': '#bfbf00', 'Lz': '#ff0000', 'Ly': '#bf00bf', 'Lk': '#808080',
               'u': '#5252ff',  'g': '#12ffff',  'r': '#00d200',  'i': '#ffff12',  'z': '#ff5252',  'y': '#ff12ff'}
order = ['Lg', 'Lr', 'Li', 'Lz', 'Ly']
short = {'Lg': 'g', 'Lr': 'r', 'Li': 'i', 'Lz': 'z', 'Ly': 'y'}
bands = wavelengths.keys()
all_bands = ['Lu', 'Lg', 'Lr', 'Li', 'Lz', 'Ly', 'Lk', 'u', 'g', 'r', 'i', 'z', 'y']

WAVELENGTHS = {'Lg': 483, 'Lr': 622, 'Li': 755, 'Lz': 869, 'Ly': 971}
SOLAR_COLOR = {'Lg': 0.45, 'Lr': 0.00, 'Li': -0.12, 'Lz': -0.16, 'Ly': -0.18}
BAND_COLORS = {'Lu': '#0000ff', 'Lg': '#00bfbf', 'Lr': '#008000', 'Li': '#bfbf00', 'Lz': '#ff0000', 'Ly': '#bf00bf', 'Lk': '#808080',
               'u': '#5252ff',  'g': '#12ffff',  'r': '#00d200',  'i': '#ffff12',  'z': '#ff5252',  'y': '#ff12ff'}
ORDER = ['Lg', 'Lr', 'Li', 'Lz', 'Ly']
SHORT = {'Lg': 'g', 'Lr': 'r', 'Li': 'i', 'Lz': 'z', 'Ly': 'y'}
BANDS = WAVELENGTHS.keys()
ALL_BANDS = ['Lu', 'Lg', 'Lr', 'Li', 'Lz', 'Ly', 'Lk', 'u', 'g', 'r', 'i', 'z', 'y']

def print_list(lst, width=80):
    print(textwrap.fill(", ".join(lst), width=width))

# Convert longitudes [0, 360) to Mollweide angles: 0° at center, East (0° -> 180°) left, West (360° -> 180°) right
f2m = lambda lon: -((lon + 180.0) % 360.0 - 180.0)
def split_mollweide(ra_deg, dec_deg):
    """Convert RA and Dec to Mollweide radians and split lines across the 180° boundary."""
    x = f2m(ra_deg) * DEG
    y = np.deg2rad(dec_deg)
    jumps = np.where(np.abs(np.diff(x)) > np.pi)[0]
    return np.insert(x, jumps + 1, np.nan), np.insert(y, jumps + 1, np.nan)

def ecl_gal(npoints=1000):
    # Ecliptic
    lon_ecl = np.linspace(0, 360, npoints) * u.deg
    ecl = SkyCoord(lon=lon_ecl, lat=0 * u.deg, frame='geocentrictrueecliptic').icrs
    idx_ecl = np.argsort(ecl.ra.deg)
    ecl_ra = ecl.ra.deg[idx_ecl]
    ecl_dec = ecl.dec.deg[idx_ecl]
    # Galactic 
    l_gal = np.linspace(0, 360, npoints) * u.deg
    gal = SkyCoord(l=l_gal, b=0 * u.deg, frame='galactic').icrs
    idx_gal = np.argsort(gal.ra.deg)
    gal_ra = gal.ra.deg[idx_gal]
    gal_dec = gal.dec.deg[idx_gal]

    return [[ecl_ra, ecl_dec], [gal_ra, gal_dec]]

def get_data(data_dir='data/'):
    """
    Downloads the data files specified in get_data.sh.
    Creates the 'data/' directory if it does not exist and downloads the required files using wget or gdown.
    """
    import os
    import subprocess

    files = [
        {
            "filename": data_dir + "obs_sbn_X05_full.parquet",
            "cmd": f'wget -P {data_dir} https://storage.googleapis.com/asteroid-institute-public/production/rubin/mpc/obs_sbn/full/parquet/obs_sbn_X05_full.parquet'
        },  
        {
            "filename": data_dir + "mpcorb_extended.dat.gz",
            "cmd": f'wget -P {data_dir} https://minorplanetcenter.net/Extended_Files/mpcorb_extended.dat.gz'
        },
        {
            "filename": data_dir + "asteroids-with-rotation.csv",
            "cmd": f'gdown -O {data_dir}asteroids-with-rotation.csv 1afQHJYa6bAyIhbP1_dwBoyY0YYzMI7y5'
        },
        {
            "filename": data_dir + "X05_simplified.csv",
            "cmd": f'gdown -O {data_dir}X05_simplified.csv 1bre3OmMF1xvr3puWUfhhUq_OwRXFltS5'
        },
        {
            "filename": data_dir + "obs_final_elements_geometry_H.parquet",
            "cmd": f'gdown -O {data_dir}obs_final_elements_geometry_H.parquet 1jvuymsOJ8gKjKQ5fMruuqP_K_FNAvMxs'
        },
        {
            "filename": data_dir + "sbdb_physical_properties.csv",
            "cmd": f'gdown -O {data_dir}sbdb_physical_properties.csv 1bai9ysjkai3nPdRDgjGxxooBFfqOho0D'
        },
        {
            "filename": data_dir + "mpc_designations.csv",
            "cmd": f'gdown -O {data_dir}mpc_designations.csv 1ye9aftlBReq9HvJX0RzGWz2gB-G4pQkO'
        }
    ]

    os.makedirs(data_dir, exist_ok=True)
    for file in files:
        dest = file["filename"]
        # Older gdown calls saved these next to the notebook, not in data/.
        local = os.path.basename(dest)
        if not os.path.exists(dest) and os.path.exists(local):
            os.replace(local, dest)
        if os.path.exists(dest):
            print(f"Data already exists: {dest}")
        else:
            print(f"Downloading {dest}...")
            retval = os.system(file["cmd"])
            if retval != 0:
                raise RuntimeError(f"Error while downloading: {file['cmd']}")

    print("Download complete or files already present.")

import sys

def total_sizeof_globals(local_vars=None):
    """Return the total size in bytes of all variables in globals() or local_vars (if provided)"""
    seen_ids = set()
    total = 0

    def sizeof(obj):
        if id(obj) in seen_ids:
            return 0
        seen_ids.add(id(obj))
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(sizeof(k) + sizeof(v) for k,v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(sizeof(i) for i in obj)
        return size

    if local_vars is None:
        import __main__
        local_vars = __main__.__dict__
    for k, v in local_vars.items():
        if k.startswith("__") and k.endswith("__"):
            continue
        try:
            total += sizeof(v)
        except Exception:
            pass
    ram_bytes = total
    ram_mb = ram_bytes / (1024*1024)
    print(f"Total RAM used by notebook variables: {ram_mb:.2f} MB")

def orbit_designations(line):
    """Provisional designations written in readable form on one MPCORB line."""
    desigs = []
    readable = line[166:194].strip()
    if ')' in readable:
        readable = readable.split(')', 1)[1].strip()
    if readable:
        desigs.append(readable)
    tail = line[217:]
    for i in range(0, len(tail), 11):
        token = tail[i:i + 11].strip()
        if token:
            desigs.append(token)
    return desigs

def unpack_mpc_number(field):
    """Permanent number from the packed designation in columns 1-7."""
    token = field.strip()
    if token.isdigit():
        return int(token)
    if len(token) == 5 and token[1:].isdigit():
        c = token[0]
        if 'A' <= c <= 'Z':
            high = ord(c) - ord('A') + 10
        elif 'a' <= c <= 'z':
            high = ord(c) - ord('a') + 36
        else:
            return None
        return high * 10000 + int(token[1:])
    return None

def extract_orbital_elements_from_mpcorb(provids, data_dir='data/'):
    """
    Returns a DataFrame with all orbital elements for the given provids,
    reading the file 'mpcorb_extended.dat.gz' in the specified directory.
    """
    import pandas as pd
    import re

    mpcorb_file = data_dir + 'mpcorb_extended.dat.gz'
    if not os.path.exists(mpcorb_file):
        raise FileNotFoundError(f"File not found: {mpcorb_file}")

    provids_set = set(map(str, provids))

    rows = []
    matched = set()
    with gzip.open(mpcorb_file, 'rt') as f:
        in_data = False
        for line in f:
            if not in_data:
                if line.startswith('----'):
                    in_data = True
                continue
            if len(line) < 103:
                continue
            hits = [d for d in dict.fromkeys(orbit_designations(line)) if d in provids_set]
            if not hits:
                continue
            number = unpack_mpc_number(line[:7])
            readable = line[166:194].strip()
            name = readable.split(')', 1)[1].strip() if readable.startswith('(') and ')' in readable else ''
            if re.match(r'^\d{4} [A-Z]', name):
                name = ''
            record = {
                'number': number,
                'name': name,
                'H': line[8:13].strip(),
                'G': line[14:19].strip(),
                'epoch': line[20:25].strip(),
                'epoch_jd': line[203:217].strip(),
                'M': line[26:35].strip(),
                'argperi': line[37:46].strip(),
                'node': line[48:57].strip(),
                'i': line[59:68].strip(),
                'e': line[70:79].strip(),
                'n': line[80:91].strip(),
                'a': line[92:103].strip(),
                'nobs': line[117:122].strip(),
                'nopp': line[123:126].strip(),
                'arc': line[127:136].strip(),
                'rms': line[137:141].strip(),
            }
            for provid in hits:
                if provid in matched:
                    continue
                matched.add(provid)
                rows.append({'provid': provid, **record})

    df_orbital = pd.DataFrame(rows)
    for col in ['number', 'epoch_jd', 'a', 'e', 'i', 'node', 'argperi', 'M', 'H', 'G', 'nobs']:
        df_orbital[col] = pd.to_numeric(df_orbital[col], errors='coerce')
    df_orbital['number'] = df_orbital['number'].astype('Int64')
    df_orbital = df_orbital.sort_values('provid').reset_index(drop=True)
    return df_orbital

def wmean(m, s):
    m = np.asarray(m, float)
    s = np.asarray(s, float)
    w = 1.0 / s**2
    return np.sum(w * m) / np.sum(w), 1.0 / np.sqrt(np.sum(w))

def to_jd(obstime):
    """Julian date. obstime may be a pandas Timestamp, a datetime, or an ISO string."""
    if hasattr(obstime, 'to_pydatetime'):
        obstime = obstime.to_pydatetime()
    return Time(obstime, scale='utc').jd

def mag_abs(mag, alpha, dist, r, G=0.15):
    """
    Calculate the absolute magnitude H using the asteroid H-G model.

    Parameters
    ----------
    mag : float or array-like
        Apparent magnitude.
    alpha : float or array-like
        Phase angle in degrees.
    dist : float or array-like
        Observer-object distance Delta in au.
    r : float or array-like
        Heliocentric distance in au.
    G : float
        Asteroid slope parameter (default 0.15).

    Returns
    -------
    H : float or array-like
        Absolute magnitude.
    """
    alpha = np.radians(alpha)

    phi1 = np.exp(-3.33 * np.tan(alpha / 2)**0.63)
    phi2 = np.exp(-1.87 * np.tan(alpha / 2)**1.22)

    phase = (1 - G) * phi1 + G * phi2

    H = mag - 5 * np.log10(r * dist) + 2.5 * np.log10(phase)

    return H

def query_dists_and_elements(nombre_objeto, obstimes, batch_size=50):
    """
    Geocentric distance, phase angle, heliocentric distance and heliocentric
    elements for many epochs of one object.

    Horizons accepts at most 50 discrete epochs per request, so the epochs are
    split into batches. Each batch requests the ephemeris (geocenter) and the
    heliocentric elements.

    Returns
    -------
    dist, alpha, r, a, e, i, omega, Omega, M : ndarray
        One value per epoch, in the same order as ``obstimes``.
    """
    epochs = [to_jd(t) for t in obstimes]
    n = len(epochs)
    dist = np.empty(n)
    alpha = np.empty(n)
    r = np.empty(n)
    a = np.empty(n)
    e = np.empty(n)
    inc = np.empty(n)
    omega = np.empty(n)
    Omega = np.empty(n)
    M = np.empty(n)

    n_batches = int(np.ceil(n / batch_size)) if n else 0
    for b, start in enumerate(range(0, n, batch_size), start=1):
        chunk = epochs[start:start + batch_size]
        stop = start + len(chunk)
        print(f"Horizons batch {b}/{n_batches}: {len(chunk)} epochs of {nombre_objeto}")

        eph = Horizons(id=nombre_objeto, location='500', epochs=chunk).ephemerides()
        elems = Horizons(id=nombre_objeto, location='@sun', epochs=chunk).elements()
        if len(eph) != len(chunk) or len(elems) != len(chunk):
            raise RuntimeError(
                f"Horizons returned {len(eph)} ephemeris rows and {len(elems)} "
                f"element rows for {len(chunk)} epochs."
            )

        dist[start:stop] = np.asarray(eph['delta'], dtype=float)
        alpha[start:stop] = np.asarray(eph['alpha'], dtype=float)
        r[start:stop] = np.asarray(eph['r'], dtype=float)
        a[start:stop] = np.asarray(elems['a'], dtype=float)
        e[start:stop] = np.asarray(elems['e'], dtype=float)
        inc[start:stop] = np.asarray(elems['incl'], dtype=float)
        omega[start:stop] = np.asarray(elems['w'], dtype=float)
        Omega[start:stop] = np.asarray(elems['Omega'], dtype=float)
        M[start:stop] = np.asarray(elems['M'], dtype=float)

    return dist, alpha, r, a, e, inc, omega, Omega, M

import re

def geometry_designations(datafile):
    """Map an MPC number, and a provisional designation, to mpc_designation.

    Numbered asteroids are stored as '(1232) Cortusa', not as '1931 TF2'.
    Unnumbered ones keep the provisional designation.
    """
    designations = pd.read_parquet(
        datafile,
        columns=["mpc_designation"],
    )["mpc_designation"].dropna().astype(str).unique()
    by_number, by_provid = {}, {}
    numbered = re.compile(r"^\((\d+)\)")
    for name in designations:
        match = numbered.match(name)
        if match:
            by_number[match.group(1)] = name
        else:
            by_provid[name] = name
    return by_number, by_provid


def mpc_designation_for(df_set, provid, datafile, permid=None):
    """Geometry-file designation for this asteroid, or None."""
    by_number, by_provid = geometry_designations(datafile)
    if permid is None:
        ids = df_set.loc[df_set["provid"] == provid, "permid"].dropna()
        if len(ids):
            permid = ids.iloc[0]
    if pd.notna(permid) and str(permid).strip() != "":
        number = str(int(float(permid)))
        if number in by_number:
            return by_number[number]
    return by_provid.get(provid)
