"""
parsers.py
----------
Rule-based parsing functions for NJ MOD-IV parcel data.

Extracts occupancy type, foundation type, story count, and other
building characteristics from raw MOD-IV fields for use in HAZUS-
compatible flood risk assessments.

Functions are pure (no side effects, no GeoDataFrame mutations) so
they can be pickled and run in parallel worker processes.

References
----------
- NJ Real Property Appraisal Manual:
  https://www.nj.gov/treasury/taxation/pdf/lpt/realpropertyappraisal.pdf
- NHERI Occupancy Class Rulesets for MOD-IV Data
- FEMA Hazus Inventory Technical Manual (section 6.1):
  https://www.fema.gov/sites/default/files/documents/fema_hazus-6-inventory-technical-manual.pdf
- Pollack et al. (2025). Unrefined national building inventories can mislead
  risk assessments and decisions. SSRN 5575271.
- Kijewski-Correa et al. (2023). Validation of an augmented parcel approach
  for hurricane regional loss assessments. Natural Hazards Review, 24(3).
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Triangular distribution parameters (low, mode, high) for foundation height
# (FFE) by foundation type, taken from the UNSAFE framework which in turn
# draws from a FATHOM/USACE survey.
FFE_PARAMS = {
    'B': (0.0, 0.5, 1.5),   # Basement
    'C': (0.0, 1.5, 4),     # Crawl space
    'S': (0.0, 1.5, 4),     # Slab
    'I': (6.0, 9.0, 12.0),  # Pile
    'W': (6.0, 9.0, 12.0),  # Solid Wall
    'P': (6.0, 9.0, 12.0),  # Pier
}

FND_TYPES = np.array(['B', 'C', 'S', 'I', 'W', 'P'])

# Stories value used as a sentinel for split-level buildings so that the
# correct split-level DDF can be selected downstream.
SPLIT_LEVEL_SENTINEL = -0.5

# Any parsed story count above this is assumed to be square footage or other
# data accidentally matching a story-count pattern.
MAX_REASONABLE_STORIES = 500


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def is_missing(val):
    """
    Return True when a field value is effectively absent.

    Covers numpy/pandas NaN, empty strings, and the literal string 'None'
    that some assessors write into MOD-IV fields.

    Parameters
    ----------
    val : any
        Raw field value from a GeoDataFrame row.

    Returns
    -------
    bool
    """
    return pd.isna(val) or str(val).strip() in ('', 'None')


# ---------------------------------------------------------------------------
# Additional-lots parsing
# ---------------------------------------------------------------------------

def parse_additional_lots(add_lots_str):
    """
    Parse ADD_LOTS field.

    Returns tuple: (type, values) where type is 'lots', 'pins', or None.

    Parameters
    ----------
    add_lots_str : str or float
        Raw value from ADD_LOTS1 or ADD_LOTS2 column.

    Returns
    -------
    lot_type : str or None
        'lots' if comma-separated lot numbers, 'pins' if PAMS_PIN list,
        None if unparseable.
    values : list of str
        Parsed lot numbers or PAMS_PINs.
    """
    if is_missing(add_lots_str):
        return None, []

    s = str(add_lots_str).strip()

    # Pattern 1: Comma-separated lot numbers like '2,3,4' or '8,9,9.01' or
    # just a single number like '1'
    if re.match(r'^[\d.,\s]+$', s):
        lots = [x.strip() for x in s.split(',') if x.strip()]
        return 'lots', lots

    # Pattern 2: PAMS_PIN format - contains underscores
    if '_' in s:
        pins = [x.strip() for x in s.split(',') if x.strip()]
        if all('_' in p for p in pins):
            return 'pins', pins

    return None, []


def merge_additional_lots(parcels_gdf, debug_path=None):
    """
    Merge additional lots into parent parcels based on ADD_LOTS1/ADD_LOTS2.

    Only merges where valid matches are found. Leaves everything else
    unchanged.

    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        Parcel data with at least PAMS_PIN, PCL_MUN, PCLBLOCK, ADD_LOTS1,
        ADD_LOTS2, and geometry columns.
    debug_path : str, optional
        If provided, save parent and child parcel debug files to this
        directory as GeoPackages.

    Returns
    -------
    GeoDataFrame
        Parcel data with child geometries unioned into their parents and
        child rows dropped.
    """
    parcels = parcels_gdf.copy()
    pin_to_idx = dict(zip(parcels['PAMS_PIN'], parcels.index))

    # Track merges: child_idx -> parent_idx
    merges = {}

    stats = {
        'lots_parsed': 0,
        'pins_parsed': 0,
        'children_found': 0,
        'children_missing': 0,
    }

    for idx, row in parcels.iterrows():
        parent_pin = row['PAMS_PIN']
        pcl_mun = row['PCL_MUN']
        pclblock = row['PCLBLOCK']

        for col in ['ADD_LOTS1', 'ADD_LOTS2']:
            if col not in parcels.columns:
                continue

            lot_type, values = parse_additional_lots(row[col])

            if lot_type == 'lots':
                stats['lots_parsed'] += len(values)
                for lot in values:
                    child_pin = f"{pcl_mun}_{pclblock}_{lot}"
                    if child_pin in pin_to_idx and child_pin != parent_pin:
                        merges[pin_to_idx[child_pin]] = idx
                        stats['children_found'] += 1
                    elif child_pin != parent_pin:
                        stats['children_missing'] += 1

            elif lot_type == 'pins':
                stats['pins_parsed'] += len(values)
                for child_pin in values:
                    if child_pin in pin_to_idx and child_pin != parent_pin:
                        merges[pin_to_idx[child_pin]] = idx
                        stats['children_found'] += 1
                    elif child_pin != parent_pin:
                        stats['children_missing'] += 1

    if debug_path is not None:
        parents_with_merges = parcels.loc[list(set(merges.values()))].copy()
        children_merged = parcels.loc[list(merges.keys())].copy()
        children_merged['parent_pin'] = children_merged.index.map(
            {child: parcels.loc[parent, 'PAMS_PIN']
             for child, parent in merges.items()}
        )
        parents_with_merges.to_file(f"{debug_path}/debug_parent_parcels.gpkg")
        children_merged.to_file(f"{debug_path}/debug_child_parcels.gpkg")
        print(f"Saved debug files to {debug_path}")

    for child_idx, parent_idx in merges.items():
        child_geom = parcels.loc[child_idx, 'geometry']
        parent_geom = parcels.loc[parent_idx, 'geometry']
        parcels.loc[parent_idx, 'geometry'] = parent_geom.union(child_geom)

    parcels_out = parcels.drop(index=list(merges.keys()))

    print("=" * 50)
    print("ADDITIONAL LOTS MERGE SUMMARY")
    print("=" * 50)
    print(f"Lot numbers parsed:      {stats['lots_parsed']}")
    print(f"PAMS_PINs parsed:        {stats['pins_parsed']}")
    print(f"Children found & merged: {stats['children_found']}")
    print(f"Children not in data:    {stats['children_missing']}")
    print(f"Parcels before:          {len(parcels_gdf)}")
    print(f"Parcels after:           {len(parcels_out)}")
    print(f"Parcels removed:         {len(parcels_gdf) - len(parcels_out)}")
    print("=" * 50)

    return parcels_out


# ---------------------------------------------------------------------------
# Occupancy-type parsing
# ---------------------------------------------------------------------------

def get_base_occupancy(prop_class):
    """
    Map PROP_CLASS to base HAZUS occupancy type.

    This is the starting point before refining with BLDG_DESC/DWELL.
    Based on the NHERI assessment for Atlantic County with modifications
    per the NJ Tax Appraisal Manual.

    Parameters
    ----------
    prop_class : str
        Property class code from MOD-IV parcel data.

    Returns
    -------
    str or None
        HAZUS occupancy type, or None for vacant land or unrecognised codes.
    """
    pc = str(prop_class).strip()

    mapping = {
        '1':   None,       # Vacant land - no building
        '2':   'RES1',     # Residential - default single family, refine later
        '3A':  'AGR1',     # Farm (regular)
        '3B':  'AGR1',     # Farm (qualified)
        '4A':  'COM1',     # Commercial
        '4B':  'IND1',     # Industrial
        '4C':  'RES3D',    # Apartment - refined with DWELL
        '5A':  'IND1',     # Railroad Class I
        '5B':  'IND1',     # Railroad Class II
        '6A':  'COM4',     # Telephone utility
        '6B':  'IND3',     # Petroleum refinery
        '15A': 'EDU1',     # Public school
        '15B': 'EDU1',     # Other school
        '15C': 'GOV1',     # Public property
        '15D': 'REL1',     # Church/charitable
        '15E': 'REL1',     # Cemetery
        '15F': 'GOV1',     # Other exempt
    }

    return mapping.get(pc, None)


def get_occupancy_from_prop_use(prop_use):
    """
    Map PROP_USE code to HAZUS occupancy type.

    Returns None if no mapping found (caller should fall back to PROP_CLASS).
    Based on FEMA/NHERI Occupancy Class Rulesets for MOD-IV Data.

    NOTE: More specific PROP_USE codes could be added to improve DDF
    resolution for certain commercial/industrial subtypes.

    Parameters
    ----------
    prop_use : str
        Three-digit property use code from MOD-IV parcel data.

    Returns
    -------
    str or None
        HAZUS occupancy type.
    """
    PROP_USE_DIRECT = {
        '999': 'RES1',
        '512': 'RES2',
        '020': 'RES3B',
        '029': 'RES3C',
        '021': 'RES3E',
        '635': 'RES5',
        '335': 'RES5',
        '180': 'COM2',
        '562': 'COM7',
        '650': 'IND4',
        '191': 'GOV2',
        '075': 'EDU2',
        '074': 'REL1',
        '130': 'REL1',
    }

    PROP_USE_RES4  = {'280', '281', '282', '283', '530'}
    PROP_USE_RES6  = {'270', '273', '278', '636', '637'}
    PROP_USE_COM1  = {'525', '526', '527', '528', '529'}
    PROP_USE_COM3  = {'110', '210', '219', '220'}
    PROP_USE_COM4  = {'190', '441', '760', '761', '500', '561', '563',
                      '565', '566', '569'}
    PROP_USE_COM5  = {'050', '051', '059'}
    PROP_USE_COM6  = {'271', '272', '279'}
    PROP_USE_COM10 = {'211', '212', '750'}
    PROP_USE_IND3  = {'970', '940', '218', '571'}
    PROP_USE_IND6  = {'040', '440', '580'}
    PROP_USE_AGR1  = {'120', '222', '430', '740'}
    PROP_USE_GOV1  = {'221', '564', '570', '230'}
    PROP_USE_EDU1  = {'442', '660', '661'}

    if is_missing(prop_use):
        return None

    code = str(prop_use).strip().zfill(3)

    if code in PROP_USE_DIRECT:
        return PROP_USE_DIRECT[code]

    if code in PROP_USE_RES4:  return 'RES4'
    if code in PROP_USE_RES6:  return 'RES6'
    if code in PROP_USE_COM1:  return 'COM1'
    if code in PROP_USE_COM3:  return 'COM3'
    if code in PROP_USE_COM4:  return 'COM4'
    if code in PROP_USE_COM5:  return 'COM5'
    if code in PROP_USE_COM6:  return 'COM6'
    if code in PROP_USE_COM10: return 'COM10'
    if code in PROP_USE_IND3:  return 'IND3'
    if code in PROP_USE_IND6:  return 'IND6'
    if code in PROP_USE_AGR1:  return 'AGR1'
    if code in PROP_USE_GOV1:  return 'GOV1'
    if code in PROP_USE_EDU1:  return 'EDU1'

    try:
        code_int = int(code)

        # COM1: 525-529 (inclusive)
        if 525 <= code_int <= 529: return 'COM1'

        # COM3: 638-650 (inclusive)
        if 638 <= code_int <= 650: return 'COM3'

        # COM4: 729-740 (inclusive)
        if 729 <= code_int <= 740: return 'COM4'

        # COM8: several ranges
        if 609 <= code_int <= 630: return 'COM8'
        if 70  <= code_int <= 73:  return 'COM8'
        if 80  <= code_int <= 81:  return 'COM8'
        if 510 <= code_int <= 511: return 'COM8'
        if 540 <= code_int <= 541: return 'COM8'
        if 769 <= code_int <= 773: return 'COM8'

        # COM9: 768-779 (inclusive)
        if 768 <= code_int <= 779: return 'COM9'

        # IND1
        if code_int == 330:        return 'IND1'
        if 939 <= code_int <= 961: return 'IND1'

        # IND2
        if 10  <= code_int <= 31:  return 'IND2'
        if 790 <= code_int <= 791: return 'IND2'
        if 949 <= code_int <= 961: return 'IND2'

    except ValueError:
        pass

    return None


def get_occupancy_from_bldg_class(bldg_class):
    """
    Map BLDG_CLASS to HAZUS occupancy type.

    Returns None if no mapping found. BLDG_CLASS is a numeric code
    describing building quality/type.

    Townhouses (BldgClass 33-39) are classified as RES3C rather than RES1
    as in the original NHERI rules, because they share walls and their
    footprint appears as a single building — consistent with the Pollack
    et al. (2025) Philadelphia study's treatment of twin row homes.

    See NJ Real Property Appraisal Manual page 50 for R-number definitions.

    Parameters
    ----------
    bldg_class : str or float
        Building class code from MOD-IV parcel data.

    Returns
    -------
    str or None
        HAZUS occupancy type.
    """
    if is_missing(bldg_class):
        return None

    try:
        bc = int(float(bldg_class))
    except (ValueError, TypeError):
        return None

    # Single Family Dwelling: BldgClass 11-32
    if 11 <= bc <= 32:
        return 'RES1'

    # Row/Town houses: BldgClass 33-39
    # Classified as RES3C (multi-family 5-9 units) rather than RES1 because
    # they share walls. A footprint-based unit count would be more accurate
    # but is not yet implemented.
    if 33 <= bc <= 39:
        return 'RES3C'

    # Mobile Home: BldgClass 49-55
    if 49 <= bc <= 55:
        return 'RES2'

    # Multi-family 3-4 Units: BldgClass 42-50 OR BldgClass 145
    if (42 <= bc <= 50) or bc == 145:
        return 'RES3B'

    # Agriculture: BldgClass 151-165
    if 151 <= bc <= 165:
        return 'AGR1'

    return None


def refine_occupancy_from_units(current_occ, num_units, prop_class=None,
                                bldg_class=None):
    """
    Refine residential occupancy type based on number of dwelling units.

    Only applies to residential types (RES1, RES3x). Does not override:
    - 4C apartment parcels (DWELL column frequently misclassifies these
      as RES1 due to incorrect single-unit entries)
    - Townhouses (bldg_class 33-39, already corrected to RES3 in
      get_occupancy_from_bldg_class)

    Parameters
    ----------
    current_occ : str
        Current occupancy type.
    num_units : float or str
        Number of dwelling units from DWELL column.
    prop_class : str, optional
        Property class; prevents override of 4C apartment classification.
    bldg_class : str or float, optional
        Building class; prevents override of RES3 townhouse classification.

    Returns
    -------
    str
        Refined (or unchanged) occupancy type.
    """
    RESIDENTIAL_TYPES = {'RES1', 'RES3A', 'RES3B', 'RES3C',
                         'RES3D', 'RES3E', 'RES3F'}

    if str(prop_class).strip() == '4C':
        return current_occ

    if not is_missing(bldg_class):
        try:
            bc = int(float(bldg_class))
            if 33 <= bc <= 39:
                return 'RES3C'
        except (ValueError, TypeError):
            pass

    if current_occ not in RESIDENTIAL_TYPES:
        return current_occ

    if is_missing(num_units):
        return current_occ

    try:
        units = int(float(num_units))
    except (ValueError, TypeError):
        return current_occ

    if units == 1:          return 'RES1'
    elif units == 2:        return 'RES3A'
    elif units in (3, 4):   return 'RES3B'
    elif 5 <= units <= 9:   return 'RES3C'
    elif 10 <= units <= 19: return 'RES3D'
    elif 20 <= units <= 49: return 'RES3E'
    elif units >= 50:       return 'RES3F'

    return current_occ


def get_hazus_occupancy(row):
    """
    Get HAZUS occupancy type using all available MOD-IV fields.

    Priority order:
    1. PROP_CLASS  (base classification)
    2. PROP_USE    (overrides if more specific)
    3. BLDG_CLASS  (overrides if more specific)
    4. BLDG_DESC   (mobile-home check; returns RES2 immediately if matched)
    5. DWELL       (refines residential subtypes by unit count)

    Parameters
    ----------
    row : dict-like
        A row from the MOD-IV GeoDataFrame (supports .get()).

    Returns
    -------
    str or None
        Final HAZUS occupancy type.
    """
    prop_class = row.get('PROP_CLASS')
    bldg_class = row.get('BLDG_CLASS')

    occ = get_base_occupancy(prop_class)

    occ_from_use = get_occupancy_from_prop_use(row.get('PROP_USE'))
    if occ_from_use:
        occ = occ_from_use

    occ_from_bldg = get_occupancy_from_bldg_class(bldg_class)
    if occ_from_bldg:
        occ = occ_from_bldg

    if check_mobile_home(row.get('BLDG_DESC')):
        return 'RES2'

    occ = refine_occupancy_from_units(occ, row.get('DWELL'),
                                      prop_class, bldg_class)
    return occ


# ---------------------------------------------------------------------------
# Building-description parsing
# ---------------------------------------------------------------------------

def check_mobile_home(bldg_desc):
    """
    Check if BLDG_DESC indicates a mobile/manufactured home.

    Mobile homes are largely misattributed in MOD-IV as COM4 parcels
    (prop_class 4A) and as RES1 in NSI, because most manufactured homes
    sit in corporate-owned parks where the land owner and home owner differ.
    Correctly identifying mobile homes is therefore difficult, and this
    function will undercount RES2 — see the data quality section of the
    notebook.

    Mobile home indicators
    ----------------------
    - Explicit words: MOBILE, TRAILER, MFD HOME, MANUFACTURED
    - Style code M in coded format: 1SFM, 1SM, 1SALM, 1S-AL-M, etc.

    Parameters
    ----------
    bldg_desc : str
        Building description from MOD-IV BLDG_DESC column.

    Returns
    -------
    bool
    """
    if is_missing(bldg_desc):
        return False

    desc = str(bldg_desc).upper().strip()

    if re.search(r'\bMOBILE\b|\bTRAILER\b|\bMFD\s*HOME\b|\bMANUFACTURED\b',
                 desc):
        return True

    # Coded format: M as style code after stories + material
    # Catches: 1SFM, 1SM, 1SALM, 1SBM, 1SCBM, 1S-F-M, 1S-AL-M, 1S-M
    if re.match(r'^\d+\.?\d*S[FABCW]?[AL]?M(\s|$|\d|G|P|C)', desc):
        return True

    if re.match(r'^\d+\.?\d*S-[A-Z]*-?M(\s|$|-|\d)', desc):
        return True

    if re.match(r'^\d+\.?\d*SM(\s|$|-|\d|G|P|C)', desc):
        return True

    return False


def detect_basement_from_desc(bldg_desc):
    """
    Check if BLDG_DESC indicates presence of a basement.

    Returns True/False/None where None means the description gives no
    indication either way. Some assessors use ADD_LOTS overflow fields
    for this information, but parsing those is not yet implemented.

    Parameters
    ----------
    bldg_desc : str
        Building description from MOD-IV BLDG_DESC column.

    Returns
    -------
    bool or None
    """
    if is_missing(bldg_desc):
        return None

    desc = str(bldg_desc).upper().strip()

    if re.search(
        r'\bBSMT\b|\bBASEMENT\b|\bBSM\b|\bBST\b|/B\b|W/B\b|\bFULL\s*BASE',
        desc
    ):
        return True

    if re.search(r'\bSLAB\b|\bSLB\b|\bCRAWL\b|\bNO\s*BSMT\b|\bNOBSMT\b',
                 desc):
        return False

    return None


def parse_foundation(bldg_desc):
    """
    Extract foundation type code from BLDG_DESC.

    Parameters
    ----------
    bldg_desc : str
        Building description from MOD-IV BLDG_DESC column.

    Returns
    -------
    str or None
        'B' (basement), 'S' (slab), 'C' (crawl space), or None.
    """
    if is_missing(bldg_desc):
        return None

    desc = str(bldg_desc).upper().strip()

    if re.search(
        r'\bBSMT\b|\bBASEMENT\b|\bBSM\b|\bBST\b|/B\b|W/B\b|\bFULL\s*BASE',
        desc
    ):
        return 'B'

    if re.search(r'\bSLAB\b|\bSLB\b|/SL\b', desc):
        return 'S'

    if re.search(r'\bCRAWL\b|\bCRWL\b', desc):
        return 'C'

    return None


def parse_stories(bldg_desc):
    """
    Extract number of stories from BLDG_DESC.

    Patterns are tried in order of confidence, starting with entries that
    follow the official MOD-IV metadata and ending with architectural-style
    inference (e.g. colonial → 2 stories). There is likely some data loss
    and misclassification, but manual inspection of results suggests the
    overall accuracy is reasonable.

    Special return values
    ---------------------
    SPLIT_LEVEL_SENTINEL (-0.5) : split-level or bi-level building, so that
        the correct split-level DDF can be selected downstream.
    None : story count could not be determined.

    Parameters
    ----------
    bldg_desc : str
        Building description from MOD-IV BLDG_DESC column.

    Returns
    -------
    float or None
    """
    if is_missing(bldg_desc) or str(bldg_desc).strip() in ('0', '00', '000'):
        return None

    desc = str(bldg_desc).upper().strip()

    def validate(val):
        """Return val if within reasonable range, else None."""
        if val is not None and val <= MAX_REASONABLE_STORIES:
            return val
        return None

    # --- Standard MOD-IV coded patterns ---

    # "1 1/2" at start (e.g. "1 1/2 S F", "1 1/2SF")
    m = re.match(r'^(\d+)\s*1/2', desc)
    if m:
        result = validate(int(m.group(1)) + 0.5)
        if result is not None:
            return result

    # "1.5S " or "2S " or "1.5S-" (space or dash after S)
    m = re.match(r'^(\d+\.?\d*)S[\s\-/]', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1SF", "2SF", "1.5SF", "2SFG" (S followed by F for frame)
    m = re.match(r'^(\d+\.?\d*)SF', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1SB", "2SB" (S followed by B for brick)
    m = re.match(r'^(\d+\.?\d*)SB', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1SCB", "2SCB" (S followed by CB for concrete block)
    m = re.match(r'^(\d+\.?\d*)SCB', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1SS", "2SS" (S followed by S for stucco/structured steel)
    # Avoids matching SST (stone) via negative lookahead
    m = re.match(r'^(\d+\.?\d*)SS[^T]', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1SST", "2SST" (S followed by ST for stone)
    m = re.match(r'^(\d+\.?\d*)SST', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1SAL", "2SAL" (S followed by AL for aluminum)
    m = re.match(r'^(\d+\.?\d*)SAL', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1S-F-L", "2S-F-L" (dashes between components)
    m = re.match(r'^(\d+\.?\d*)S-', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "2 STORY", "2-STORY", "1 STORY"
    m = re.search(r'(\d+\.?\d*)\s*-?\s*STOR', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "2STY", "1STY"
    m = re.search(r'(\d+\.?\d*)\s*STY', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # "1 S F", "2 S B" (spaces between components)
    m = re.match(r'^(\d+\.?\d*)\s+S\s+[FBACW]', desc)
    if m:
        result = validate(float(m.group(1)))
        if result is not None:
            return result

    # --- Structural type inference ---

    if re.search(r'BI-?\s*LEVEL|BILEVEL|SPLIT\s*LEVEL|\bSPLIT\b|/S/L|S/L\b',
                 desc):
        return SPLIT_LEVEL_SENTINEL

    if re.search(r'TRI-?\s*LEVEL|TRILEVEL', desc):
        return 3.0

    # --- Architectural style inference (least certain) ---

    if re.search(r'\bRANCH\b|\bRANCHER\b', desc) and \
            not re.search(r'RAISED\s*RANCH', desc):
        return 1.0

    if re.search(r'\bCAPE\b', desc):
        return 1.5

    if re.search(r'\bCOLONIAL\b', desc):
        return 2.0

    if re.search(r'\bGEORGIAN\b', desc):
        return 2.0

    if re.search(r'\bBUNGALOW\b|\bBUNGELOW\b', desc):
        return 1.0

    if re.search(r'\bREGENCY\b', desc):
        return 2.0

    return None


def add_parsed_bldg_columns(gdf):
    """
    Apply all BLDG_DESC parsers and add result columns to a parcel GDF.

    Adds PARSED_STORIES, PARSED_HAS_BASEMENT, and PARSED_FOUNDATION.

    Parameters
    ----------
    gdf : GeoDataFrame
        MOD-IV parcel data with a BLDG_DESC column.

    Returns
    -------
    GeoDataFrame
        Copy of input with three additional columns.
    """
    gdf = gdf.copy()
    gdf['PARSED_STORIES']      = gdf['BLDG_DESC'].apply(parse_stories)
    gdf['PARSED_HAS_BASEMENT'] = gdf['BLDG_DESC'].apply(detect_basement_from_desc)
    gdf['PARSED_FOUNDATION']   = gdf['BLDG_DESC'].apply(parse_foundation)
    return gdf


# ---------------------------------------------------------------------------
# FFE Monte Carlo
# ---------------------------------------------------------------------------

def resample_ffe(df, found_type_col='found_type', n_sims=2000, seed=None):
    """
    Resample first-floor elevation (FFE) via Monte Carlo using triangular
    distributions parameterised by foundation type.

    Parameters from the UNSAFE framework (Pollack et al., 2025), which
    draws from a FATHOM/USACE survey. While the FATHOM sourcing is not
    fully transparent, it is the best available and is also used in the
    Philly risk study.

    Parameters
    ----------
    df : DataFrame
        Must contain a foundation-type column with values in FFE_PARAMS.
    found_type_col : str
        Column name for foundation type.
    n_sims : int
        Number of Monte Carlo simulations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Original df with added columns: ffe_q05, ffe_q25, ffe_q50,
        ffe_q75, ffe_q95.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    ffe_sims = np.empty((n, n_sims))

    for i in range(n_sims):
        ffes = np.zeros(n)
        for fnd_type, (left, mode, right) in FFE_PARAMS.items():
            mask = df[found_type_col] == fnd_type
            if mask.any():
                ffes[mask] = rng.triangular(left, mode, right,
                                            size=mask.sum())
        ffe_sims[:, i] = ffes

    df['ffe_q05'] = np.round(np.percentile(ffe_sims,  5, axis=1), 2)
    df['ffe_q25'] = np.round(np.percentile(ffe_sims, 25, axis=1), 2)
    df['ffe_q50'] = np.round(np.percentile(ffe_sims, 50, axis=1), 2)
    df['ffe_q75'] = np.round(np.percentile(ffe_sims, 75, axis=1), 2)
    df['ffe_q95'] = np.round(np.percentile(ffe_sims, 95, axis=1), 2)

    return df


# ---------------------------------------------------------------------------
# Final-merge resolution helpers
# ---------------------------------------------------------------------------

def resolve_occtype(parcel_occ, nsi_occ):
    """
    Choose final occupancy type, giving priority to parcel data.

    In practice, parcels without a parseable occupancy type should have
    been dropped before this step (as shown in Part 4 of the notebook).
    This function is included mainly for use in future national-scale runs.

    Parameters
    ----------
    parcel_occ : str or None
        Occupancy type parsed from parcel data.
    nsi_occ : str or None
        Occupancy type from NSI.

    Returns
    -------
    str or None
    """
    if pd.notna(parcel_occ) and parcel_occ != '':
        return parcel_occ
    return nsi_occ


def resolve_stories(parcel_stories, nsi_stories):
    """
    Choose final story count, giving priority to parcel data.

    Also normalises fractional values to standard DDF buckets:
    - (1.0, 1.5) → 1.0
    - (1.5, 2.0) → 2.0
    - >= 2.5     → 3.0

    Parameters
    ----------
    parcel_stories : float or None
    nsi_stories : float or None

    Returns
    -------
    float or None
    """
    stories = parcel_stories if pd.notna(parcel_stories) else nsi_stories
    if pd.isna(stories):
        return None
    if 1.0 < stories < 1.5:
        return 1.0
    if 1.5 < stories < 2.0:
        return 2.0
    if stories >= 2.5:
        return 3.0
    return stories


def resolve_foundation(parcel_found, nsi_found):
    """
    Choose final foundation type, giving priority to parcel data.

    Parameters
    ----------
    parcel_found : str or None
    nsi_found : str or None

    Returns
    -------
    str or None
    """
    if pd.notna(parcel_found) and parcel_found != '':
        return parcel_found
    return nsi_found


def foundation_is_basement(foundation):
    """
    Return True when the resolved foundation type indicates a basement.

    Parameters
    ----------
    foundation : str or None
        Foundation type code ('B', 'BASEMENT', 'S', 'SLAB', 'C', 'CRAWL',
        or None).

    Returns
    -------
    bool
    """
    if pd.isna(foundation):
        return False
    return str(foundation).upper() in ('B', 'BASEMENT')


# ---------------------------------------------------------------------------
# Data-quality comparison flags
# ---------------------------------------------------------------------------

def compare_occtype(parcel_occ, nsi_occ):
    """
    Return a match-quality flag for occupancy type comparison.

    Parameters
    ----------
    parcel_occ : str or None
    nsi_occ : str or None

    Returns
    -------
    str
        'MATCH', 'PARTIAL' (same 3-char category), 'MISMATCH', or 'MISSING'.
    """
    if pd.isna(parcel_occ) or pd.isna(nsi_occ):
        return 'MISSING'
    if parcel_occ == nsi_occ:
        return 'MATCH'
    if parcel_occ[:3] == nsi_occ[:3]:
        return 'PARTIAL'
    return 'MISMATCH'


def compare_stories(parcel_stories, nsi_stories):
    """
    Return a match-quality flag for story-count comparison.

    Parameters
    ----------
    parcel_stories : float or None
    nsi_stories : float or None

    Returns
    -------
    str
        'MATCH', 'CLOSE' (within 0.5 stories), 'MISMATCH', or 'MISSING'.
    """
    if pd.isna(parcel_stories) or pd.isna(nsi_stories):
        return 'MISSING'
    if parcel_stories == nsi_stories:
        return 'MATCH'
    if abs(parcel_stories - nsi_stories) <= 0.5:
        return 'CLOSE'
    return 'MISMATCH'


# ---------------------------------------------------------------------------
# DDF assignment
# ---------------------------------------------------------------------------

def get_structure_ddf_id(occtype, stories, has_basement, ddf_lookup):
    """
    Look up Structure Function ID from the HAZUS DDF lookup table.

    For RES1, story count and basement presence are used to select the
    appropriate curve. For all other occupancy types, the first matching
    row is returned (typically a single entry per type).

    Parameters
    ----------
    occtype : str or None
        HAZUS occupancy type.
    stories : float or None
        Number of stories (or SPLIT_LEVEL_SENTINEL for split-level).
    has_basement : bool
        Whether the structure has a basement foundation.
    ddf_lookup : DataFrame
        DDF lookup table with columns: occtype, Number of Stories,
        Basement, Structure Function ID.

    Returns
    -------
    str or None
        Structure Function ID, or None if no match found.
    """
    if pd.isna(occtype):
        return None

    matches = ddf_lookup[ddf_lookup['occtype'] == occtype]

    if len(matches) == 0:
        return None

    if len(matches) == 1:
        return matches['Structure Function ID'].iloc[0]

    if occtype == 'RES1':
        if pd.isna(stories):
            stories_match = '1'
        elif stories in (SPLIT_LEVEL_SENTINEL, 0.5):
            stories_match = '-0.5'
        elif stories < 1.5:
            stories_match = '1'
        elif stories == 1.5:
            stories_match = '1.5'
        elif stories < 3:
            stories_match = '2'
        else:
            stories_match = '3'

        matches = matches[matches['Number of Stories'] == stories_match]

    if len(matches) > 1:
        if has_basement:
            bsmt_matches = matches[
                matches['Basement'].str.startswith('Basement', na=False)
            ]
        else:
            bsmt_matches = matches[
                matches['Basement'].str.contains('No Basement', na=False)
            ]
        if len(bsmt_matches) > 0:
            matches = bsmt_matches

    if len(matches) > 0:
        return matches['Structure Function ID'].iloc[0]

    return None


def assign_ddf_ids(df, ddf_lookup):
    """
    Assign Structure Function ID to every row in the building inventory.

    Parameters
    ----------
    df : GeoDataFrame
        Final building inventory with final_occtype, final_stories,
        and has_basement columns.
    ddf_lookup : DataFrame
        DDF lookup table (see get_structure_ddf_id).

    Returns
    -------
    GeoDataFrame
        Copy of df with added structure_ddf_id column.
    """
    df = df.copy()
    df['structure_ddf_id'] = df.apply(
        lambda r: get_structure_ddf_id(
            r['final_occtype'],
            r['final_stories'],
            r['has_basement'],
            ddf_lookup,
        ),
        axis=1,
    )
    print(f"DDF assigned: {df['structure_ddf_id'].notna().sum()} / {len(df)}")
    return df