from functions import indicator_core
import ast
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_blink_signal(signal_pct, fps,
                         detrend_window_sec, detrend_noise_factor,
                         min_detrended_range,
                         autocorr_min_peak,
                         min_on_frames, min_off_frames,
                         blink_min_hz, blink_max_hz,
                         min_cycles, max_jitter_pct):
    """
    Detrend + Autocorrelation blink detection on a 1D amber-% signal.

    Stage 1: Rolling-median baseline subtraction (removes slow drift from
             illumination changes, vehicle movement, etc.).
    Stage 2: Flatness check on the detrended signal.
    Stage 3: Autocorrelation — shape-agnostic periodic detector.
    Stage 4: Binarize detrended signal for precise edge timing.
    Stage 5: Inter-pulse spacing (secondary confirmation).
    Stage 6: Final decision: freq_ok AND autocorr_strength_ok AND cycles_ok.
    """
    
    def run_length_clean(arr, min_on, min_off):
        """
        Two-pass run-length denoiser.
        Pass 1: fill short OFF gaps  (fixes notches at the top of pulses).
        Pass 2: remove short ON pulses (removes single-frame noise spikes).
        """
        n, out = len(arr), arr.copy()
        i = 0
        while i < n:                       # Pass 1: fill OFF gaps
            if out[i] == 0:
                j = i + 1
                while j < n and out[j] == 0:
                    j += 1
                if (j - i) < min_off and i > 0 and j < n:
                    out[i:j] = 1
                i = j
            else:
                i += 1
        i = 0
        while i < n:                       # Pass 2: remove short ON pulses
            if out[i] == 1:
                j = i + 1
                while j < n and out[j] == 1:
                    j += 1
                if (j - i) < min_on:
                    out[i:j] = 0
                i = j
            else:
                i += 1
        return out

    n_sig = len(signal_pct)

    # ── Stage 1: Rolling-median detrend ───────────────────────────────────────
    window = max(5, int(fps * detrend_window_sec))
    if window % 2 == 0:
        window += 1
    baseline  = pd.Series(signal_pct).rolling(
        window=window, center=True, min_periods=1
    ).median().values
    detrended = signal_pct - baseline

    det_p5    = np.percentile(detrended, 5)
    det_p95   = np.percentile(detrended, 95)
    det_range = float(det_p95 - det_p5)
    det_std   = float(np.std(detrended))

    # Base result dict
    result = dict(
        baseline=baseline, detrended=detrended,
        det_range=det_range, det_std=det_std,
        binary=np.zeros(n_sig, dtype=int),
        cleaned=np.zeros(n_sig, dtype=int),
        rising_edges=np.array([]), falling_edges=np.array([]),
        on_durations=np.array([]), spacings=np.array([]),
        autocorr=np.array([]), autocorr_lags=(0, 0),
        best_lag=None, autocorr_strength=0.0,
        median_spacing=None, mad_sp=None, jitter_pct=None,
        detected_hz=None, is_blinking=False, reason='', n_cycles=0,
        det_threshold=0.0,
        frame_state=np.full(n_sig, 'OFF', dtype=object),
    )

    # ── Stage 2: Flatness gate ─────────────────────────────────────────────────
    if det_range < min_detrended_range:
        result['reason'] = (
            f"Flat after detrending — detrended range {det_range:.1f}% "
            f"(need >= {min_detrended_range}%)"
        )
        return result

    # ── Stage 3: Autocorrelation ───────────────────────────────────────────────
    sig_c = detrended - detrended.mean()
    norm  = float(np.sum(sig_c ** 2))
    if norm < 1e-10:
        result['reason'] = "Signal energy is zero after detrending"
        return result

    autocorr_full = np.correlate(sig_c, sig_c, mode='full')
    autocorr      = autocorr_full[len(autocorr_full) // 2:]
    autocorr      = autocorr / autocorr[0]     # Normalise to [0, 1]
    result['autocorr'] = autocorr

    min_lag = max(2, int(np.floor(fps / blink_max_hz)))
    max_lag = min(len(autocorr) - 1, int(np.ceil(fps / blink_min_hz)) + 2)
    result['autocorr_lags'] = (min_lag, max_lag)

    search_region = autocorr[min_lag:max_lag + 1]
    ac_peaks, ac_props = find_peaks(search_region, height=autocorr_min_peak)

    if len(ac_peaks) == 0:
        result['reason'] = (
            f"No periodic peak in autocorrelation "
            f"(searched lags {min_lag}–{max_lag}, "
            f"need height >= {autocorr_min_peak})"
        )
        return result

    peaks_adj   = ac_peaks + min_lag
    best_idx    = np.argmax(ac_props['peak_heights'])
    best_lag    = int(peaks_adj[best_idx])
    ac_strength = float(ac_props['peak_heights'][best_idx])
    detected_hz = float(fps / best_lag)

    result['best_lag']          = best_lag
    result['autocorr_strength'] = ac_strength
    result['detected_hz']       = detected_hz

    # ── Stage 4: Binarize detrended for edge timing ────────────────────────────
    det_threshold = det_std * detrend_noise_factor
    result['det_threshold'] = float(det_threshold)
    binary  = (detrended > det_threshold).astype(int)
    cleaned = run_length_clean(binary, min_on_frames, min_off_frames)
    result['binary']  = binary
    result['cleaned'] = cleaned

    diff          = np.diff(cleaned, prepend=0)
    rising_edges  = np.where(diff ==  1)[0]
    falling_edges = np.where(diff == -1)[0]
    result['rising_edges']  = rising_edges
    result['falling_edges'] = falling_edges

    on_durations = []
    for r in rising_edges:
        fe = falling_edges[falling_edges > r]
        on_durations.append(int(fe[0] - r) if len(fe) else int(n_sig - r))
    result['on_durations'] = np.array(on_durations)

    # ── Stage 5: Edge spacing (secondary confirmation) ─────────────────────────
    n_pulses = len(rising_edges)
    n_cycles = n_pulses - 1 if n_pulses >= 2 else 0
    result['n_cycles'] = n_cycles

    if n_pulses >= 2:
        spacings   = np.diff(rising_edges).astype(float)
        median_sp  = float(np.median(spacings))
        mad_sp     = float(np.median(np.abs(spacings - median_sp)))
        jitter_pct = (mad_sp / median_sp * 100) if median_sp > 0 else 999.0
        result['spacings']       = spacings
        result['median_spacing'] = median_sp
        result['mad_sp']         = mad_sp
        result['jitter_pct']     = jitter_pct

    # ── Stage 6: Final decision ────────────────────────────────────────────────
    freq_ok     = blink_min_hz <= detected_hz <= blink_max_hz
    strength_ok = ac_strength >= autocorr_min_peak
    cycles_ok   = n_cycles >= min_cycles
    
    jitter_ok = True
    if n_pulses >= 2 and result.get('jitter_pct') is not None:
        jitter_ok = result['jitter_pct'] <= max_jitter_pct

    is_blinking = freq_ok and strength_ok and cycles_ok and jitter_ok

    if not strength_ok:
        reason = (
            f"Weak autocorrelation: {ac_strength:.2f} "
            f"(need >= {autocorr_min_peak}) at lag {best_lag} fr"
        )
    elif not freq_ok:
        reason = (
            f"{detected_hz:.2f} Hz outside [{blink_min_hz}, {blink_max_hz}] Hz  "
            f"(autocorr peak at lag {best_lag} fr)"
        )
    elif not cycles_ok:
        reason = f"Only {n_cycles} visible cycle(s), need >= {min_cycles}"
    elif not jitter_ok:
        reason = (
            f"Irregular spacing: MAD={result['mad_sp']:.1f} fr "
            f"({result['jitter_pct']:.0f}% jitter > {max_jitter_pct}% limit)"
        )
    else:
        sp_info = ''
        if result['median_spacing'] is not None:
            sp_info = (
                f"  |  edge median={result['median_spacing']:.1f} fr, "
                f"MAD={result['mad_sp']:.1f} fr"
            )
        reason = (
            f"OK  {detected_hz:.2f} Hz  |  autocorr={ac_strength:.2f}  |  "
            f"{n_cycles} cycles  |  lag={best_lag} fr{sp_info}"
        )

    result['is_blinking'] = is_blinking
    result['reason']      = reason
    
    # ── Stage 7: Temporal Segmentation (Frame-level State) ─────────────────────
    frame_state = np.full(n_sig, 'OFF', dtype=object)
    
    if is_blinking and len(rising_edges) >= 2:
        max_gap = best_lag * 1.5
        episodes = []
        curr_start = 0
        for i in range(1, len(rising_edges)):
            if (rising_edges[i] - rising_edges[i-1]) > max_gap:
                episodes.append((curr_start, i-1))
                curr_start = i
        episodes.append((curr_start, len(rising_edges) - 1))
        
        for (start_i, end_i) in episodes:
            if end_i >= start_i:
                start_frame = rising_edges[start_i]
                fe = falling_edges[falling_edges > rising_edges[end_i]]
                end_frame = fe[0] if len(fe) > 0 else (n_sig - 1)
                frame_state[start_frame:end_frame+1] = 'BLINKING'

    result['frame_state'] = frame_state

    return result


# ── MAIN PIPELINE — Camera-level blink detection ──────────────────────────────

def detect_blinks_per_camera(
    df,
    fps,
    # Detrending
    detrend_window_sec=2.5,
    detrend_noise_factor=0.3,
    # Flatness gate
    min_detrended_range=5.0,
    # Autocorrelation
    autocorr_min_peak=0.3,
    # Run-length cleaning
    min_on_frames=2,
    min_off_frames=2,
    # Frequency validation
    blink_min_hz=0.8,
    blink_max_hz=2.5,
    min_cycles=2,
    max_jitter_pct=20.0,
):
    """
    detect_blinks_per_camera — camera-level blink detection using
    Detrend + Autocorrelation. Input is the output DataFrame of
    process_all_valid_images_with_sam (one row = one camera frame).

    Algorithm per (camera, track_id, indicator):
      Stage 1  Rolling-median baseline subtraction (removes slow drift).
      Stage 2  Flatness gate on the detrended signal.
      Stage 3  Autocorrelation — shape-agnostic periodic frequency detector.
      Stage 4  Binarize detrended signal for per-frame light_on state.
      Stage 5  Edge spacing (secondary confirmation).
      Stage 6  Final decision: freq_ok AND autocorr_strength_ok AND cycles_ok.

    Output per track:
      {col}_blink_state  in {'BLINKING', 'OFF'}

    light_on    -- True only when track is BLINKING and this frame is in the ON phase.
                   Always False when blink_state='OFF', even if amber_pct is nonzero.
    blink_state -- the track-segment verdict: BLINKING or OFF only.

    All original columns are preserved unchanged.

    Parameters
    ----------
    df                   : pd.DataFrame  — output of process_all_valid_images_with_sam
    fps                  : float         — video frame-rate (e.g. 10.0)
    detrend_window_sec   : rolling-median window in seconds (must be > 1 blink period)
    detrend_noise_factor : binarize threshold = detrended_std x this (noise floor)
    min_detrended_range  : p95-p5 of detrended must exceed this to pass flatness gate
    autocorr_min_peak    : min autocorrelation peak height to declare periodic signal
    min_on_frames        : short ON bursts below this are removed (noise)
    min_off_frames       : short OFF gaps below this are filled (waveform notches)
    blink_min_hz         : lower bound of valid blink frequency
    blink_max_hz         : upper bound of valid blink frequency
    min_cycles           : minimum complete ON->OFF cycles to declare BLINKING

    Returns
    -------
    pd.DataFrame — copy of df with:
        4 new *_blink_state columns (track-level verdict)
        light_on and blink_state fields patched into each indicator dict
    """
    
    # ── CONSTANTS ──────────────────────────────────────────────────────────────

    INDICATOR_COLS = [
        'rear_left_indicator',
        'rear_right_indicator',
        'front_left_indicator',
        'front_right_indicator',
    ]

    # ── HELPER FUNCTIONS ────────────────────────────────────────────────────────
    def patch_dict(val, **kwargs):
        """Inject key=value pairs into an indicator dict-string. Returns updated string."""
        s = str(val).strip() if val is not None else ''
        if s in ('', 'None', 'nan'):
            return val
        try:
            d = ast.literal_eval(s)
            d.update(kwargs)
            return str(d)
        except Exception:
            return val
    
    def parse_amber_pct(val):
        """Safely extract amber_pct from an indicator dict-string. Returns 0.0 on failure."""
        if val is None:
            return 0.0
        try:
            if pd.isna(val):
                return 0.0
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        if s in ('', 'None', 'nan'):
            return 0.0
        try:
            return float(ast.literal_eval(s).get('amber_pct', 0.0))
        except Exception:
            return 0.0
    
    
    df = df.copy()

    # Pre-create blink state columns (preserves order for all 4 even if col missing)
    for col in INDICATOR_COLS:
        blink_col = f'{col}_blink_state'
        if blink_col not in df.columns:
            df[blink_col] = pd.NA

    for (cam, track_id), grp in df.groupby(['camera', 'track_id'], sort=False):
        grp_sorted = grp.sort_values('timestamp')
        idx_sorted = grp_sorted.index   # original df row indices, sorted by time

        for col in INDICATOR_COLS:
            if col not in df.columns:
                continue

            # Build amber_pct signal for this (camera, track, indicator)
            signal = np.array([
                parse_amber_pct(df.at[i, col]) for i in idx_sorted
            ], dtype=float)

            res = detect_blink_signal(
                signal, fps,
                detrend_window_sec, detrend_noise_factor,
                min_detrended_range,
                autocorr_min_peak,
                min_on_frames, min_off_frames,
                blink_min_hz, blink_max_hz,
                min_cycles,
                max_jitter_pct,
            )

            # ── Frame-level State ──────────────────────────────────────────────────
            # Apply dynamic frame-wise blink states computed in Stage 7
            frame_states = res['frame_state']
            
            # Set the column dynamically per-frame
            df.loc[idx_sorted, f'{col}_blink_state'] = frame_states

            # ── Per-frame update ───────────────────────────────────────────────────
            for frame_i, orig_idx in enumerate(idx_sorted):
                f_state = frame_states[frame_i]
                frame_light_on = bool(res['cleaned'][frame_i]) if f_state == 'BLINKING' else False
                
                df.at[orig_idx, col] = patch_dict(
                    df.at[orig_idx, col],
                    light_on=frame_light_on,
                    blink_state=f_state,
                )

    return df


def fuse_to_global(df_camera):
    """
    Fuses camera-level rows (output of detect_blinks_per_camera) into one
    global row per (timestamp, track_id), collapsing all camera views.

    Fusion rules
    ------------
    left_indicator   : highest-priority state across rear_left + front_left cameras
    right_indicator  : highest-priority state across rear_right + front_right cameras
    Priority         : BLINKING > OFF
    LiDAR columns   : first non-null value (object-level, camera-independent)
    object_type      : mode (most frequent) across cameras for that timestamp+track

    Two datasets are produced by this pipeline:
      df_camera = detect_blinks_per_camera(...)  -> camera-level, all columns preserved
      df_global = fuse_to_global(df_camera)      -> timestamp-level, merged indicators

    Parameters
    ----------
    df_camera : pd.DataFrame — output of detect_blinks_per_camera

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, track_id, object_type,
        left_indicator, right_indicator,
        [available LiDAR columns: x_3d, y_3d, z_3d, yaw_3d, width_3d, length_3d]
    """

    def highest_priority_state(states):
        """
        Returns the highest-priority state from a list.
        Priority: BLINKING > OFF
        """

        STATE_PRIORITY = {'BLINKING': 1, 'OFF': 0}
        best = 'OFF'
        for s in states:
            if isinstance(s, str) and STATE_PRIORITY.get(s, -1) > STATE_PRIORITY[best]:
                best = s
        return best

    # LiDAR / 3D object columns (camera-independent, taken from first non-null camera)
    LIDAR_COLS = ['x_3d', 'y_3d', 'z_3d', 'yaw_3d', 'width_3d', 'length_3d']
    
    rows = []

    for (ts, track_id), grp in df_camera.groupby(['timestamp', 'track_id'], sort=False):

        # ── Left / Right indicator fusion ─────────────────────────────────────
        def collect(cols):
            states = []
            for col in cols:
                blink_col = f'{col}_blink_state'
                if blink_col in grp.columns:
                    states += grp[blink_col].dropna().tolist()
            return states

        left_indicator  = highest_priority_state(
            collect(['rear_left_indicator', 'front_left_indicator'])
        )
        right_indicator = highest_priority_state(
            collect(['rear_right_indicator', 'front_right_indicator'])
        )

        # ── Object type ───────────────────────────────────────────────────────
        if 'object_type' in grp.columns:
            mode_vals   = grp['object_type'].dropna()
            object_type = mode_vals.mode().iloc[0] if len(mode_vals) else None
        else:
            object_type = None

        row = {
            'timestamp':       ts,
            'track_id':        track_id,
            'object_type':     object_type,
            'left_indicator':  left_indicator,
            'right_indicator': right_indicator,
        }

        # ── LiDAR / 3D columns (first non-null value across cameras) ──────────
        for lcol in LIDAR_COLS:
            if lcol in grp.columns:
                valid    = grp[lcol].dropna()
                row[lcol] = valid.iloc[0] if len(valid) else None

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    global_df = pd.DataFrame(rows)

    # Build final column order: core first, then available LiDAR cols
    core_cols     = ['timestamp', 'track_id', 'object_type',
                     'left_indicator', 'right_indicator']
    lidar_present = [c for c in LIDAR_COLS if c in global_df.columns]

    return (
        global_df[core_cols + lidar_present]
        .sort_values(['timestamp', 'track_id'])
        .reset_index(drop=True)
    )
