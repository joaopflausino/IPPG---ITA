import streamlit as st
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, List
import tempfile
import os

# ======== PAGE CONFIG ========
st.set_page_config(
    page_title="iPPG Video Preprocessor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======== CONSTANTS ========
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
HAAR_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv.CascadeClassifier(HAAR_PATH)

EMA_ALPHA = 0.25
FOREHEAD_Y_TOP = 0.15
FOREHEAD_H = 0.18
FOREHEAD_W = 0.50
FOREHEAD_X_PAD = 0.25

HEART_BAND = (0.7, 4.0)

# ======== UTILITY FUNCTIONS ========
def safe_rect(x, y, w, h, H, W):
    """Clamp rectangle to image boundaries."""
    x = int(max(0, min(x, W-1)))
    y = int(max(0, min(y, H-1)))
    w = int(max(1, min(w, W-x)))
    h = int(max(1, min(h, H-y)))
    return x, y, w, h

def smooth_box(prev, curr, alpha=EMA_ALPHA):
    """Smooth face bounding box using EMA."""
    if prev is None:
        return curr
    return tuple(int(alpha*p + (1-alpha)*c) for p, c in zip(prev, curr))

def forehead_roi(face_rect, H, W):
    """Calculate forehead ROI within face bounding box."""
    x, y, w, h = face_rect
    fx = x + int(FOREHEAD_X_PAD * w)
    fy = y + int(FOREHEAD_Y_TOP * h)
    fw = int(FOREHEAD_W * w)
    fh = int(FOREHEAD_H * h)
    return safe_rect(fx, fy, fw, fh, H, W)

def frame_rgb_means(frame, roi):
    """Extract RGB means and luminance from ROI."""
    x, y, w, h = roi
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return (np.nan,)*5

    B = patch[:,:,0].astype(np.float32)
    G = patch[:,:,1].astype(np.float32)
    R = patch[:,:,2].astype(np.float32)

    Rm, Gm, Bm = float(R.mean()), float(G.mean()), float(B.mean())
    Y = 0.2126*R + 0.7152*G + 0.0722*B
    return Rm, Gm, Bm, float(Y.mean()), float(Y.std(ddof=0))

def bandpass(sig, fs, f_lo=0.7, f_hi=4.0, order=3):
    """Bandpass filter for cardiac frequencies."""
    if fs <= 2*f_hi:
        s = pd.Series(sig)
        return (s - s.rolling(max(3,int(fs)), min_periods=1, center=True)
                  .median().bfill().ffill()).values

    b, a = butter(order, [f_lo/(fs/2.0), f_hi/(fs/2.0)], btype="band")
    return filtfilt(b, a, sig)

def snr_green_db(g_trace, fs, band=(0.7,4.0)):
    """Calculate SNR of green channel in dB."""
    g = np.asarray(g_trace, dtype=np.float64)
    if len(g) < max(64, int(5*fs)):
        return np.nan

    s = pd.Series(g)
    g = (g - s.rolling(int(min(max(3,fs), len(g)//5)), min_periods=1, center=True)
                .median().bfill().ffill()).values
    g_bp = bandpass(g, fs, band[0], band[1])

    if np.allclose(g_bp.std(), 0):
        return np.nan

    f, Pxx = welch(g_bp, fs=fs, nperseg=min(256, len(g_bp)))
    sel = (f>=band[0]) & (f<=band[1])
    if sel.sum() < 3:
        return np.nan

    fb, Pb = f[sel], Pxx[sel]
    i = int(np.argmax(Pb))
    peak = float(Pb[i])
    fpk = float(fb[i])

    noise_vals = Pb[(fb<fpk-0.1)|(fb>fpk+0.1)]
    if noise_vals.size < 1 or peak <= 0:
        return np.nan

    noise = float(np.median(noise_vals))
    return np.nan if noise <= 0 else 10*np.log10(peak/noise)

def process_video(video_path: Path, progress_callback=None) -> Tuple[pd.DataFrame, dict]:
    """
    Process a single video and extract RGB signals.

    Returns:
        - DataFrame with frame-by-frame data
        - Dictionary with quality metrics
    """
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    prev_box = None
    frame_idx = 0
    rows = []
    g_trace = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            flags=cv.CASCADE_SCALE_IMAGE,
            minSize=(int(0.15*W), int(0.15*H))
        )

        if len(faces) > 0:
            i = int(np.argmax([w*h for (x,y,w,h) in faces]))
            box = tuple(int(v) for v in faces[i])
        else:
            box = prev_box if prev_box is not None else None

        if box is None:
            Rm = Gm = Bm = Ymean = Ystd = np.nan
        else:
            box = smooth_box(prev_box, box, EMA_ALPHA)
            prev_box = box
            roi = forehead_roi(box, H, W)
            Rm, Gm, Bm, Ymean, Ystd = frame_rgb_means(frame, roi)

        t_sec = frame_idx / float(fps)
        rows.append({
            "frame_idx": frame_idx,
            "t_sec": t_sec,
            "R_mean": Rm,
            "G_mean": Gm,
            "B_mean": Bm,
            "Y_mean": Ymean,
            "Y_std": Ystd,
            "fps": fps
        })

        if not np.isnan(Gm):
            g_trace.append(Gm)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)

    duration_s = df["t_sec"].max()
    quality_metrics = {
        "n_frames": len(df),
        "fps": float(fps),
        "duration_s": float(duration_s),
        "Y_mean_avg": float(df["Y_mean"].mean()),
        "Y_mean_std_over_time": float(df["Y_mean"].std(ddof=0)),
        "snr_green_db": float(snr_green_db(g_trace, fps, HEART_BAND))
    }

    return df, quality_metrics

# ======== STREAMLIT APP ========
def main():
    st.title("iPPG Video Preprocessor")
    st.markdown("""
    **Aprimoramento de Vídeos para Processamento iPPG (GREEN, CHROM e POS)**

    This tool processes videos to extract RGB signals from the forehead region and calculates quality metrics
    for imaging photoplethysmography (iPPG) analysis.
    """)

    st.sidebar.header("Configuration")

    processing_mode = st.sidebar.radio(
        "Processing Mode",
        ["Single Video", "Batch Processing (FaceForensics)"],
        help="Choose whether to process a single video or batch process from FaceForensics dataset"
    )

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'quality_metrics' not in st.session_state:
        st.session_state.quality_metrics = None

    # ======== SINGLE VIDEO MODE ========
    if processing_mode == "Single Video":
        st.header("Single Video Processing")

        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video containing a face for iPPG signal extraction"
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = Path(tmp_file.name)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.video(str(tmp_path))

            with col2:
                st.info(f"""
                **File Info:**
                - Name: {uploaded_file.name}
                - Size: {uploaded_file.size / 1024 / 1024:.2f} MB
                """)

                if st.button("Process Video", type="primary", use_container_width=True):
                    with st.spinner("Processing video..."):
                        progress_bar = st.progress(0)

                        def update_progress(value):
                            progress_bar.progress(value)

                        try:
                            df, metrics = process_video(tmp_path, update_progress)
                            st.session_state.processed_data = df
                            st.session_state.quality_metrics = metrics
                            st.success("✅ Video processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing video: {str(e)}")
                        finally:
                            os.unlink(tmp_path)

            if st.session_state.processed_data is not None:
                display_results(st.session_state.processed_data, st.session_state.quality_metrics)

    # ======== BATCH PROCESSING MODE ========
    else:
        st.header("Batch Processing (FaceForensics)")

        st.warning("This mode requires the FaceForensics++ dataset to be available locally.")

        dataset_path = st.text_input(
            "Dataset Root Path",
            placeholder="/path/to/FaceForensics/original_sequences/youtube",
            help="Path to the FaceForensics original_sequences/youtube folder"
        )

        compressions = st.multiselect(
            "Compression Types",
            ["raw", "c23", "c40"],
            default=["raw", "c23"],
            help="Select which compression types to process"
        )

        output_dir = st.text_input(
            "Output Directory",
            value="./rgb_signals",
            help="Directory to save processed CSV files"
        )

        if st.button("Start Batch Processing", type="primary"):
            if not dataset_path:
                st.error("Please provide the dataset path")
            else:
                st.info("Batch processing will be implemented here. This requires the full FaceForensics dataset.")
                st.markdown("""
                **Next steps:**
                1. Enumerate videos from the dataset
                2. Process each video with progress tracking
                3. Generate master_rgb.csv and quality_summary.csv
                4. Save individual video CSVs
                """)

def display_results(df: pd.DataFrame, metrics: dict):
    """Display processing results with visualizations."""
    st.divider()
    st.header("Results")

    st.subheader("Quality Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Duration", f"{metrics['duration_s']:.2f}s")
        st.metric("FPS", f"{metrics['fps']:.1f}")

    with col2:
        st.metric("Total Frames", f"{metrics['n_frames']:,}")
        st.metric("Avg Brightness", f"{metrics['Y_mean_avg']:.2f}")

    with col3:
        brightness_stability = "Stable" if metrics['Y_mean_std_over_time'] <= 12 else "Unstable"
        st.metric("Brightness Variation", f"{metrics['Y_mean_std_over_time']:.2f}")
        st.caption(brightness_stability)

    with col4:
        snr_quality = "Good" if metrics['snr_green_db'] >= 3 else "Poor"
        st.metric("SNR (Green)", f"{metrics['snr_green_db']:.2f} dB")
        st.caption(snr_quality)

    st.divider()
    quality_score = assess_quality(metrics)
    st.subheader(f"Overall Quality: {quality_score}")

    st.divider()
    st.subheader("📈 Signal Visualization")

    tab1, tab2, tab3, tab4 = st.tabs(["RGB Channels", "Luminance", "Green Channel Analysis", "Data Table"])

    with tab1:
        plot_rgb_channels(df)

    with tab2:
        plot_luminance(df)

    with tab3:
        plot_green_analysis(df, metrics['fps'])

    with tab4:
        st.dataframe(df, use_container_width=True, height=400)

        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Signal Data (CSV)",
                csv,
                "signal_data.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            metrics_df = pd.DataFrame([metrics])
            metrics_csv = metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Quality Metrics (CSV)",
                metrics_csv,
                "quality_metrics.csv",
                "text/csv",
                use_container_width=True
            )

def assess_quality(metrics: dict) -> str:
    """Assess overall video quality based on metrics."""
    brightness_ok = 80 <= metrics['Y_mean_avg'] <= 180
    stability_ok = metrics['Y_mean_std_over_time'] <= 12
    snr_ok = metrics['snr_green_db'] >= 3

    score = sum([brightness_ok, stability_ok, snr_ok])

    if score == 3:
        return "Excellent (All criteria met)"
    elif score == 2:
        return "Good (2/3 criteria met)"
    elif score == 1:
        return "Fair (1/3 criteria met)"
    else:
        return "Poor (0/3 criteria met)"

def plot_rgb_channels(df: pd.DataFrame):
    """Plot RGB channel means over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['t_sec'], y=df['R_mean'],
        name='Red', line=dict(color='red', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df['t_sec'], y=df['G_mean'],
        name='Green', line=dict(color='green', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df['t_sec'], y=df['B_mean'],
        name='Blue', line=dict(color='blue', width=1)
    ))

    fig.update_layout(
        title="RGB Channel Intensities Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Mean Intensity",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_luminance(df: pd.DataFrame):
    """Plot luminance mean and variation over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['t_sec'], y=df['Y_mean'],
        name='Y_mean (Brightness)',
        line=dict(color='yellow', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['t_sec'], y=df['Y_std'],
        name='Y_std (Spatial Variation)',
        line=dict(color='orange', width=1, dash='dot'),
        yaxis='y2'
    ))

    fig.update_layout(
        title="Luminance Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Y_mean (Brightness)",
        yaxis2=dict(
            title="Y_std (Variation)",
            overlaying='y',
            side='right'
        ),
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_green_analysis(df: pd.DataFrame, fps: float):
    """Plot green channel with frequency analysis."""
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['t_sec'], y=df['G_mean'],
            name='Green Channel',
            line=dict(color='green', width=1)
        ))

        fig1.update_layout(
            title="Green Channel (Time Domain)",
            xaxis_title="Time (seconds)",
            yaxis_title="Intensity",
            height=350
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        g_trace = df['G_mean'].dropna().values
        if len(g_trace) > 64:
            try:
                g_bp = bandpass(g_trace, fps, 0.7, 4.0)
                f, Pxx = welch(g_bp, fs=fps, nperseg=min(256, len(g_bp)))

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=f, y=Pxx,
                    fill='tozeroy',
                    line=dict(color='green')
                ))

                fig2.add_vrect(
                    x0=0.7, x1=4.0,
                    fillcolor="green", opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text="Cardiac Band"
                )

                fig2.update_layout(
                    title="Green Channel (Frequency Domain)",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Power Spectral Density",
                    height=350
                )

                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate frequency plot: {str(e)}")
        else:
            st.warning("Insufficient data for frequency analysis")

if __name__ == "__main__":
    main()
