import json
import math
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Driver Dashcam Intelligence", layout="wide")

APP_DIR = Path(__file__).parent
DEFAULT_VIDEO = APP_DIR / "driver_video.mp4"
CACHE_DIR = APP_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


@st.cache_data(show_spinner=False)
def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps > 0 else 0
    cap.release()
    return {
        "ok": True,
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


@st.cache_data(show_spinner=True)
def analyze_video(video_path: str, sample_stride: int = 3) -> Tuple[pd.DataFrame, dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    prev_small = None
    sec_rows = []

    sec_motion = []
    sec_brightness = []
    sec_sharpness = []
    sec_face_detected = []
    sec_face_centered = []
    sec_frame_count = []

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sec = int(frame_idx / fps) if fps > 0 else 0

        if frame_idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90))
            brightness = float(np.mean(small))
            sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())

            motion = 0.0
            if prev_small is not None:
                diff = cv2.absdiff(prev_small, small)
                motion = float(np.mean(diff))
            prev_small = small

            faces = face_cascade.detectMultiScale(
                small,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(24, 24),
            )
            face_detected = 1 if len(faces) > 0 else 0
            face_centered = 0
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                cx = x + w / 2
                cy = y + h / 2
                if 0.25 * 160 <= cx <= 0.75 * 160 and 0.2 * 90 <= cy <= 0.8 * 90:
                    face_centered = 1

            while len(sec_motion) <= sec:
                sec_motion.append([])
                sec_brightness.append([])
                sec_sharpness.append([])
                sec_face_detected.append([])
                sec_face_centered.append([])
                sec_frame_count.append(0)

            sec_motion[sec].append(motion)
            sec_brightness[sec].append(brightness)
            sec_sharpness[sec].append(sharpness)
            sec_face_detected[sec].append(face_detected)
            sec_face_centered[sec].append(face_centered)
            sec_frame_count[sec] += 1

        frame_idx += 1

    cap.release()

    for sec in range(len(sec_motion)):
        motion = float(np.mean(sec_motion[sec])) if sec_motion[sec] else 0.0
        brightness = float(np.mean(sec_brightness[sec])) if sec_brightness[sec] else 0.0
        sharpness = float(np.mean(sec_sharpness[sec])) if sec_sharpness[sec] else 0.0
        face_visible_ratio = float(np.mean(sec_face_detected[sec])) if sec_face_detected[sec] else 0.0
        attention_ratio = float(np.mean(sec_face_centered[sec])) if sec_face_centered[sec] else 0.0

        sec_rows.append(
            {
                "second": sec,
                "motion_intensity": motion,
                "brightness": brightness,
                "sharpness": sharpness,
                "face_visible_ratio": face_visible_ratio,
                "attention_ratio": attention_ratio,
            }
        )

    df = pd.DataFrame(sec_rows)
    if df.empty:
        raise RuntimeError("No frames were analyzed.")

    # Normalized risk components
    motion_n = normalize_series(df["motion_intensity"])
    dark_n = 1.0 - normalize_series(df["brightness"])
    blur_n = 1.0 - normalize_series(df["sharpness"])
    face_missing_n = 1.0 - df["face_visible_ratio"].clip(0, 1)
    attn_low_n = 1.0 - df["attention_ratio"].clip(0, 1)

    df["driver_risk_score"] = (
        100
        * (
            0.22 * motion_n
            + 0.18 * dark_n
            + 0.12 * blur_n
            + 0.22 * face_missing_n
            + 0.26 * attn_low_n
        )
    ).round(1)

    df["event_label"] = df.apply(label_event, axis=1)

    summary = {
        "duration_seconds": round(duration, 2),
        "seconds_analyzed": int(len(df)),
        "peak_risk_score": float(df["driver_risk_score"].max()),
        "avg_risk_score": float(df["driver_risk_score"].mean()),
        "face_visible_pct": round(float(df["face_visible_ratio"].mean() * 100), 1),
        "attention_pct": round(float(df["attention_ratio"].mean() * 100), 1),
        "low_visibility_seconds": int((df["brightness"] < df["brightness"].quantile(0.2)).sum()),
        "high_motion_seconds": int((df["motion_intensity"] > df["motion_intensity"].quantile(0.85)).sum()),
    }
    return df, summary


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    min_v = float(s.min())
    max_v = float(s.max())
    if math.isclose(min_v, max_v):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)


def label_event(row: pd.Series) -> str:
    labels = []
    if row["face_visible_ratio"] < 0.35:
        labels.append("face not visible")
    if row["attention_ratio"] < 0.35:
        labels.append("possible distraction")
    if row["brightness"] < 65:
        labels.append("low light")
    if row["motion_intensity"] > 18:
        labels.append("high cabin motion")
    if row["sharpness"] < 25:
        labels.append("blurry frame")
    return ", ".join(labels) if labels else "normal"


@st.cache_data(show_spinner=False)
def build_demo_telemetry(duration_seconds: float, driver_df: pd.DataFrame) -> pd.DataFrame:
    seconds = np.arange(0, max(1, math.ceil(duration_seconds)))
    risk = driver_df["driver_risk_score"].reindex(range(len(seconds)), fill_value=driver_df["driver_risk_score"].mean())
    base_speed = 42 + 10 * np.sin(seconds / 3.5) + 5 * np.sin(seconds / 1.7)
    speed = np.clip(base_speed - (risk.values / 100) * 7, 5, 95)
    accel = np.gradient(speed)
    steering = 8 * np.sin(seconds / 2.1) + 2 * np.cos(seconds / 1.4)
    brake = np.where(accel < -2.0, np.abs(accel) * 8, 0)

    df = pd.DataFrame(
        {
            "second": seconds,
            "speed_kph": np.round(speed, 1),
            "accel_delta": np.round(accel, 2),
            "steering_deg": np.round(steering, 1),
            "brake_intensity": np.round(brake, 1),
        }
    )
    return df


@st.cache_data(show_spinner=False)
def parse_telemetry(uploaded_csv) -> pd.DataFrame:
    df = pd.read_csv(uploaded_csv)
    df.columns = [c.strip() for c in df.columns]
    required = {"second", "speed_kph"}
    if not required.issubset(df.columns):
        raise ValueError("Telemetry CSV must include at least: second, speed_kph")
    for col in ["accel_delta", "steering_deg", "brake_intensity"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["second", "speed_kph", "accel_delta", "steering_deg", "brake_intensity"]].copy()


st.title("🚘 Driver Dashcam Intelligence")
st.caption(
    "Streamlit demo for driver-monitoring and driving-behavior review. This package analyzes the uploaded driver-facing video and can combine it with telemetry for speed and behavior insights."
)

with st.sidebar:
    st.header("Inputs")
    uploaded_video = st.file_uploader("Driver video (MP4/MOV)", type=["mp4", "mov", "avi"])
    uploaded_csv = st.file_uploader("Optional telemetry CSV", type=["csv"])
    stride = st.slider("Analysis sampling stride", 1, 8, 3, help="Higher stride = faster analysis, lower stride = denser analysis.")
    st.markdown("**CSV columns supported**: `second`, `speed_kph`, `accel_delta`, `steering_deg`, `brake_intensity`")

video_path = DEFAULT_VIDEO
if uploaded_video is not None:
    video_path = CACHE_DIR / uploaded_video.name
    video_path.write_bytes(uploaded_video.read())

if not Path(video_path).exists():
    st.error("No driver video found. Add `driver_video.mp4` to the app root or upload a file from the sidebar.")
    st.stop()

info = get_video_info(str(video_path))
if not info.get("ok"):
    st.error("The selected video could not be opened.")
    st.stop()

left, right = st.columns([1.3, 1])
with left:
    st.subheader("🎥 Driver Video")
    st.video(str(video_path))
    st.markdown(
        f"**Video info**  \\nDuration: `{info['duration']:.1f}s`  \\nResolution: `{info['width']} x {info['height']}`  \\nFPS: `{info['fps']:.2f}`"
    )

try:
    driver_df, summary = analyze_video(str(video_path), sample_stride=stride)
except Exception as exc:
    st.error(f"Video analysis failed: {exc}")
    st.stop()

if uploaded_csv is not None:
    try:
        telemetry_df = parse_telemetry(uploaded_csv)
        telemetry_source = "Uploaded telemetry"
    except Exception as exc:
        st.error(f"Telemetry CSV error: {exc}")
        st.stop()
else:
    telemetry_df = build_demo_telemetry(info["duration"], driver_df)
    telemetry_source = "Generated demo telemetry"

merged = pd.merge(driver_df, telemetry_df, on="second", how="left")
for col in ["speed_kph", "accel_delta", "steering_deg", "brake_intensity"]:
    merged[col] = merged[col].fillna(method="ffill").fillna(method="bfill").fillna(0)

merged["harsh_brake_flag"] = (merged["brake_intensity"] >= 16).astype(int)
merged["speeding_flag"] = (merged["speed_kph"] >= 80).astype(int)
merged["aggressive_steer_flag"] = (merged["steering_deg"].abs() >= 10).astype(int)
merged["combined_trip_risk"] = (
    0.45 * merged["driver_risk_score"]
    + 20 * merged["harsh_brake_flag"]
    + 10 * merged["speeding_flag"]
    + 10 * merged["aggressive_steer_flag"]
).clip(0, 100).round(1)

with right:
    st.subheader("📊 Summary")
    a, b = st.columns(2)
    c, d = st.columns(2)
    a.metric("Average driver risk", f"{summary['avg_risk_score']:.1f}/100")
    b.metric("Peak driver risk", f"{summary['peak_risk_score']:.1f}/100")
    c.metric("Face visible", f"{summary['face_visible_pct']:.1f}%")
    d.metric("Attention proxy", f"{summary['attention_pct']:.1f}%")

    st.markdown("**Auto-detected observations**")
    st.markdown(
        f"- Low-visibility seconds: `{summary['low_visibility_seconds']}`\n"
        f"- High-motion seconds: `{summary['high_motion_seconds']}`\n"
        f"- Telemetry source: `{telemetry_source}`"
    )

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "Driver monitoring",
    "Driving behavior",
    "Events",
    "Ask the app",
])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(merged["second"], merged["driver_risk_score"])
    ax1.set_xlabel("Second")
    ax1.set_ylabel("Risk score")
    ax1.set_title("Driver Risk Over Time")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(merged["second"], merged["face_visible_ratio"], label="Face visible ratio")
    ax2.plot(merged["second"], merged["attention_ratio"], label="Attention ratio")
    ax2.set_xlabel("Second")
    ax2.set_ylabel("Ratio")
    ax2.set_title("Driver Presence & Attention Proxies")
    ax2.legend()
    st.pyplot(fig2)

with tab2:
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(merged["second"], merged["speed_kph"])
    ax3.set_xlabel("Second")
    ax3.set_ylabel("Speed (kph)")
    ax3.set_title("Speed Timeline")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(merged["second"], merged["combined_trip_risk"])
    ax4.set_xlabel("Second")
    ax4.set_ylabel("Combined trip risk")
    ax4.set_title("Combined Driver + Driving Behavior Risk")
    st.pyplot(fig4)

    x1, x2, x3 = st.columns(3)
    x1.metric("Harsh brake events", int(merged["harsh_brake_flag"].sum()))
    x2.metric("Speeding seconds", int(merged["speeding_flag"].sum()))
    x3.metric("Aggressive steering seconds", int(merged["aggressive_steer_flag"].sum()))

with tab3:
    flagged = merged[(merged["combined_trip_risk"] >= 55) | (merged["event_label"] != "normal")].copy()
    flagged = flagged[[
        "second",
        "event_label",
        "driver_risk_score",
        "speed_kph",
        "brake_intensity",
        "steering_deg",
        "combined_trip_risk",
    ]].sort_values(["combined_trip_risk", "second"], ascending=[False, True])
    st.dataframe(flagged, use_container_width=True)

with tab4:
    q = st.chat_input("Ask about risk, speed, attention, braking, visibility, or events...")
    if q:
        st.chat_message("user").write(q)
        lower = q.lower()
        top_row = merged.sort_values("combined_trip_risk", ascending=False).iloc[0]
        if "highest risk" in lower or "peak risk" in lower:
            answer = f"The highest combined trip risk was {top_row['combined_trip_risk']:.1f} at second {int(top_row['second'])}."
        elif "speed" in lower and ("max" in lower or "highest" in lower):
            r = merged.loc[merged["speed_kph"].idxmax()]
            answer = f"The highest speed was {r['speed_kph']:.1f} kph at second {int(r['second'])}."
        elif "brake" in lower:
            count = int(merged["harsh_brake_flag"].sum())
            answer = f"The app flagged {count} harsh-braking moments based on telemetry intensity."
        elif "attention" in lower or "distract" in lower:
            low_attn = merged[merged["attention_ratio"] < 0.35]
            answer = f"Potential distraction or off-center attention was seen in {len(low_attn)} analyzed seconds."
        elif "face" in lower:
            answer = f"The driver's face was visible for about {summary['face_visible_pct']:.1f}% of analyzed time."
        elif "light" in lower or "visibility" in lower:
            answer = f"Low visibility was flagged in {summary['low_visibility_seconds']} seconds based on frame brightness."
        else:
            answer = (
                "Try questions like: \n"
                "- What was the highest risk moment?\n"
                "- What was the highest speed?\n"
                "- How many harsh braking events were there?\n"
                "- How often was the driver distracted?"
            )
        st.chat_message("assistant").write(answer)

st.markdown("---")
with st.expander("Important notes"):
    st.markdown(
        "- This demo is optimized for a driver-facing camera and basic telemetry fusion.\n"
        "- **Observed directly from video:** motion, brightness, blur, face visibility, center-position attention proxy.\n"
        "- **Needs external sensors for production accuracy:** true speed, acceleration, harsh braking, lane behavior, road events, GPS/CAN telemetry.\n"
        "- If no telemetry is uploaded, the app generates demo telemetry so the UI is presentation-ready on Streamlit Cloud or GitHub deployment."
    )
