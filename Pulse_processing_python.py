import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# ====================================
# Helpers
# ====================================

def load_file(file) -> tuple[dict, pd.DataFrame]:
    """Load JSON or CSV into metadata + DataFrame with consistent schema"""

    # --- JSON ---
    if file.name.lower().endswith(".json"):
        payload = json.load(file)
        meta = payload.get("metadata", {})
        for k, v in payload.items():
            if k not in ("samples", "pulse_events", "metadata"):
                meta[k] = v
        df = pd.DataFrame(payload["samples"])
        # Guarantee correct dtypes
        for col in ["t_s","i_a","v_v","r_ohm","temp_c"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "pulse_active" in df:
            df["pulse_active"] = df["pulse_active"].astype(bool)
        return meta, df

    # --- CSV ---
    elif file.name.lower().endswith(".csv"):
        raw = file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
    
        lines = raw.splitlines()
    
        # Separate header vs data
        header_lines = [line for line in lines if line.startswith("#")]
        data_lines   = [line for line in lines if not line.startswith("#") and line.strip()]
    
        # Parse metadata
        meta = {}
        for line in header_lines:
            if ":" in line and not line.startswith("# ---"):
                try:
                    k, v = line[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
                except Exception:
                    pass
    
        # Build DataFrame using regex separator (comma or tab)
        csv_str = "\n".join(data_lines)
        df = pd.read_csv(io.StringIO(csv_str), sep=r"[,\t]+")
    
        # Normalize expected column names
        rename_map = {
            "time_s": "t_s",
            "current_a": "i_a",
            "voltage_v": "v_v",
            "resistance_ohm": "r_ohm",
            "temperature_c": "temp_c",
        }
        df = df.rename(columns=rename_map)
    
        if "pulse_active" in df:
            df["pulse_active"] = df["pulse_active"].astype(bool)
    
        return meta, df



    else:
        raise ValueError("Unsupported file format (must be CSV or JSON)")


def detect_pulses(df: pd.DataFrame):
    """Detect start/end times of pulse_active regions"""
    if "pulse_active" not in df:
        return []
    arr = df["pulse_active"].astype(int).values
    edges = np.diff(arr, prepend=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    if len(ends) < len(starts):
        ends = np.append(ends, len(arr) - 1)
    events = []
    for s, e in zip(starts, ends):
        events.append({
            "start_idx": s,
            "end_idx": e,
            "start_time": df.iloc[s]["t_s"] if "t_s" in df else df.index[s],
            "end_time": df.iloc[e]["t_s"] if "t_s" in df else df.index[e],
        })
    return events


def fit_linear(df, t0, start_ms, end_ms):
    """Fit slope of temperature vs time in given window after pulse start"""
    if "t_s" not in df or "temp_c" not in df:
        return np.nan, np.nan, np.nan
    mask = (df["t_s"] >= t0 + start_ms/1000) & (df["t_s"] <= t0 + end_ms/1000)
    sub = df.loc[mask]
    if len(sub) < 3:
        return np.nan, np.nan, np.nan
    X = sub["t_s"].values.reshape(-1,1)
    y = sub["temp_c"].values
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0]), float(model.intercept_), float(model.score(X, y))

# ====================================
# Streamlit App
# ====================================
st.set_page_config(page_title="Pulse Analysis Dashboard", layout="wide")
st.title("⚡ Pulse Heating Analysis Dashboard")

uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv","json"])

if uploaded_file:
    meta, df = load_file(uploaded_file)

    # Sidebar: metadata
    st.sidebar.header("Experiment Metadata")
    for k,v in meta.items():
        st.sidebar.write(f"**{k}:** {v}")

    # Pulse detection
    pulses = detect_pulses(df)
    st.sidebar.metric("Detected pulses", len(pulses))

    # Controls
    st.sidebar.subheader("Analysis Settings")
    start_ms = st.sidebar.number_input("Fit Start (ms)", value=10.0, step=10.0)
    end_ms   = st.sidebar.number_input("Fit End (ms)", value=300.0, step=10.0)
    mode     = st.sidebar.radio("Analysis Mode", ["Slope","Offset"])
    smooth_n = st.sidebar.slider("Smoothing window", 1, 201, 1, step=2)

    # Tabs
    tabs = st.tabs(["Temperature","Current/Voltage","Resistance","Analysis","Raw Pulses"])

    # --- Temperature plot
    with tabs[0]:
        if "temp_c" in df:
            if smooth_n > 1:
                df["temp_smoothed"] = df["temp_c"].rolling(window=smooth_n, center=True).mean()
                fig = px.line(df, x="t_s", y=["temp_c","temp_smoothed"], title="Temperature vs Time")
            else:
                fig = px.line(df, x="t_s", y="temp_c", title="Temperature vs Time")
            for ev in pulses:
                fig.add_vrect(x0=ev["start_time"], x1=ev["end_time"],
                              fillcolor="LightSalmon", opacity=0.3, line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No temperature data in file")

    # --- Current/Voltage
    with tabs[1]:
        if "i_a" in df and "v_v" in df:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["t_s"], y=df["i_a"], name="Current (A)"))
            fig.add_trace(go.Scatter(x=df["t_s"], y=df["v_v"], name="Voltage (V)", yaxis="y2"))
            fig.update_layout(
                title="Current & Voltage vs Time",
                yaxis=dict(title="Current (A)"),
                yaxis2=dict(title="Voltage (V)", overlaying="y", side="right")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No current/voltage data in file")

    # --- Resistance
    with tabs[2]:
        if "r_ohm" in df:
            fig = px.line(df, x="t_s", y="r_ohm", title="Resistance vs Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No resistance data in file")

    # --- Analysis
    with tabs[3]:
        slopes, offsets, r2s, t0s = [],[],[],[]
        for ev in pulses:
            slope, intercept, r2 = fit_linear(df, ev["start_time"], start_ms, end_ms)
            slopes.append(slope); offsets.append(intercept); r2s.append(r2); t0s.append(ev["start_time"])
        results = pd.DataFrame({"pulse_start_s": t0s,"slope_c_per_s": slopes,"offset_c": offsets,"r2": r2s})
        if mode=="Slope":
            base = results["slope_c_per_s"].dropna().iloc[0] if results["slope_c_per_s"].notna().any() else np.nan
            results["percent_change"] = (results["slope_c_per_s"]-base)/base*100 if pd.notna(base) else np.nan
        else:
            base = results["offset_c"].dropna().iloc[0] if results["offset_c"].notna().any() else np.nan
            results["percent_change"] = (results["offset_c"]-base)/base*100 if pd.notna(base) else np.nan
        st.plotly_chart(px.scatter(results, x="pulse_start_s", y="slope_c_per_s" if mode=="Slope" else "offset_c",
                                   title=f"{mode} per Pulse"), use_container_width=True)
        st.plotly_chart(px.scatter(results, x="pulse_start_s", y="percent_change", title=f"{mode} Percent Change (%)"), use_container_width=True)
        st.plotly_chart(px.scatter(results, x="pulse_start_s", y="r2", title="R² per Pulse"), use_container_width=True)
        st.dataframe(results)
        st.download_button("Download Results CSV", results.to_csv(index=False).encode("utf-8"), "pulse_analysis.csv")

    # --- Raw Pulses
    with tabs[4]:
        st.subheader("All Raw Pulses (ΔT per pulse, overlaid)")
        if pulses and "temp_c" in df:
            fig = go.Figure()
            for ev in pulses:
                seg = df.iloc[ev["start_idx"]:ev["end_idx"]+1]
                delta_temp = seg["temp_c"] - seg["temp_c"].iloc[0]
                fig.add_trace(go.Scatter(
                    x=(seg["t_s"]-seg["t_s"].iloc[0])*1000, y=delta_temp,
                    mode="lines", line=dict(width=1), opacity=0.5, showlegend=False
                ))
            fig.update_layout(xaxis_title="Time within pulse (ms)", yaxis_title="Δ Temperature (°C)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pulses detected or no temperature data")

else:
    st.info("👆 Upload a CSV or JSON file to get started.")
