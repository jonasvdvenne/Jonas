import streamlit as st
from pathlib import Path
import time
import numpy as np
import datetime
from streamlit_autorefresh import st_autorefresh
from smu.smu_device import KeysightB2901, MockSMU
from smu.control import Controller
from smu.config import list_profiles, load_profile, save_profile, PIDProfile, PIDBand
from smu.data_io import save_csv, save_json
from smu.utils import DEFAULT_ALPHA, DEFAULT_ROOM_TEMP_C

st.set_page_config(page_title="Keysight B2901 Controller", layout="wide")

# ---------- Resource / singleton ----------
@st.cache_resource
def get_controller(use_mock=False, resource_addr=None, initial_profile_name="TCAP_Heater"):
    if use_mock:
        smu = MockSMU()
    else:
        smu = KeysightB2901(resource_addr)
    profile = load_profile(initial_profile_name)
    ctl = Controller(smu=smu, pid_profile=profile)
    # sane initial pulse defaults
    ctl.set_pulse(power_w=0.0, duration_s=0.0, period_s=0.0)
    return ctl

# ---------- Sidebar: connection & profiles ----------
with st.sidebar:
    st.header("Connection")
    use_mock = st.toggle("Use Mock SMU (dev)", value=False)
    resource = st.text_input("VISA Resource (blank = auto)", value="")
    st.caption("Example: USB0::0x2A8D::0x1201::MY12345678::INSTR")

    # ---- Profiles ----
    profiles = list_profiles()

    if not profiles:
        st.error("❌ No PID profiles found in configs/. Please create one (e.g. configs/tcap_heater.yaml).")
        st.stop()

    default_profile = "TCAP_Heater" if "TCAP_Heater" in profiles else list(profiles.keys())[0]
    prof_choice = st.selectbox(
        "PID Profile",
        options=list(profiles.keys()),
        index=list(profiles.keys()).index(default_profile)
    )

    connect = st.button("Connect / Initialize", type="primary")

    if connect:
        st.session_state.ctl = get_controller(
            use_mock=use_mock,
            resource_addr=(resource or None),
            initial_profile_name=prof_choice,
        )
        st.success(f"Controller initialized with profile {prof_choice}")

    if "ctl" not in st.session_state:
        st.info("Initialize the controller from the sidebar.")
        st.stop()

    ctl: Controller = st.session_state.ctl

    with st.expander("Create/Save New Profile"):
        name = st.text_input("Profile name", value=f"{ctl.pid_profile.name}_copy")
        tfw = st.number_input("Temp filter window", value=ctl.pid_profile.temp_filter_window, min_value=1, step=1)
        dclip = st.number_input("Derivative clip (°C/s)", value=float(ctl.pid_profile.deriv_clip))
        iclip = st.number_input("Integral clip", value=float(ctl.pid_profile.int_clip))
        cooldown = st.number_input("Post-pulse cooldown (s)", value=float(ctl.pid_profile.post_pulse_cooldown_s))
        drop = st.number_input("Post-pulse drop limit (A)", value=float(ctl.pid_profile.post_pulse_drop_limit), format="%.4f")

        st.markdown("### Bands (highest |err| first)")
        bands = []
        current = ctl.pid_profile.bands
        for idx in range(max(1, len(current))):
            if idx < len(current):
                b = current[idx]
            else:
                b = PIDBand(abs_error_gt=0.0, Kp=0.00025, Ki=0.000001, Kd=0.00035)
            cols = st.columns(4, gap="small")
            with cols[0]: abs_gt = st.number_input(f"Band {idx+1} |err| >", value=float(b.abs_error_gt), key=f"abs{idx}")
            with cols[1]: kp = st.number_input(f"Kp {idx+1}", value=float(b.Kp), key=f"kp{idx}", format="%.8f")
            with cols[2]: ki = st.number_input(f"Ki {idx+1}", value=float(b.Ki), key=f"ki{idx}", format="%.8f")
            with cols[3]: kd = st.number_input(f"Kd {idx+1}", value=float(b.Kd), key=f"kd{idx}", format="%.8f")
            bands.append(PIDBand(abs_error_gt=abs_gt, Kp=kp, Ki=ki, Kd=kd))
        if st.button("Save Profile"):
            prof = PIDProfile(
                name=name, temp_filter_window=int(tfw), deriv_clip=float(dclip),
                int_clip=float(iclip), post_pulse_cooldown_s=float(cooldown),
                post_pulse_drop_limit=float(drop), bands=bands
            )
            path = save_profile(prof)
            st.success(f"Saved: {path.name}. It will appear in the selector after app reload.")

# ---------- Main controls ----------
st.title("Keysight B2901 — Web Control (PID + Pulses + Slope)")

colA, colB = st.columns([2, 1.2])

with colA:
    st.subheader("Basic Parameters")
    # use 5 columns since we removed the top-level target
    cols = st.columns(5)
    room_temp = cols[0].number_input("Room Temp (°C)", value=float(ctl.room_temp_c))
    alpha = cols[1].number_input("Alpha (1/°C)", value=float(ctl.alpha), step=0.00001, format="%.5f")
    band = cols[2].number_input("Boundary Δ (°C)", value=float(ctl.temp_band))
    i_max = cols[3].number_input("Max Current Clamp (A)", value=float(ctl.max_current), format="%.3f")
    v_comp = cols[4].number_input("Compliance Voltage (V)", value=float(ctl.volt_compliance), format="%.2f")

    # Live update ONLY the temp band while running
    if ctl._running:
        ctl.temp_band = band

    st.subheader("SMU Ranges/Protection")
    cols2 = st.columns(3)
    src_rng = cols2[0].selectbox("Source Current Range (A)", options=[0.01,0.03,0.1,0.3,0.4,1.0], index=3)
    meas_i_rng = cols2[1].selectbox("Meas Current Range (A)", options=[0.01,0.03,0.1,0.3,0.4,1.0], index=4)
    meas_v_rng = cols2[2].selectbox("Meas Voltage Range (V)", options=[0.2,2,20,200], index=1)

    if st.button("Apply SMU Settings"):
        ctl.set_basic_params(
            room_temp_c=room_temp,
            alpha=alpha,
            target_temp_c=ctl.target_temp_c,  # single source of truth lives near the plots
            temp_band=band,
            max_current=i_max,
            volt_compliance=v_comp
        )
        ctl.apply_basic_settings(src_range_a=src_rng, meas_i_range_a=meas_i_rng, meas_v_range_v=meas_v_rng)
        st.success("Applied.")

    st.divider()
    st.subheader("R₀ Measurement")
    r0_cols = st.columns(3)
    r0_i = r0_cols[0].number_input("Measure current (A)", value=0.001, format="%.6f")
    r0_d = r0_cols[1].number_input("Delay (s)", value=0.01, format="%.3f")
    r0_n = r0_cols[2].number_input("Samples", value=100, step=10)
    if st.button("Measure R₀"):
        r0 = ctl.measure_r0(i_r0=r0_i, samples=int(r0_n), dly_s=float(r0_d))
        if r0:
            st.success(f"R₀ = {r0:.6f} Ω  (floor current = {ctl.r0_floor:.6f} A)")
        else:
            st.error("R₀ measurement failed.")

    st.divider()
    st.subheader("Pulses")
    cols4 = st.columns(3)
    pwr = cols4[0].number_input("Pulse Power (W)", value=float(ctl.pulse.power_w))
    dur = cols4[1].number_input("Pulse Duration (s)", value=float(ctl.pulse.duration_s))
    per = cols4[2].number_input("Pulse Period (s)", value=float(ctl.pulse.period_s))
    if st.button("Apply Pulse Settings"):
        ctl.set_pulse(power_w=pwr, duration_s=dur, period_s=per)
        st.success("Pulse settings applied.")

    cols5 = st.columns(3)
    with cols5[0]:
        mode = st.radio(
            "Pulse Mode",
            options=["Auto", "Force Paused", "Force Active"],
            horizontal=True,
            index=0 if ctl.pulse_manual_override is None else (0 if ctl.pulse_manual_override else 2 if ctl.pulse_manual_override is False else 1)
        )
        if mode == "Auto":
            ctl.pulse_manual_override = None
        elif mode == "Force Paused":
            ctl.pulse_manual_override = True
        else:
            ctl.pulse_manual_override = False
    with cols5[1]:
        ctl.slope_t0 = st.number_input("Slope Window Start (s)", value=float(ctl.slope_t0), format="%.2f")
    with cols5[2]:
        ctl.slope_t1 = st.number_input("Slope Window End (s)", value=float(ctl.slope_t1), format="%.2f")

    st.divider()
    st.subheader("Run Control")
    cols6 = st.columns(3)
    if cols6[0].button("Start Output", type="primary"):
        try:
            ctl.start()
            st.success("Output started.")
        except Exception as e:
            st.error(str(e))
    if cols6[1].button("Stop Output"):
        ctl.stop()
        st.success("Output stopped.")
    if cols6[2].button("Reset Logs"):
        ctl.log_t.clear(); ctl.log_i.clear(); ctl.log_v.clear(); ctl.log_r.clear(); ctl.log_temp.clear(); ctl.log_pulse_flag.clear()
        ctl.pulse_events.clear()
        st.success("Cleared logs.")


with colB:
    st.subheader("Status")
    status = ctl.latest_status()
    st.metric("Running", "Yes" if status["running"] else "No")
    st.metric("Latest Cmd (A)", f'{status["latest_cmd_a"]:.6f}')
    st.metric("Setpoint (A)", f'{status["setpoint_a"]:.6f}')
    st.metric("Samples", status["n_samples"])
    st.metric("Pulse Active", "Yes" if status["pulse_active"] else "No")
    st.metric("R₀ (Ω)", f"{ctl.r0_measured:.6f}" if ctl.r0_measured else "—")
    st.metric("R₀ floor current (A)", f"{ctl.r0_floor:.6f}")


    st.subheader("Export")
    save_dir = Path(st.text_input("Save directory", value="."))
    base_name = st.text_input("File name (without extension)", value="run")
    date_prefix = datetime.datetime.now().strftime("%y%m%d")
    file_stem = f"{date_prefix}_{base_name}"
    csv_path = save_dir / f"{file_stem}.csv"
    json_path = save_dir / f"{file_stem}.json"
    
    exp_cols = st.columns(2)
    if exp_cols[0].button("Save CSV"):
        save_csv(str(csv_path), ctl.metadata(), ctl.export_rows())
        st.success(f"Saved {csv_path.resolve()}")
    
    if exp_cols[1].button("Save JSON"):
        save_json(str(json_path), ctl.export_payload())
        st.success(f"Saved {json_path.resolve()}")



st.divider()
st.subheader("Live Control")


# Single source of truth for target temperature
live_target = st.number_input(
    "Target Temp (°C)",
    value=float(ctl.target_temp_c),
    key="live_target_temp"
)

# Always push the value into the controller (running or not),
# so when you press Start Output it already has the right setpoint.
ctl.set_target_temp(live_target)


# Auto-refresh every 2 seconds
st_autorefresh(interval=1000, limit=None, key="data_refresh")

# Snapshot and rolling window
t, i, v, r, temp, flag = ctl.snapshot()
WINDOW = 250
t, i, v, temp = t[-WINDOW:], i[-WINDOW:], v[-WINDOW:], temp[-WINDOW:]

if len(t) > 2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.line_chart({"t": t, "Current (A)": i}, x="t")
    with c2:
        st.line_chart({"t": t, "Voltage (V)": v}, x="t")
    with c3:
        st.line_chart({"t": t, "Temp (°C)": temp}, x="t")
else:
    st.info("Waiting for data…")



