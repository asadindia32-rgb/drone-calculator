import streamlit as st
from math import pi, sqrt
import json, io
import numpy as np
import pandas as pd
import altair as alt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# =========================================================
# App config
# =========================================================
st.set_page_config(page_title="Aircraft & Multirotor Calculator v6 (safe)", page_icon="✈️", layout="wide")
st.title("✈️ Aircraft & Multirotor Calculator — v6 (safe session state)")

# ---------- unit helpers ----------
g = 9.81
def kg_to_N(kg): return kg * g
def N_to_kg(N): return N / g
def in_to_m(x): return x * 0.0254
def m_to_in(x): return x / 0.0254
def m2_to_ft2(x): return x / (0.3048**2)
def ft2_to_m2(x): return x * (0.3048**2)
def kts_to_mps(v): return v * 0.514444
def mps_to_kts(v): return v / 0.514444
def lb_to_kg(lb): return lb * 0.45359237
def kg_to_lb(kg): return kg / 0.45359237
def N_to_lbf(N): return N * 0.224809
def lbf_to_N(lbf): return lbf / 0.224809

# Keys that may exist in session_state from presets/uploads (fine)
FIELD_KEYS = [
    "rho","mu","unit_mode",
    "span_m","S","taper","m_kg",
    "CLmax","Cd0","e","cruise_kts","safety",
    "n_motors","thrust_each_N","eta_prop","eta_elec","prop_diam_in",
    "V_pack","cap_mAh","DoD","reserve_min","avg_factor","target_endurance_min","batt_C_rating",
    "x_batt","m_batt","x_motor","m_motor","x_wing","m_wing","x_tail","m_tail",
    "x_payload","m_payload","MAC_in","LEMAC_cm",
    "mr_span_diam_in","mr_n_motors","mr_thrust_each_N","mr_FM","mr_eta_elec","mr_payload_g",
    "esc_margin_factor"
]

# =========================================================
# Sidebar — Units, Environment, Presets, Libraries
# =========================================================
st.sidebar.header("Environment & Units")
unit_mode = st.sidebar.radio("Units", ["Metric (SI)", "Imperial"], key="unit_mode")
rho = st.sidebar.number_input("Air density ρ (kg/m³)", value=st.session_state.get("rho",1.225),
                              min_value=0.6, max_value=1.6, step=0.005, key="rho")
mu  = st.sidebar.number_input("Dynamic viscosity μ (Pa·s)", value=st.session_state.get("mu",1.81e-5),
                              format="%.2e", key="mu")

st.sidebar.divider()
st.sidebar.subheader("Presets")

PRESETS = {
    "Trainer 1.6 kg (Fixed-wing)": {
        "unit_mode":"Metric (SI)",
        "rho":1.225,"mu":1.81e-5,"span_m":in_to_m(60),"S":in_to_m(60)*in_to_m(10),
        "taper":1.0,"m_kg":1.6,"CLmax":1.2,"Cd0":0.035,"e":0.85,"cruise_kts":35.0,"safety":1.3,
        "n_motors":1,"thrust_each_N":20.0,"eta_prop":0.70,"eta_elec":0.92,"prop_diam_in":10.0,
        "V_pack":14.8,"cap_mAh":5000.0,"DoD":0.8,"reserve_min":3.0,"avg_factor":1.0,"target_endurance_min":0.0,
        "batt_C_rating":20.0,"esc_margin_factor":1.3,
        "x_batt":30.0,"m_batt":400.0,"x_motor":5.0,"m_motor":250.0,"x_wing":40.0,"m_wing":300.0,
        "x_tail":80.0,"m_tail":150.0,"x_payload":45.0,"m_payload":100.0,"MAC_in":10.0,"LEMAC_cm":35.0,
        "mr_span_diam_in":13.0,"mr_n_motors":4,"mr_thrust_each_N":25.0,"mr_FM":0.72,"mr_eta_elec":0.92,"mr_payload_g":0.0
    },
    "Quadcopter 2.0 kg": {
        "unit_mode":"Metric (SI)",
        "rho":1.225,"mu":1.81e-5,"span_m":in_to_m(20),"S":in_to_m(20)*in_to_m(5),
        "taper":1.0,"m_kg":2.0,"CLmax":1.0,"Cd0":0.06,"e":0.8,"cruise_kts":0.0,"safety":1.3,
        "n_motors":1,"thrust_each_N":10.0,"eta_prop":0.65,"eta_elec":0.9,"prop_diam_in":10.0,
        "V_pack":22.2,"cap_mAh":6000.0,"DoD":0.8,"reserve_min":3.0,"avg_factor":1.0,"target_endurance_min":0.0,
        "batt_C_rating":30.0,"esc_margin_factor":1.3,
        "x_batt":25.0,"m_batt":500.0,"x_motor":10.0,"m_motor":500.0,"x_wing":20.0,"m_wing":300.0,
        "x_tail":40.0,"m_tail":200.0,"x_payload":30.0,"m_payload":200.0,"MAC_in":8.0,"LEMAC_cm":18.0,
        "mr_span_diam_in":13.0,"mr_n_motors":4,"mr_thrust_each_N":25.0,"mr_FM":0.72,"mr_eta_elec":0.92,"mr_payload_g":0.0
    }
}

preset_name = st.sidebar.selectbox("Built-in presets", list(PRESETS.keys()))
if st.sidebar.button("Load preset"):
    st.session_state.update(PRESETS[preset_name])
    st.rerun()

def current_config():
    cfg = {k: st.session_state.get(k) for k in FIELD_KEYS}
    return {k:v for k,v in cfg.items() if v is not None}

st.sidebar.download_button("💾 Download current preset (JSON)",
                           data=json.dumps(current_config(), indent=2),
                           file_name="aircraft_preset.json",
                           mime="application/json")

uploaded_preset = st.sidebar.file_uploader("Upload preset (JSON)", type=["json"])
if uploaded_preset:
    try:
        data = json.load(uploaded_preset)
        st.session_state.update(data)
        st.sidebar.success("Preset loaded.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")

# ---- Motor/Prop library (sample + upload) ----
st.sidebar.divider()
st.sidebar.subheader("Motor / Prop Library")

MOTOR_DB = pd.DataFrame([
    {"model":"2207-1700KV (5\" prop)","kv":1700,"max_current_A":35,"voltage_V":14.8,"prop":"5x4.3","max_thrust_N":14.0,"notes":"Freestyle motor"},
    {"model":"2814-900KV (10\" prop)","kv":900,"max_current_A":28,"voltage_V":14.8,"prop":"10x5","max_thrust_N":22.0,"notes":"General fixed-wing"},
    {"model":"3508-700KV (13\" prop)","kv":700,"max_current_A":30,"voltage_V":22.2,"prop":"13x4.4","max_thrust_N":30.0,"notes":"Light quad"},
    {"model":"4114-400KV (15\" prop)","kv":400,"max_current_A":35,"voltage_V":22.2,"prop":"15x5.5","max_thrust_N":40.0,"notes":"Camera quad"},
])

motor_file = st.sidebar.file_uploader("Upload motor CSV", type=["csv"], help="Columns: model,kv,max_current_A,voltage_V,prop,max_thrust_N,notes")
if motor_file:
    try:
        dfu = pd.read_csv(motor_file)
        required = {"model","kv","max_current_A","voltage_V","prop","max_thrust_N","notes"}
        if required.issubset(set(dfu.columns)):
            MOTOR_DB = dfu.copy()
            st.sidebar.success("Custom motor library loaded.")
        else:
            st.sidebar.error("CSV missing required columns.")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

motor_choice = st.sidebar.selectbox("Select motor (optional)", ["—"] + MOTOR_DB["model"].tolist())
apply_to = st.sidebar.multiselect("Apply selected motor thrust to:", ["Fixed-wing motor thrust", "Quad motor thrust"], default=[])

if motor_choice != "—" and apply_to:
    row = MOTOR_DB[MOTOR_DB["model"]==motor_choice].iloc[0]
    if "Fixed-wing motor thrust" in apply_to:
        st.session_state["thrust_each_N"] = float(row["max_thrust_N"])
        st.sidebar.info(f"Fixed-wing thrust set to {row['max_thrust_N']} N.")
    if "Quad motor thrust" in apply_to:
        st.session_state["mr_thrust_each_N"] = float(row["max_thrust_N"])
        st.sidebar.info(f"Quad thrust set to {row['max_thrust_N']} N.")

# ---- Static test table for interpolation ----
st.sidebar.subheader("Static Test Table (optional)")
st.sidebar.caption("Upload per-motor thrust/current vs throttle. Required columns: prop_diam_in, throttle (0–1), thrust_N, current_A, voltage_V")
test_table_file = st.sidebar.file_uploader("Upload test table CSV", type=["csv"])
TEST_DB = None
if test_table_file:
    try:
        TEST_DB = pd.read_csv(test_table_file)
        need_cols = {"prop_diam_in","throttle","thrust_N","current_A","voltage_V"}
        if not need_cols.issubset(TEST_DB.columns):
            st.sidebar.error("Table missing required columns.")
            TEST_DB = None
        else:
            st.sidebar.success("Test table loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read test table: {e}")
        TEST_DB = None

def interp_thrust_current(prop_in, throttle, voltage):
    """Linear interpolation over throttle; pick nearest voltage & prop in table."""
    if TEST_DB is None:
        return None, None
    dfp = TEST_DB.iloc[(TEST_DB["prop_diam_in"]-prop_in).abs().argsort()[:200]].copy()
    if dfp.empty: return None, None
    dfv = dfp.iloc[(dfp["voltage_V"]-voltage).abs().argsort()[:200]].copy()
    if dfv.empty: return None, None
    d = dfv.sort_values("throttle")
    th = float(np.clip(throttle, 0.0, 1.0))
    try:
        thrust = float(np.interp(th, d["throttle"], d["thrust_N"]))
        current = float(np.interp(th, d["throttle"], d["current_A"]))
        return thrust, current
    except Exception:
        return None, None

# =========================================================
# Tabs
# =========================================================
tab_geom, tab_aero, tab_prop, tab_batt, tab_cg, tab_mr, tab_sum = st.tabs(
    ["Geometry", "Aerodynamics", "Propulsion", "Battery / Endurance", "Mass & Balance", "Multirotor (Quad)", "Summary / Report"]
)

# =========================================================
# GEOMETRY (fixed-wing) — safe calc_* writes
# =========================================================
with tab_geom:
    st.subheader("Geometry (Fixed-wing) — linked inputs (safe)")

    ctrl_mode = st.radio("Control by", ["Chord × Span", "Direct area", "Target wing loading"],
                         horizontal=True, key="ctrl_mode")

    colA, colB, colC = st.columns(3)
    with colA:
        if unit_mode == "Imperial":
            span_in = st.number_input("Wingspan (in)", value=m_to_in(st.session_state.get("span_m", in_to_m(60))),
                                      min_value=10.0, step=1.0, key="span_in")
            span_m = in_to_m(span_in)
            mass_lb = st.number_input("All-up mass (lb)", value=kg_to_lb(st.session_state.get("m_kg", 1.6)),
                                      min_value=0.2, step=0.2, key="m_lb")
            m_kg = lb_to_kg(mass_lb)
        else:
            span_m = st.number_input("Wingspan (m)", value=st.session_state.get("span_m", in_to_m(60)),
                                     min_value=0.25, step=0.01, format="%.3f", key="span_m")
            m_kg = st.number_input("All-up mass (kg)", value=st.session_state.get("m_kg", 1.6),
                                   min_value=0.1, step=0.1, key="m_kg")
        W = kg_to_N(m_kg)

    with colB:
        if ctrl_mode == "Chord × Span":
            if unit_mode == "Imperial":
                chord_in = st.number_input("Mean chord (in)", value=10.0, min_value=2.0, step=0.5, key="chord_in")
                chord_m = in_to_m(chord_in)
            else:
                chord_m = st.number_input("Mean chord (m)", value=0.254, min_value=0.05, step=0.005, format="%.3f", key="chord_m")
            S = span_m * chord_m
            wing_loading = W / S
        elif ctrl_mode == "Direct area":
            if unit_mode == "Imperial":
                area_ft2 = st.number_input("Wing area (ft²)", value=m2_to_ft2(st.session_state.get("S", in_to_m(60)*in_to_m(10))),
                                           min_value=0.3, step=0.1, format="%.2f", key="area_ft2")
                S = ft2_to_m2(area_ft2)
            else:
                S = st.number_input("Wing area (m²)", value=st.session_state.get("S", in_to_m(60)*in_to_m(10)),
                                    min_value=0.03, step=0.005, format="%.3f", key="area_m2")
            wing_loading = W / S
            chord_m = S / span_m
        else:
            wl_label = "Target wing loading (N/m²)" if unit_mode=="Metric (SI)" else "Target wing loading (lbf/ft²)"
            default_wl = 80.0
            wl_input = st.number_input(
                wl_label,
                value= default_wl if unit_mode=="Metric (SI)" else (N_to_lbf(default_wl)*m2_to_ft2(1)),
                min_value=10.0, step=1.0, key="wl_input"
            )
            wl_target = wl_input if unit_mode=="Metric (SI)" else lbf_to_N(wl_input)/ft2_to_m2(1)
            S = W / max(wl_target, 1e-6)
            chord_m = S / span_m
            wing_loading = wl_target

        taper = st.number_input("Taper ratio λ (tip/root)", value=st.session_state.get("taper",1.0),
                                min_value=0.2, max_value=1.0, step=0.05, key="taper")

    with colC:
        AR = span_m**2 / max(S, 1e-9)
        if unit_mode == "Imperial":
            st.metric("Aspect Ratio AR", f"{AR:.2f}")
            st.metric("Wing Area", f"{m2_to_ft2(S):.2f} ft² ({S:.3f} m²)")
            st.metric("Wing Loading", f"{N_to_lbf(wing_loading)*m2_to_ft2(1):.1f} lbf/ft² ({wing_loading:.1f} N/m²)")
            st.caption(f"Chord auto-updates: {m_to_in(chord_m):.2f} in ({chord_m:.3f} m)")
        else:
            st.metric("Aspect Ratio AR", f"{AR:.2f}")
            st.metric("Wing Area", f"{S:.3f} m² ({m2_to_ft2(S):.2f} ft²)")
            st.metric("Wing Loading", f"{wing_loading:.1f} N/m² ({N_to_lbf(wing_loading)*m2_to_ft2(1):.1f} lbf/ft²)")
            st.caption(f"Chord auto-updates: {chord_m:.3f} m ({m_to_in(chord_m):.2f} in)")

    # IMPORTANT: store computed values under calc_* keys (never overwrite widget keys)
    st.session_state["calc_span_m"] = span_m
    st.session_state["calc_S"] = S
    st.session_state["calc_AR"] = AR
    st.session_state["calc_m_kg"] = m_kg

# =========================================================
# AERODYNAMICS — read calc_* (fallback to widget)
# =========================================================
with tab_aero:
    st.subheader("Aerodynamics (Fixed-wing)")

    c1, c2, c3 = st.columns(3)
    with c1:
        CLmax = st.number_input("CL_max", value=st.session_state.get("CLmax",1.2), min_value=0.6, max_value=2.5, step=0.05, key="CLmax")
        Cd0   = st.number_input("C_D0 (parasitic)", value=st.session_state.get("Cd0",0.035),
                                min_value=0.010, max_value=0.090, step=0.001, key="Cd0")
    with c2:
        e     = st.number_input("Oswald efficiency e", value=st.session_state.get("e",0.85),
                                min_value=0.5, max_value=1.0, step=0.01, key="e")
        cruise_kts = st.number_input("Cruise speed (kt)", value=st.session_state.get("cruise_kts",35.0),
                                     min_value=0.0, max_value=200.0, step=1.0, key="cruise_kts")
    with c3:
        safety = st.slider("Stall safety factor", 1.1, 1.6, st.session_state.get("safety",1.3), 0.05, key="safety")

    # Use calc_* values when present
    S  = st.session_state.get("calc_S",  st.session_state.get("S", 0.25))
    AR = st.session_state.get("calc_AR", st.session_state.get("AR", 6.0))
    m_kg = st.session_state.get("calc_m_kg", st.session_state.get("m_kg", 1.6))
    W = kg_to_N(m_kg)

    Vs = sqrt((2*W)/(rho*S*max(CLmax,1e-9)))
    Vcr = kts_to_mps(cruise_kts)
    k = 1/(pi*max(e*AR, 1e-9))
    CL_cr = (2*W)/(rho*max(Vcr,1e-6)**2*S) if Vcr>0 else 0.0
    Cd_cr = Cd0 + k*CL_cr**2
    D_cr  = 0.5*rho*max(Vcr,0)**2*S*Cd_cr
    P_req = D_cr*Vcr

    V_vec = np.linspace(max(0.1, Vs*0.6), max(Vcr*1.8, Vs*1.5, 30.0), 60)
    CL_vec = (2*W)/(rho*(V_vec**2)*S)
    Cd_vec = Cd0 + k*(CL_vec**2)
    D_vec  = 0.5*rho*(V_vec**2)*S*Cd_vec
    P_vec  = D_vec*V_vec
    df = pd.DataFrame({"Speed (kt)": mps_to_kts(V_vec), "Power (kW)": P_vec/1000})
    st.altair_chart(alt.Chart(df).mark_line().encode(x="Speed (kt)", y="Power (kW)").properties(height=260),
                    use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Stall speed Vs", f"{mps_to_kts(Vs):.1f} kt ({Vs:.1f} m/s)")
        st.caption(f"Rotate/approach ≥ {mps_to_kts(Vs*safety):.1f} kt")
    with c5:
        st.metric("Cruise C_L", f"{CL_cr:.2f}")
        st.metric("Cruise drag", f"{D_cr:.1f} N ({N_to_lbf(D_cr):.1f} lbf)")
    with c6:
        st.metric("Ideal shaft power @ cruise", f"{P_req/1000:.2f} kW")

    # Computed-only keys (safe to store)
    st.session_state["Vs"] = Vs
    st.session_state["Vcr"] = Vcr
    st.session_state["P_req"] = P_req
    st.session_state["D_cr"] = D_cr

# =========================================================
# PROPULSION — fixed-wing (ESC sizing + interpolation)
# =========================================================
with tab_prop:
    st.subheader("Propulsion (Fixed-wing)")

    p1, p2, p3 = st.columns(3)
    with p1:
        n_motors = st.number_input("Number of motors", value=st.session_state.get("n_motors",1),
                                   min_value=1, max_value=4, step=1, key="n_motors")
        thrust_each_N = st.number_input("Static thrust per motor (N)", value=st.session_state.get("thrust_each_N",20.0),
                                        min_value=1.0, step=0.5, key="thrust_each_N")
        total_thrust = n_motors * thrust_each_N
    with p2:
        eta_prop = st.slider("Propulsive efficiency η_prop", 0.4, 0.9, st.session_state.get("eta_prop",0.7), 0.01, key="eta_prop")
        eta_elec = st.slider("Electrical efficiency η_elec", 0.7, 0.99, st.session_state.get("eta_elec",0.92), 0.01, key="eta_elec")
        esc_margin = st.number_input("ESC margin factor (≥1.2)", value=st.session_state.get("esc_margin_factor",1.3),
                                     min_value=1.0, step=0.05, key="esc_margin_factor")
    with p3:
        prop_diam_in = st.number_input("Prop diameter (in)", value=st.session_state.get("prop_diam_in",10.0),
                                       min_value=6.0, max_value=24.0, step=0.5, key="prop_diam_in")
        prop_diam_m = in_to_m(prop_diam_in)
        disk_area = n_motors * (pi*(prop_diam_m/2)**2)
        calc_m_kg = st.session_state.get("calc_m_kg", st.session_state.get("m_kg", 1.6))
        disk_loading = kg_to_N(calc_m_kg)/disk_area if disk_area>0 else 0.0

    WN = kg_to_N(st.session_state.get("calc_m_kg", st.session_state.get("m_kg", 1.6)))
    T_W = total_thrust / max(WN,1e-9)
    Vcr = st.session_state.get("Vcr", 0.0)
    P_req = st.session_state.get("P_req", 0.0)
    P_shaft = P_req / max(eta_prop, 1e-6) if Vcr>0 else 0.0
    P_elec  = P_shaft / max(eta_elec, 1e-6) if Vcr>0 else 0.0

    cA, cB, cC = st.columns(3)
    with cA: st.metric("Thrust-to-Weight (T/W)", f"{T_W:.2f}")
    with cB: st.metric("Disk loading", f"{disk_loading:.0f} N/m²")
    with cC: st.metric("Electrical power @ cruise", f"{P_elec/1000:.2f} kW")

    # store computed P_elec safely
    st.session_state["P_elec"] = P_elec

    st.markdown("**Optional:** Estimate per-motor current from test table")
    cruise_throttle = st.slider("Assumed cruise throttle (0–1)", 0.2, 0.9, 0.6, 0.05, key="fw_cruise_throttle")
    V_pack = st.session_state.get("V_pack", 14.8)

    i_est_list = []
    if TEST_DB is not None:
        for _ in range(n_motors):
            _, current_i = interp_thrust_current(prop_diam_in, cruise_throttle, V_pack)
            if current_i is not None:
                i_est_list.append(current_i)

    per_motor_current = float(np.mean(i_est_list)) if i_est_list else (P_elec/max(n_motors,1))/max(V_pack,1e-6) if Vcr>0 else 0.0
    esc_min_A = per_motor_current * esc_margin
    st.caption(f"Per-motor current est.: {per_motor_current:.1f} A → ESC ≥ {esc_min_A:.0f} A recommended.")

# =========================================================
# BATTERY / ENDURANCE — dual mode + C-rating
# =========================================================
with tab_batt:
    st.subheader("Battery & Endurance — dual mode + C-rating check")

    mode = st.radio("Mode", ["Given capacity → compute endurance",
                             "Given target endurance → compute required capacity"],
                    horizontal=True, key="batt_mode")

    b1, b2, b3 = st.columns(3)
    with b1:
        V_pack = st.number_input("Battery voltage (V)", value=st.session_state.get("V_pack",14.8),
                                 min_value=7.4, max_value=50.0, step=0.1, key="V_pack")
        DoD = st.slider("Usable depth of discharge", 0.3, 0.95, st.session_state.get("DoD",0.8), 0.05, key="DoD")
        reserve_min = st.number_input("Reserve time (min)", value=st.session_state.get("reserve_min",3.0),
                                      min_value=0.0, step=0.5, key="reserve_min")
        C_rating = st.number_input("Battery C-rating (continuous)", value=st.session_state.get("batt_C_rating",20.0),
                                   min_value=1.0, step=1.0, key="batt_C_rating")
    with b2:
        avg_factor = st.slider("Avg power factor vs cruise", 0.3, 1.5, st.session_state.get("avg_factor",1.0), 0.05, key="avg_factor")
        P_avg = st.session_state.get("P_elec",0.0) * avg_factor
        I_avg = P_avg / max(V_pack, 1e-6)
        st.metric("Estimated avg power", f"{P_avg/1000:.2f} kW")
        st.metric("Estimated avg current", f"{I_avg:.1f} A")
    with b3:
        if mode == "Given capacity → compute endurance":
            cap_mAh = st.number_input("Capacity (mAh)", value=st.session_state.get("cap_mAh",5000.0),
                                      min_value=200.0, step=100.0, key="cap_mAh")
            Ah = cap_mAh/1000.0
            usable_Ah = Ah*DoD
            endurance_hr = usable_Ah / max(I_avg, 1e-9) if I_avg>0 else 0.0
            endurance_min = max(0.0, endurance_hr*60.0 - reserve_min)
            st.metric("Endurance (minus reserve)", f"{endurance_min:.1f} min")
            st.session_state["endurance_min"] = endurance_min  # computed key (safe)
            st.session_state["target_endurance_min"] = 0.0
        else:
            target_endurance = st.number_input("Target endurance (min)", value=st.session_state.get("target_endurance_min",20.0),
                                               min_value=1.0, step=1.0, key="target_endurance_min")
            need_hr = max((target_endurance + reserve_min)/60.0, 1e-9)
            required_Ah = need_hr * I_avg / max(DoD, 1e-9)
            required_mAh = required_Ah * 1000.0
            st.metric("Required capacity", f"{required_mAh:.0f} mAh")
            st.session_state["cap_mAh"] = required_mAh  # updating widget key is okay (user set via this control)

    # C-rating sanity
    Ah = st.session_state.get("cap_mAh",5000.0)/1000.0
    I_allowed = Ah * C_rating
    if I_avg > I_allowed:
        st.warning(f"⚠️ Average current {I_avg:.1f} A exceeds pack continuous capability ~{I_allowed:.1f} A "
                   f"(C={C_rating}). Use higher C-rating or larger capacity.")

# =========================================================
# MASS & BALANCE (1D)
# =========================================================
with tab_cg:
    st.subheader("Mass & Balance (1D)")
    st.caption("Positions in cm from nose → tail.")

    cg1, cg2, cg3 = st.columns(3)
    with cg1:
        x_batt  = st.number_input("x_batt (cm)", value=st.session_state.get("x_batt",30.0), step=1.0, key="x_batt")
        m_batt  = st.number_input("m_batt (g)",  value=st.session_state.get("m_batt",400.0), step=10.0, key="m_batt")
        x_motor = st.number_input("x_motor (cm)",value=st.session_state.get("x_motor",5.0),  step=1.0, key="x_motor")
        m_motor = st.number_input("m_motor (g)", value=st.session_state.get("m_motor",250.0), step=10.0, key="m_motor")
    with cg2:
        x_wing  = st.number_input("x_wing AC (cm)", value=st.session_state.get("x_wing",40.0), step=1.0, key="x_wing")
        m_wing  = st.number_input("m_wing (g)",     value=st.session_state.get("m_wing",300.0), step=10.0, key="m_wing")
        x_tail  = st.number_input("x_tail (cm)",    value=st.session_state.get("x_tail",80.0),  step=1.0, key="x_tail")
        m_tail  = st.number_input("m_tail (g)",     value=st.session_state.get("m_tail",150.0), step=10.0, key="m_tail")
    with cg3:
        x_payload = st.number_input("x_payload (cm)", value=st.session_state.get("x_payload",45.0), step=1.0, key="x_payload")
        m_payload = st.number_input("m_payload (g)",  value=st.session_state.get("m_payload",100.0), step=10.0, key="m_payload")
        MAC_in   = st.number_input("Mean Aerodynamic Chord (in)", value=st.session_state.get("MAC_in",10.0), step=0.5, key="MAC_in")
        LEMAC_cm = st.number_input("LEMAC from nose (cm)", value=st.session_state.get("LEMAC_cm",35.0), step=1.0, key="LEMAC_cm")

    m_total_g = st.session_state.get("calc_m_kg", st.session_state.get("m_kg",1.6))*1000
    masses_g = [m_batt, m_motor, m_wing, m_tail, m_payload, max(0.0, m_total_g - (m_batt+m_motor+m_wing+m_tail+m_payload))]
    positions_cm = [x_batt, x_motor, x_wing, x_tail, x_payload, x_wing]
    total_mass_g = sum(masses_g)
    x_cg_cm = sum(m*p for m,p in zip(masses_g, positions_cm)) / max(total_mass_g, 1e-6)

    MAC_cm = m_to_in(in_to_m(MAC_in))*2.54
    cg_percent_MAC = ((x_cg_cm - LEMAC_cm) / max(MAC_cm, 1e-6)) * 100.0

    st.metric("CG from nose", f"{x_cg_cm:.1f} cm")
    st.metric("CG as %MAC", f"{cg_percent_MAC:.1f}%")
    st.caption("Typical safe CG: 25–35% MAC for trainers/sport models.")

    # computed-only keys
    st.session_state["x_cg_cm"] = x_cg_cm
    st.session_state["cg_percent_MAC"] = cg_percent_MAC

# =========================================================
# MULTIROTOR (Quad) — safe calc_* reads + interpolation
# =========================================================
with tab_mr:
    st.subheader("Multirotor (Quadcopter)")
    st.caption("Hover power from momentum theory. Endurance & throttle vs payload below.")

    q1, q2, q3 = st.columns(3)
    with q1:
        mr_n = st.number_input("Motors (usually 4)", value=st.session_state.get("mr_n_motors",4),
                               min_value=1, max_value=12, step=1, key="mr_n_motors")
        mr_thrust_each_N = st.number_input("Max static thrust per motor (N)", value=st.session_state.get("mr_thrust_each_N",25.0),
                                           min_value=5.0, step=0.5, key="mr_thrust_each_N")
    with q2:
        mr_prop_diam_in = st.number_input("Prop diameter (in)", value=st.session_state.get("mr_span_diam_in",13.0),
                                          min_value=8.0, max_value=30.0, step=0.5, key="mr_span_diam_in")
        prop_diam_m = in_to_m(mr_prop_diam_in)
        A_disk = mr_n * (pi*(prop_diam_m/2.0)**2)
    with q3:
        mr_FM = st.slider("Figure of Merit (FM)", 0.5, 0.85, st.session_state.get("mr_FM",0.72), 0.01, key="mr_FM")
        mr_eta_elec = st.slider("Electrical efficiency η_elec", 0.80, 0.98, st.session_state.get("mr_eta_elec",0.92), 0.01, key="mr_eta_elec")
        mr_payload_g = st.number_input("Payload for current calc (g)", value=st.session_state.get("mr_payload_g",0.0),
                                       step=50.0, key="mr_payload_g")

    base_mass_kg = st.session_state.get("calc_m_kg", st.session_state.get("m_kg", 1.6))
    total_mass_kg = base_mass_kg + mr_payload_g/1000.0
    W = kg_to_N(total_mass_kg)

    P_ind = (W**1.5) / max(sqrt(2*rho*A_disk), 1e-9) if A_disk>0 else 0.0
    P_shaft = P_ind / max(mr_FM, 1e-6)
    P_elec_hover = P_shaft / max(mr_eta_elec, 1e-6)

    V_pack = st.session_state.get("V_pack",22.2)
    cap_mAh = st.session_state.get("cap_mAh",6000.0)
    DoD = st.session_state.get("DoD",0.8)
    reserve_min = st.session_state.get("reserve_min",3.0)
    Ah = cap_mAh/1000.0
    usable_Ah = Ah*DoD

    # Hover throttle / current
    thrust_needed_each = W / max(mr_n,1)
    if TEST_DB is not None:
        th_grid = np.linspace(0.3, 0.95, 30)
        best_th, best_err, best_curr = None, 1e9, None
        for th in th_grid:
            th_N, th_A = interp_thrust_current(mr_prop_diam_in, th, V_pack)
            if th_N is None: continue
            err = abs(th_N - thrust_needed_each)
            if err < best_err:
                best_err, best_th, best_curr = err, th, th_A
        hover_throttle = best_th if best_th is not None else thrust_needed_each / max(mr_thrust_each_N,1e-6)
        per_motor_hover_A = best_curr if best_curr is not None else (P_elec_hover/max(mr_n,1))/max(V_pack,1e-6)
    else:
        hover_throttle = thrust_needed_each / max(mr_thrust_each_N,1e-6)
        per_motor_hover_A = (P_elec_hover/max(mr_n,1))/max(V_pack,1e-6)

    I_hover = per_motor_hover_A * mr_n
    end_hover_min = usable_Ah / max(I_hover, 1e-9) * 60.0 if I_hover>0 else 0.0
    end_hover_min = max(0.0, end_hover_min - reserve_min)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Disk area (total)", f"{A_disk:.3f} m²")
        st.metric("Disk loading", f"{W/A_disk:.0f} N/m²" if A_disk>0 else "—")
    with r2:
        st.metric("Elec. power @ hover", f"{P_elec_hover/1000:.2f} kW")
        st.metric("Total hover current", f"{I_hover:.1f} A")
    with r3:
        st.metric("Hover throttle", f"{hover_throttle:.2f}")
        st.metric("Hover endurance (minus reserve)", f"{end_hover_min:.1f} min")
    st.caption("Aim for hover throttle ≤ 0.6–0.65. Higher means underpowered.")

    # Payload curves
    payloads_g = np.linspace(0, max(2000, mr_payload_g+500), 50)
    end_list, thr_list = [], []
    for p in payloads_g:
        mkg = base_mass_kg + p/1000.0
        Wp = kg_to_N(mkg)
        Pind = (Wp**1.5) / max(sqrt(2*rho*A_disk), 1e-9) if A_disk>0 else 0.0
        Pelec = (Pind / max(mr_FM,1.0e-6)) / max(mr_eta_elec,1.0e-6)
        if TEST_DB is not None:
            th_grid = np.linspace(0.3, 0.95, 25)
            need_each = Wp/mr_n
            def err(th):
                tn,_ = interp_thrust_current(mr_prop_diam_in, th, V_pack)
                return abs((tn if tn is not None else need_each) - need_each)
            best_th = min(th_grid, key=err)
            th_motor_A = (interp_thrust_current(mr_prop_diam_in, best_th, V_pack)[1]
                          or (Pelec/max(mr_n,1))/max(V_pack,1e-6))
            Ihov = th_motor_A * mr_n
            thr_list.append(best_th)
        else:
            Ihov = Pelec / max(V_pack, 1e-6)
            thr_list.append((Wp/mr_n) / max(mr_thrust_each_N,1e-6))
        endm = (Ah*DoD)/max(Ihov,1e-9)*60.0 - reserve_min if Ihov>0 else 0.0
        end_list.append(max(0.0, endm))

    df_mr = pd.DataFrame({"Payload (g)": payloads_g, "Endurance (min)": end_list, "Hover throttle": thr_list})
    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(alt.Chart(df_mr).mark_line().encode(x="Payload (g)", y="Endurance (min)").properties(height=260),
                        use_container_width=True)
    with cB:
        st.altair_chart(alt.Chart(df_mr).mark_line().encode(x="Payload (g)", y="Hover throttle").properties(height=260),
                        use_container_width=True)

# =========================================================
# SUMMARY / REPORT — read calc_* safely + warnings
# =========================================================
with tab_sum:
    st.subheader("Summary / Report")

    m_kg = st.session_state.get("calc_m_kg", st.session_state.get("m_kg", 1.6))
    S    = st.session_state.get("calc_S",  st.session_state.get("S", 0.25))
    AR   = st.session_state.get("calc_AR", st.session_state.get("AR", 6.0))
    WN   = kg_to_N(m_kg)
    wl   = WN / max(S,1e-9)

    # Quick quad power @ zero payload
    mr_n = st.session_state.get("mr_n_motors",4)
    prop_diam_m = in_to_m(st.session_state.get("mr_span_diam_in",13.0))
    A_disk0 = mr_n * (pi*(prop_diam_m/2.0)**2)
    W0 = kg_to_N(m_kg)
    Pind0 = (W0**1.5) / max(sqrt(2*st.session_state.get('rho',1.225)*A_disk0), 1e-9) if A_disk0>0 else 0.0
    Pelec0 = Pind0 / max(st.session_state.get("mr_FM",0.72)*st.session_state.get("mr_eta_elec",0.92), 1e-6)

    T_W_fw = (st.session_state.get('n_motors',1)*st.session_state.get('thrust_each_N',20.0)) / max(WN,1e-9)

    results = {
        "Units": st.session_state.get("unit_mode","Metric (SI)"),
        "All-up mass (kg)": round(m_kg,3),
        "Wing area (m²)": round(S,4),
        "Wing area (ft²)": round(m2_to_ft2(S),2),
        "Wing loading (N/m²)": round(wl,1),
        "Wing loading (lbf/ft²)": round(N_to_lbf(wl)*m2_to_ft2(1),1),
        "Aspect ratio": round(AR,2),
        "Stall speed Vs (kt)": round(mps_to_kts(st.session_state.get("Vs",0.0)),1),
        "Cruise drag (N)": round(st.session_state.get("D_cr",0.0),1),
        "Cruise power (kW)": round(st.session_state.get("P_elec",0.0)/1000,2),
        "Fixed-wing T/W": round(T_W_fw,2),
        "Fixed-wing endurance (min)": round(st.session_state.get("endurance_min",0.0),1),
        "CG (%MAC)": round(st.session_state.get("cg_percent_MAC",0.0),1),
        "Quad hover power @ 0 payload (kW)": round(Pelec0/1000,2),
    }

    csv_text = "metric,value\n" + "\n".join([f"{k},{v}" for k,v in results.items()])
    st.download_button("⬇️ Download CSV Summary", data=csv_text, file_name="aircraft_summary.csv", mime="text/csv")

    def make_pdf_bytes(summary_dict):
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        x, y = 2*cm, height - 2*cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, "Aircraft & Multirotor Summary (v6 safe)")
        y -= 0.8*cm
        c.setFont("Helvetica", 10)
        for k, v in summary_dict.items():
            if y < 2*cm:
                c.showPage(); y = height - 2*cm; c.setFont("Helvetica", 10)
            c.drawString(x, y, f"{k}: {v}")
            y -= 0.55*cm
        c.showPage(); c.save(); buf.seek(0)
        return buf.read()

    st.download_button("⬇️ Download PDF Summary",
                       data=make_pdf_bytes(results),
                       file_name="aircraft_summary.pdf",
                       mime="application/pdf")

    st.divider()
    st.subheader("Design Warnings & Tips")

    tips = []
    if wl < 50: tips.append("Wing loading is very low → floaty handling; wind-sensitive.")
    if 50 <= wl <= 120: tips.append("Wing loading in typical sport/trainer range.")
    if wl > 130: tips.append("⚠️ High wing loading → higher stall/landing speeds; enlarge wing or reduce weight.")

    Vs = st.session_state.get("Vs", 0.0)
    if mps_to_kts(Vs) > 25: tips.append("⚠️ Stall speed > 25 kt for small RC may be aggressive; consider more area or higher CL_max.")

    if T_W_fw < 0.5: tips.append("⚠️ Fixed-wing T/W < 0.5 may struggle to climb; increase thrust or reduce weight.")
    elif T_W_fw < 0.8: tips.append("T/W around 0.5–0.8: typical trainer/sport.")
    elif T_W_fw >= 1.0: tips.append("T/W ≥ 1.0: strong performance; aerobatics possible.")

    cap_mAh = st.session_state.get("cap_mAh",5000.0); Ah = cap_mAh/1000.0
    I_avg = (st.session_state.get("P_elec",0.0) * st.session_state.get("avg_factor",1.0)) / max(st.session_state.get("V_pack",14.8), 1e-6)
    I_allowed = Ah * st.session_state.get("batt_C_rating",20.0)
    if I_avg > I_allowed: tips.append("⚠️ Avg current exceeds battery continuous capability (C-rating). Use higher C or larger capacity.")

    quad_throttle_est = (kg_to_N(m_kg)/max(st.session_state.get("mr_n_motors",4),1)) / max(st.session_state.get("mr_thrust_each_N",25.0),1e-6)
    if quad_throttle_est > 0.7: tips.append("⚠️ Quadcopter hover throttle > 0.7 → underpowered; choose larger props/motors or reduce weight.")
    elif quad_throttle_est < 0.45: tips.append("Quad hover throttle < 0.45 → generous headroom; efficient for payloads.")

    if st.session_state.get("esc_margin_factor",1.3) < 1.2:
        tips.append("Set ESC margin factor ≥ 1.2 for reliability (transients & heat).")

    if not tips:
        st.success("No issues flagged. Design looks balanced. ✅")
    else:
        for t in tips:
            (st.warning if t.startswith("⚠️") else st.info)(t)

    st.caption("Rules of thumb only—always validate with ground tests and datasheets.")
