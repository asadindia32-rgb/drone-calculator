import streamlit as st
from math import pi, sqrt

st.set_page_config(page_title="Airplane Calculator", page_icon="✈️", layout="wide")
st.title("✈️ Airplane Design Calculator – v2 (All Editable)")

# -----------------------------
# Helpers & unit conversions
# -----------------------------
g = 9.81  # m/s²
def kg_to_N(kg): return kg * g
def N_to_kg(N): return N / g
def in_to_m(x): return x * 0.0254
def m_to_in(x): return x / 0.0254
def ft_to_m(x): return x * 0.3048
def m2_to_ft2(x): return x / (0.3048**2)
def ft2_to_m2(x): return x * (0.3048**2)
def kts_to_mps(v): return v * 0.514444
def mps_to_kts(v): return v / 0.514444

# -----------------------------
# Sidebar – environment & presets
# -----------------------------
st.sidebar.header("Environment")
rho = st.sidebar.number_input("Air density ρ (kg/m³)", value=1.225, min_value=0.6, max_value=1.6, step=0.005, help="Sea level ~1.225")
mu = st.sidebar.number_input("Dynamic viscosity μ (Pa·s)", value=1.81e-5, format="%.2e", help="Air at 15°C ≈ 1.81e-5")
st.sidebar.caption("Tip: Reduce ρ for hot/high conditions.")

# -----------------------------
# Tabs
# -----------------------------
tab_geom, tab_aero, tab_prop, tab_batt, tab_cg, tab_sum = st.tabs(
    ["Geometry", "Aerodynamics", "Propulsion", "Battery / Endurance", "Mass & Balance", "Summary"]
)

# Shared state defaults
if "data" not in st.session_state:
    st.session_state.data = {}

D = st.session_state.data  # shorthand

# =========================================
# GEOMETRY
# =========================================
with tab_geom:
    st.subheader("Geometry")
    colA, colB, colC = st.columns(3)

    with colA:
        wingspan_unit = st.radio("Wingspan unit", ["in", "m"], horizontal=True, key="span_unit")
        wingspan_val = st.number_input(f"Wingspan ({wingspan_unit})", value=60.0, min_value=10.0, step=1.0)
        span_m = in_to_m(wingspan_val) if wingspan_unit == "in" else wingspan_val

        area_mode = st.radio("Wing area input", ["Chord × Span", "Direct area"], horizontal=True)

    with colB:
        if area_mode == "Chord × Span":
            chord_unit = st.radio("Chord unit", ["in", "m"], horizontal=True)
            chord_val = st.number_input(f"Mean chord ({chord_unit})", value=10.0, min_value=2.0, step=0.5)
            chord_m = in_to_m(chord_val) if chord_unit == "in" else chord_val
            S = span_m * chord_m
        else:
            area_unit = st.radio("Area unit", ["m²", "ft²"], horizontal=True)
            area_val = st.number_input(f"Wing area ({area_unit})", value=0.387, min_value=0.02, step=0.01, format="%.3f")
            S = ft2_to_m2(area_val) if area_unit == "ft²" else area_val

        taper = st.number_input("Taper ratio λ (tip/root)", value=1.0, min_value=0.2, max_value=1.0, step=0.05)

    with colC:
        mass_unit = st.radio("Mass unit", ["kg", "g"], horizontal=True)
        mass_val = st.number_input(f"All-up mass ({mass_unit})", value=1.6, min_value=0.1 if mass_unit=="kg" else 100.0, step=0.1)
        m_kg = mass_val if mass_unit == "kg" else mass_val/1000.0
        W = kg_to_N(m_kg)

        st.write("—")
        st.metric("Aspect Ratio AR", f"{(span_m**2/S):.2f}")
        st.metric("Wing Loading W/S (N/m²)", f"{(W/S):.1f}")

    # Store
    D.update({"span_m": span_m, "S": S, "AR": span_m**2/S, "m_kg": m_kg, "W": W, "taper": taper})

# =========================================
# AERODYNAMICS
# =========================================
with tab_aero:
    st.subheader("Aerodynamics")
    c1, c2, c3 = st.columns(3)
    with c1:
        CLmax = st.number_input("CL_max (takeoff/landing config)", value=1.2, min_value=0.6, max_value=2.5, step=0.05)
        Cd0 = st.number_input("Parasitic drag coefficient C_D0", value=0.035, min_value=0.010, max_value=0.090, step=0.001)
    with c2:
        e = st.number_input("Oswald efficiency e", value=0.85, min_value=0.5, max_value=1.0, step=0.01)
        cruise_kts = st.number_input("Cruise speed (knots)", value=35.0, min_value=10.0, max_value=120.0, step=1.0)
    with c3:
        safety = st.slider("Stall speed safety factor", 1.1, 1.6, 1.3, 0.05, help="Vs × factor ≈ rotation/approach")

    # Calculations
    AR = D["AR"]; S = D["S"]; W = D["W"]
    Vs = sqrt((2*W)/(rho*S*CLmax))                       # m/s
    Vcr = kts_to_mps(cruise_kts)                          # m/s
    k = 1/(pi*e*AR)
    # Lift coefficient at cruise (from equilibrium)
    CL_cr = (2*W)/(rho*Vcr**2*S)
    Cd_cr = Cd0 + k*CL_cr**2
    D_cr = 0.5*rho*Vcr**2*S*Cd_cr                         # Drag at cruise (N)
    P_req = D_cr*Vcr                                      # Shaft power without prop losses (W)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Stall speed Vs", f"{mps_to_kts(Vs):.1f} kt")
        st.caption(f"Rotate/approach ≥ {mps_to_kts(Vs*safety):.1f} kt")
    with c5:
        st.metric("Cruise C_L", f"{CL_cr:.2f}")
        st.metric("Cruise drag D", f"{D_cr:.1f} N")
    with c6:
        st.metric("Ideal required power", f"{P_req/1000:.2f} kW")
    D.update({"CLmax": CLmax, "Cd0": Cd0, "e": e, "Vs": Vs, "Vcr": Vcr, "P_req": P_req, "D_cr": D_cr})

# =========================================
# PROPULSION
# =========================================
with tab_prop:
    st.subheader("Propulsion")
    p1, p2, p3 = st.columns(3)
    with p1:
        n_motors = st.number_input("Number of motors", value=1, min_value=1, max_value=4, step=1)
        thrust_each_N = st.number_input("Static thrust per motor (N)", value=20.0, min_value=1.0, step=0.5)
        total_thrust = n_motors * thrust_each_N
    with p2:
        prop_eff = st.slider("Propulsive efficiency η_prop", 0.4, 0.9, 0.7, 0.01)
        elec_eff = st.slider("Electrical efficiency η_elec (ESC/wiring)", 0.7, 0.99, 0.92, 0.01)
    with p3:
        prop_diam_in = st.number_input("Prop diameter (in)", value=10.0, min_value=6.0, max_value=24.0, step=0.5)
        prop_diam_m = in_to_m(prop_diam_in)
        disk_area = n_motors * (pi*(prop_diam_m/2)**2)
        disk_loading = D["W"]/disk_area if disk_area>0 else 0.0

    T_W = total_thrust / D["W"]
    st.metric("Thrust-to-Weight T/W", f"{T_W:.2f}")
    st.metric("Prop disk loading (N/m²)", f"{disk_loading:.0f}")

    # Shaft and electrical power required at cruise
    P_shaft = D["P_req"] / max(prop_eff, 1e-6)
    P_elec = P_shaft / max(elec_eff, 1e-6)
    st.metric("Shaft power at cruise", f"{P_shaft/1000:.2f} kW")
    st.metric("Electrical power at cruise", f"{P_elec/1000:.2f} kW")

    D.update({"n_motors": n_motors, "T_total": total_thrust, "T_W": T_W,
              "eta_prop": prop_eff, "eta_elec": elec_eff, "P_elec": P_elec})

# =========================================
# BATTERY / ENDURANCE
# =========================================
with tab_batt:
    st.subheader("Battery & Endurance")
    b1, b2, b3 = st.columns(3)
    with b1:
        V_pack = st.number_input("Battery voltage (V)", value=14.8, min_value=7.4, max_value=50.0, step=0.1)
        cap_mAh = st.number_input("Capacity (mAh)", value=5000.0, min_value=500.0, step=100.0)
        DoD = st.slider("Usable depth of discharge", 0.3, 0.95, 0.8, 0.05)
    with b2:
        reserve_min = st.number_input("Reserve time (min)", value=3.0, min_value=0.0, step=0.5)
        avg_factor = st.slider("Avg power factor of cruise (0.5–1.2)", 0.3, 1.5, 1.0, 0.05,
                               help="Average flight power relative to cruise electrical power")
    with b3:
        # Compute endurance
        Ah = cap_mAh/1000.0
        usable_Ah = Ah*DoD
        P_avg = D["P_elec"] * avg_factor                 # W
        I_avg = P_avg / max(V_pack, 1e-6)               # A
        endurance_hr = usable_Ah / max(I_avg, 1e-9)
        endurance_min = max(0.0, endurance_hr*60.0 - reserve_min)

        st.metric("Estimated average current", f"{I_avg:.1f} A")
        st.metric("Endurance (minus reserve)", f"{endurance_min:.1f} min")

    D.update({"V_pack": V_pack, "cap_mAh": cap_mAh, "DoD": DoD,
              "I_avg": I_avg, "endurance_min": endurance_min, "reserve_min": reserve_min})

# =========================================
# MASS & BALANCE (simple 1D CG)
# =========================================
with tab_cg:
    st.subheader("Mass & Balance (1D along fuselage)")
    st.caption("Enter positions (cm) measured from nose → tail. Positive forward-to-aft.")
    cg1, cg2, cg3 = st.columns(3)
    with cg1:
        x_batt = st.number_input("Battery position x_batt (cm)", value=30.0, step=1.0)
        m_batt = st.number_input("Battery mass (g)", value=400.0, step=10.0)
        x_motor = st.number_input("Motor(s) position x_motor (cm)", value=5.0, step=1.0)
        m_motor = st.number_input("Motor(s) mass (g)", value=250.0, step=10.0)
    with cg2:
        x_wing = st.number_input("Wing AC position x_wing (cm)", value=40.0, step=1.0)
        m_wing = st.number_input("Wing structure mass (g)", value=300.0, step=10.0)
        x_tail = st.number_input("Tail group position x_tail (cm)", value=80.0, step=1.0)
        m_tail = st.number_input("Tail group mass (g)", value=150.0, step=10.0)
    with cg3:
        x_payload = st.number_input("Payload position x_payload (cm)", value=45.0, step=1.0)
        m_payload = st.number_input("Payload mass (g)", value=100.0, step=10.0)
        MAC_in = st.number_input("Mean Aerodynamic Chord (in)", value=10.0, step=0.5)
        LEMAC_cm = st.number_input("LEMAC from nose (cm)", value=35.0, step=1.0)

    # CG calc
    masses_g = [m_batt, m_motor, m_wing, m_tail, m_payload, D["m_kg"]*1000 - (m_batt+m_motor+m_wing+m_tail+m_payload)]
    positions_cm = [x_batt, x_motor, x_wing, x_tail, x_payload, x_wing]  # leftover mass at wing AC
    total_mass_g = sum(masses_g)
    x_cg_cm = sum(m*p for m,p in zip(masses_g, positions_cm)) / max(total_mass_g, 1e-6)

    MAC_cm = m_to_in(in_to_m(MAC_in))*2.54  # MAC in cm (round trip to ensure input is treated as in)
    cg_percent_MAC = ((x_cg_cm - LEMAC_cm) / max(MAC_cm, 1e-6)) * 100.0

    st.metric("CG location from nose", f"{x_cg_cm:.1f} cm")
    st.metric("CG as %MAC", f"{cg_percent_MAC:.1f}%")
    st.caption("Typical safe CG: 25–35% MAC for trainers/sport models.")

    D.update({"x_cg_cm": x_cg_cm, "cg_percent_MAC": cg_percent_MAC})

# =========================================
# SUMMARY
# =========================================
with tab_sum:
    st.subheader("Summary (key results)")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Wing loading", f"{(D['W']/D['S']):.1f} N/m²")
        st.metric("Aspect ratio AR", f"{D['AR']:.2f}")
        st.metric("Stall speed Vs", f"{mps_to_kts(D['Vs']):.1f} kt")
    with s2:
        st.metric("Cruise drag", f"{D['D_cr']:.1f} N")
        st.metric("Electrical power @ cruise", f"{D['P_elec']/1000:.2f} kW")
        st.metric("Thrust-to-weight", f"{D['T_W']:.2f}")
    with s3:
        st.metric("Endurance (minus reserve)", f"{D['endurance_min']:.1f} min")
        st.metric("CG (%MAC)", f"{D['cg_percent_MAC']:.1f}%")
        st.metric("All-up mass", f"{D['m_kg']:.2f} kg")

    st.divider()
    st.caption("All fields are editable across tabs. Change anything to explore different constraints.")

