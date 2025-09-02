import streamlit as st

st.set_page_config(page_title="RC Airplane Design Calculator", page_icon="✈️", layout="centered")

st.title("✈️ RC Airplane Design Calculator")

# Typical target wing loadings (very rough, just for first design passes)
TYPE_DEFAULT_LOADING = {
    "Trainer": 12.0,       # oz/ft²
    "Glider": 8.0,
    "Sport": 15.0,
    "Scale": 18.0,
    "3D Aerobatic": 9.0,
    "Custom": 12.0,
}

st.write("Enter your wingspan and weight. We’ll estimate the **required wing area** from a target wing loading and suggest a **chord** for a rectangular wing.")

left, right = st.columns(2)

with left:
    airplane_type = st.selectbox("Airplane Type", list(TYPE_DEFAULT_LOADING.keys()))
    wingspan_in = st.number_input("Wingspan (inches)", min_value=10.0, max_value=200.0, value=40.0, step=1.0)
    weight_g = st.number_input("All-up weight (grams)", min_value=100.0, max_value=10000.0, value=1200.0, step=50.0)

with right:
    default_loading = TYPE_DEFAULT_LOADING[airplane_type]
    target_loading = st.number_input("Target wing loading (oz/ft²)", min_value=5.0, max_value=30.0,
                                     value=default_loading, step=0.5)

def grams_to_oz(g): return g / 28.3495
def ft2_to_in2(ft2): return ft2 * 144.0

if st.button("Submit!"):
    weight_oz = grams_to_oz(weight_g)              # convert to ounces
    req_area_ft2 = weight_oz / target_loading      # ft²
    req_area_in2 = ft2_to_in2(req_area_ft2)        # in²
    # For a rectangular wing, chord = area / span
    chord_in = req_area_in2 / wingspan_in if wingspan_in > 0 else 0.0

    st.subheader("Results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Required wing area", f"{req_area_in2:.0f} in²")
        st.metric("Suggested chord (rectangular)", f"{chord_in:.2f} in")
    with c2:
        st.metric("Target wing loading", f"{target_loading:.1f} oz/ft²")
        st.metric("Model weight", f"{weight_g:.0f} g")

    st.caption("Tip: For easier, slower landings, use a **larger wing area** (lower loading). For faster models, use a **smaller area** (higher loading).")
