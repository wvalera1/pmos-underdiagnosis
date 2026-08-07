import streamlit as st
import pandas as pd
import plotly.express as px
import json
from urllib.request import urlopen

st.set_page_config(
    page_title="PMOS Underdiagnosis Risk",
    page_icon="🗺️",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("pmos_predictions_all_counties.csv", dtype={"FIPS": str})
    df["RISK_LABEL"] = df["PREDICTED_RISK"].map({1: "High Risk", 0: "Low Risk"})
    return df

@st.cache_data
def load_geojson():
    with urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as r:
        return json.load(r)
# Maps
@st.cache_data
def build_map(_geojson, _df, map_view):
    if map_view == "Risk Classification":
        fig = px.choropleth(
            _df,
            geojson=_geojson,
            locations="FIPS",
            color="RISK_LABEL",
            color_discrete_map={
                "High Risk": "#c0392b",
                "Low Risk": "#5DADE2"
            },
            scope="usa",
            hover_name="label",
            hover_data={
                "FIPS": True,
                "RISK_PROBABILITY": ":.2f",
                "RISK_LABEL": True,
                "PREDICTED_RISK": False,
            },
            labels={
                "RISK_PROBABILITY": "Risk Score",
                "RISK_LABEL": "Classification",
                "FIPS": "FIPS Code"
            }
        )

        fig.update_layout(
            legend_title_text="Risk Classification"
        )

    else:
        fig = px.choropleth(
            _df,
            geojson=_geojson,
            locations="FIPS",
            color="RISK_PROBABILITY",
            range_color=(0, 1),
            color_continuous_scale=[
                "#D6EAF8",  # Very low risk
                "#85C1E9",
                "#3498DB",
                "#F1948A",
                "#C0392B"   # Very high risk
            ],
            scope="usa",
            hover_name="label",
            hover_data={
                "FIPS": True,
                "RISK_PROBABILITY": ":.2f",
                "RISK_LABEL": True,
                "PREDICTED_RISK": False,
            },
            labels={
                "RISK_PROBABILITY": "Predicted Risk",
                "FIPS": "FIPS Code"
            }
        )

        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Risk Probability"
            )
        )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.35
    )

    fig.update_layout(
        title={
            "text": "Predicted PMOS Underdiagnosis Risk by US County",
            "x": 0.5,
            "xanchor": "center"
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=600
    )

    return fig
# --- Load ---
df = load_data()
geojson = load_geojson()

# --- Header ---
st.title("Predicting County-Level PMOS Underdiagnosis Risk in the US")
st.markdown("*AI4ALL Group 14B — Predicting which U.S. counties are at highest risk "
            "for PMOS underdiagnosis using Random Forest classification.*")

# --- Methodology ---
with st.expander("About this project"):
    st.markdown("""
    **What is PMOS?**  
    Polyendocrine Metabolic Ovarian Syndrome (PMOS) affects an estimated 1 in 8 women 
    worldwide. Up to 70% of cases go undiagnosed.

    **What does this model do?**  
    Using three CDC datasets — the Social Vulnerability Index (2022), PLACES Health Data 
    (2025), and NCHS Urban-Rural Classifications — we trained a Random Forest model to 
    classify each of the 3,144 U.S. counties as High Risk or Low Risk for PMOS underdiagnosis.

    **Key predictors identified:**  
    Socioeconomic vulnerability, housing & transportation access, poverty rate, and minority 
    population percentage.

    **Model performance:** Accuracy 92% | F1 (High Risk) 0.81 | ROC-AUC 0.98
    """)

# --- Search ---
st.subheader("Look up your county")
search = st.text_input("Enter a county name (e.g. 'Barbour County')", "")

if search:
    results = df[df["COUNTY"].str.contains(search, case=False, na=False)]
    if len(results) == 0:
        st.warning("No county found. Try a different spelling.")
    else:
        for _, row in results.iterrows():
            icon = "🔴" if row["PREDICTED_RISK"] == 1 else "🟢"
            st.markdown(
                f"**{row['label']}** — {icon} **{row['RISK_LABEL']}** "
                f"(Risk Score: {row['RISK_PROBABILITY']:.2f})"
            )

# --- Map ---
st.subheader("County Risk Map")
map_view = st.radio(
    "Map View",
    ["Risk Classification", "Risk Probability Heat Map"],
    horizontal=True
)
st.plotly_chart(
    build_map(geojson, df, map_view),
    use_container_width=True
)
st.caption(
    "Data: CDC Social Vulnerability Index 2022 · CDC PLACES 2025 · "
    "NCHS Urban-Rural Classification 2023"
)
