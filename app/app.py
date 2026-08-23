import streamlit as st
import pandas as pd
import plotly.express as px
import json
from urllib.request import urlopen

st.set_page_config(
    page_title="PMOS Underdiagnosis Risk",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/predictions/pmos_predictions_all_counties.csv", dtype={"FIPS": str})
    df["RISK_LABEL"] = df["PREDICTED_RISK"].map({1: "High Risk", 0: "Low Risk"})
    return df

@st.cache_data
def load_geojson():
    with urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as r:
        return json.load(r)

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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
            landcolor="rgba(0,0,0,0)",
            showland=True,
            showlakes=True,
            showcoastlines=False
        ),
        height=600
    )

    return fig

# --- Load ---
df = load_data()
geojson = load_geojson()

# --- Header ---
st.title("Predicting County-Level PMOS Underdiagnosis Risk in the US")
st.markdown(
    "*AI4ALL Ignite 2026 — Group 14B: Irene Zhang, Joyce Xu, Walter Valera, "
    "Vaishali Allibada, Andy Romero*"
)
st.markdown(
    "A machine learning pipeline predicting which U.S. counties are at highest risk "
    "of PMOS underdiagnosis, using public health datasets covering social vulnerability, "
    "healthcare access, health outcomes, and geography."
)

st.divider()

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

st.divider()

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

# --- Info Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["Approach", "Key Results", "Fairness & Bias", "Limitations & References"]
)

st.divider()

with tab1:
    st.markdown("""
    **Background**  
    Polyendocrine Metabolic Ovarian Syndrome (PMOS) affects ~170 million women worldwide.
    Up to 70% of cases go undiagnosed, with structural racism and social determinants of
    health identified as key drivers of diagnostic inequity (Silva et al., 2024).

    **1. Data Selection**  
    Three national CDC/NCHS datasets were merged on standardized 5-digit FIPS county codes:
    - [CDC Social Vulnerability Index (2022)](https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html)
    - [CDC PLACES County Data (2025)](https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb)
    - [NCHS Urban-Rural Classification (2023)](https://www.cdc.gov/nchs/data-analysis-tools/urban-rural.html)

    187 missing PLACES values were tied entirely to Kentucky and Pennsylvania counties.
    A pre-imputation bias check found notably lower minority population share and higher
    poverty for these counties; missing values were filled with national column medians.

    **2. Label Engineering**  
    No public dataset directly measures PMOS underdiagnosis. A binary composite label
    was engineered from three literature-backed signals:
    - **Signal 1**: Top-quartile SVI score — high social vulnerability (Silva et al., 2024)
    - **Signal 2**: Bottom-quartile annual checkup or mammography rate — low preventative
      care engagement (Silva et al., 2024)
    - **Signal 3**: NCHS rural code ≥ 4 — rural/peri-urban geography (Ramphul et al., 2025)

    **Underdiagnosis Risk = Signal 1 AND (Signal 2 OR Signal 3)** → 21.3% positive rate
    (671 of 3,144 counties)

    Signal features were excluded from model training to prevent data leakage.

    **3. Model Training**  
    Three classifiers were trained with class-balanced weighting to account for the
    ~21%/79% label imbalance:
    - **Random Forest** (primary): 100 trees, used for feature importance ranking
    - **Logistic Regression** (baseline): trained on top 10 RF features
    - **XGBoost** (benchmark): sequential tree-boosting

    Validation included a stratified 80/20 train-test split and 5-fold cross-validation
    across all 3,144 counties.
    """)

with tab2:
    st.markdown("**Model Performance on Held-Out Test Set**")
    results = pd.DataFrame({
        "Model":     ["Random Forest", "Logistic Regression", "XGBoost"],
        "Precision": ["81%", "65%", "79%"],
        "Recall":    ["72%", "86%", "81%"],
        "F1 Score":  ["76%", "74%", "**80%**"],
        "Accuracy":  ["90%", "87%", "**91%**"],
    })
    st.table(results.set_index("Model"))
    st.markdown("""
    XGBoost achieved the best balance of precision and recall for the high-risk class.
    Logistic Regression identified the largest share of proxy-positive counties (86% recall)
    but produced more false alerts. Random Forest had the highest precision but missed more
    high-risk counties. XGBoost provided the most balanced result with the highest F1 (80%)
    and accuracy (91%).

    > **Important**: This model produces county-level risk estimates to support future
    > research, targeted outreach, and public health planning. It should not be used to
    > diagnose PMOS, determine an individual's care, or treat a predicted hotspot as
    > confirmed underdiagnosis.
    """)

with tab3:
    st.markdown("""
    A **false-negative rate (FNR) audit** was conducted on Random Forest predictions from
    the held-out test set. FNR measures the share of truly high-risk counties the model
    failed to flag. Counties were split into lower and higher groups using the national
    median for each demographic measure.
    """)
    fairness = pd.DataFrame({
        "Group Split":       ["Minority population", "Black population", "Poverty rate"],
        "Lower Group FNR":   ["86.7%", "30.2%", "77.8%"],
        "Higher Group FNR":  ["21.0%", "27.5%", "24.8%"],
    })
    st.table(fairness.set_index("Group Split"))
    st.markdown("""
    The audit found substantial FNR differences for the minority-population and poverty-rate
    splits: the model missed more proxy-positive counties in the **lower-minority** and
    **lower-poverty** groups. FNRs for the Black-population split were more similar across
    groups. These results identify uneven error patterns that warrant continued investigation;
    they do not establish that demographic characteristics cause the model's errors or that
    any county's true PMOS burden is known.
    """)

with tab4:
    st.markdown("""
    **Limitations**

    - **Target engineering**: Features used as predictors come from the same datasets used
      to engineer the target variable and may contain inherent biases.
    - **Data availability**: PMOS cannot be diagnosed by a single test and no patient-level
      clinical dataset is publicly available. The label is a proxy, not a ground truth.
    - **Data bias**: CDC survey data may underrepresent populations with lower healthcare
      engagement due to survey selection bias.
    - **Geographical scope**: Limited to U.S. counties; findings should not be extrapolated
      to other populations without careful consideration.
    - **Kentucky & Pennsylvania**: 187 counties across these two states are missing most
      PLACES 2025 measures and were imputed with national medians. Predictions for these
      counties are less reliable.

    ---

    **Data Sources**
    - CDC PLACES: Local Data for Better Health, County Data 2025 Release
    - CDC/ATSDR Social Vulnerability Index 2022, United States
    - NCHS 2023 Urban-Rural Classification Scheme for Counties

    **Literature**
    - Silva et al. (2024). PCOS Underdiagnosis Patterns by Social Vulnerability Measures. *JCEM*, 110(6), 1657.
    - Ramphul et al. (2025). Geographic Cold Spots of PCOS Diagnosis in Texas. *Journal of the Endocrine Society*.
    - Neven et al. (2026). Prevalence of PCOS: global systematic review. *Human Reproduction Update*, 32(3), 277.
    - Teede et al. (2026). Polyendocrine metabolic ovarian syndrome, the new name for PCOS. *The Lancet*.
    - WHO (2026). Polycystic ovarian syndrome. WHO Fact Sheets.
    - Gadhoumi et al. (2026). Strategies for mitigating AI bias in healthcare. *JAMIA Open*, 9(3).
    - Sung et al. (2026). PCOS: An Update on Diagnosis and Management. *Cleveland Clinic Journal of Medicine*, 93(3).
    """)
