import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from itertools import combinations
from sklearn.linear_model import Ridge

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="Microbiome Open Day", layout="wide")

# ============================================================
# Constants
# ============================================================
TRAIN_N0 = 4
TRAIN_C = 5

TAXA_ORDER = [
    'Actinomycetota',
    'Bacillota_A_368345',
    'Bacillota_C',
    'Bacillota_I',
    'Bacteroidota',
    'Campylobacterota_A',
    'Fusobacteriota',
    'Patescibacteria',
    'Pseudomonadota',
    'Synergistota'
]

TAXA_COLORS = {
    'Actinomycetota': '#1f77b4',
    'Bacillota_A_368345': '#ff7f0e',
    'Bacillota_C': '#2ca02c',
    'Bacillota_I': '#17becf',
    'Bacteroidota': '#9467bd',
    'Campylobacterota_A': '#8c564b',
    'Fusobacteriota': '#e377c2',
    'Patescibacteria': '#7f7f7f',
    'Pseudomonadota': '#bcbd22',
    'Synergistota': '#d62728',
}

CLASS_INFO = {
    'healthy': {'nugent_range': '0–3', 'label': 'Healthy'},
    'intermediate': {'nugent_range': '4–6', 'label': 'Intermediate'},
    'BV': {'nugent_range': '7–10', 'label': 'Bacterial vaginosis - unhealthy'},
}

RENAME_MAP = {
    'd__Bacteria;p__Actinomycetota': 'Actinomycetota',
    'd__Bacteria;p__Bacillota_A_368345': 'Bacillota_A_368345',
    'd__Bacteria;p__Bacillota_C': 'Bacillota_C',
    'd__Bacteria;p__Bacillota_I': 'Bacillota_I',
    'd__Bacteria;p__Bacteroidota': 'Bacteroidota',
    'd__Bacteria;p__Campylobacterota_A': 'Campylobacterota_A',
    'd__Bacteria;p__Fusobacteriota': 'Fusobacteriota',
    'd__Bacteria;p__Patescibacteria': 'Patescibacteria',
    'd__Bacteria;p__Pseudomonadota': 'Pseudomonadota',
    'd__Bacteria;p__Synergistota': 'Synergistota'
}

# ============================================================
# Helper functions
# ============================================================
def transform_y(y, N0=None, c=None):
    if N0 is None or c is None:
        return y
    N_max = 10
    d = N_max - 2 * N0 + c
    return np.log(N0 + d) - np.log(N_max - y + c)


def convert_to_levels(y):
    return pd.cut(
        y,
        bins=[-0.1, 3, 6, 10],
        labels=[15, 50, 85]
    ).astype(int)


def construct_replicator_features(X):
    linear_part = X.values
    cross_terms = []
    for i, j in combinations(range(X.shape[1]), 2):
        interaction = -(X.iloc[:, i] * X.iloc[:, j]).values.reshape(-1, 1)
        cross_terms.append(interaction)
    quadratic_part = np.hstack(cross_terms)
    return np.hstack([linear_part, quadratic_part])


def fit_transformed_ridge_on_full_data(X, y, N0=None, c=None, alpha=1.0):
    y_transformed = transform_y((y / 10), N0, c)
    X_rep = construct_replicator_features(X)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_rep, y_transformed)
    return model


def predict_nugent_continuous(model, X_rows, N0=TRAIN_N0, c=TRAIN_C):
    X_rep = construct_replicator_features(X_rows)
    y_pred_transformed = model.predict(X_rep)
    N_pred = 10 + c - (10 - N0 + c) / np.exp(y_pred_transformed)
    return N_pred


def predict_from_row(model, X_rows, N0=TRAIN_N0, c=TRAIN_C):
    N_pred = predict_nugent_continuous(model, X_rows, N0=N0, c=c)
    levels = [15, 50, 85]
    output_labels = []
    for Np in N_pred:
        closest = min(levels, key=lambda x: abs(x - 10 * Np))
        if closest == 15:
            output_labels.append('healthy')
        elif closest == 50:
            output_labels.append('intermediate')
        else:
            output_labels.append('BV')
    return output_labels


def get_display_score_from_continuous(N_pred):
    score = float(np.clip(N_pred, 0, 10))
    return int(round(score))


def classify_score(score):
    if score <= 3:
        return 'healthy'
    elif score <= 6:
        return 'intermediate'
    return 'BV'


# ============================================================
# Data loading and model fitting
# ============================================================
@st.cache_data
def load_data(csv_path='microbiome_data.csv'):
    data = pd.read_csv(csv_path)
    y = data.iloc[:, 3].astype(int)
    X_raw = data.iloc[:, 4:]

    X_freq = X_raw[sorted(X_raw.columns)]
    X_freq = X_freq.iloc[:, 1:]  # removes d__Bacteria;__
    X_freq = X_freq.div(X_freq.sum(axis=1), axis=0)
    X_freq = X_freq.loc[:, X_freq.max(axis=0) > 0.01]
    X_freq = X_freq.div(X_freq.sum(axis=1), axis=0)

    X_freq = X_freq.rename(columns=RENAME_MAP)

    cols = [c for c in TAXA_ORDER if c in X_freq.columns]
    X_freq = X_freq[cols]

    return X_freq, y


@st.cache_resource
def load_model(csv_path='microbiome_data.csv', N0_train=TRAIN_N0, c_train=TRAIN_C, alpha=0.05):
    X_freq, y = load_data(csv_path)
    y_levels = convert_to_levels(y)
    model = fit_transformed_ridge_on_full_data(
        X_freq, y_levels, N0=N0_train, c=c_train, alpha=alpha
    )
    return model


# ============================================================
# Composition utilities
# ============================================================
def normalize_composition(values_dict):
    arr = np.array([max(0.0, float(v)) for v in values_dict.values()])
    total = arr.sum()
    if total <= 0:
        arr = np.ones_like(arr) / len(arr)
    else:
        arr = arr / total
    return dict(zip(values_dict.keys(), arr))


def composition_to_df(comp):
    return pd.DataFrame([[comp[t] for t in comp.keys()]], columns=list(comp.keys()))


def update_one_taxon(current, target_taxon, new_value):
    taxa = list(current.keys())
    old_value = current[target_taxon]
    new_value = max(0.0, min(1.0, float(new_value)))

    others = [t for t in taxa if t != target_taxon]
    old_other_total = 1.0 - old_value
    new_other_total = 1.0 - new_value

    updated = current.copy()
    updated[target_taxon] = new_value

    if len(others) == 0:
        return normalize_composition(updated)

    if old_other_total <= 1e-12:
        share = new_other_total / len(others)
        for t in others:
            updated[t] = share
    else:
        scale = new_other_total / old_other_total
        for t in others:
            updated[t] = current[t] * scale

    return normalize_composition(updated)


def reset_state_from_sample(sample_series):
    comp = {k: float(sample_series[k]) for k in sample_series.index}
    comp = normalize_composition(comp)
    st.session_state.comp = comp

    # Reset slider widget values too
    for taxon, value in comp.items():
        st.session_state[f"slider_{taxon}"] = float(round(value * 100, 1))


# ============================================================
# Plotting
# ============================================================
def illustrate_prediction_replicator_ball_live(comp_df, predicted_class):
    row = comp_df.iloc[0]

    nugent_states = ['0–3', '4–6', '7–10']
    row_values = row[::-1]
    taxa_names = row.index[::-1]
    taxa_colors = [TAXA_COLORS[t] for t in taxa_names]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Better horizontal balance
    left_x = 0
    middle_x = 1.9
    right_x = 3.8

    bottom = 0
    bars = []
    for value, color, label in zip(row_values, taxa_colors, taxa_names):
        b = ax.bar(left_x, value, bottom=bottom, color=color, width=0.5, label=label, alpha=0.85)
        bars.append(b[0])
        bottom += value

    ax.text(
        left_x - 0.45, 0.5,
        "Microbiota composition",
        ha='center',
        va='center',
        rotation=90,
        fontsize=14,
        fontweight='bold'
    )

    # Smaller middle circle
    replicator_center = (middle_x, 0.5)
    replicator_radius = 0.34
    circle = Circle(
        replicator_center,
        replicator_radius,
        facecolor="#4A4A4A",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.95
    )
    ax.add_patch(circle)
    ax.text(
        *replicator_center,
        "Replicator\nModel",
        ha='center',
        va='center',
        fontsize=13,
        fontweight='bold',
        color='white'
    )

    box_height = 1 / 3
    display_names = {
        '0–3': 'Healthy',
        '4–6': 'Intermediate',
        '7–10': 'Unhealthy'
    }

    predicted_range = CLASS_INFO[predicted_class]['nugent_range']

    for i, state in enumerate(nugent_states):
        y = i * box_height
        is_pred = (state == predicted_range)

        ax.bar(
            right_x,
            box_height,
            bottom=y,
            width=0.72,
            facecolor="#B0B0B0" if is_pred else "none",
            edgecolor="black",
            linewidth=1.0
        )

        ax.text(
            right_x,
            y + box_height / 2,
            f"{display_names[state]}\n(Nugent {state})",
            ha='center',
            va='center',
            fontsize=11,
            color='black'
        )

    ax.text(
        right_x,
        -0.05,
        "Predicted state",
        ha='center',
        va='top',
        fontsize=14,
        fontweight='bold'
    )

    # Cleaner arrows from taxa to model
    mids = np.cumsum(row_values) - row_values / 2
    for value in mids:
        arrow = FancyArrowPatch(
            posA=(left_x + 0.28, value),
            posB=(middle_x - replicator_radius + 0.02, 0.5),
            connectionstyle="arc3,rad=0.18",
            arrowstyle='-|>',
            mutation_scale=10,
            color='black',
            lw=1.0,
            alpha=0.9
        )
        ax.add_patch(arrow)

    # Arrow from model to predicted class
    state_idx = nugent_states.index(predicted_range)
    arrow_y = state_idx * (1 / 3) + 1 / 6
    arrow = FancyArrowPatch(
        posA=(middle_x + replicator_radius, 0.5),
        posB=(right_x - 0.38, arrow_y),
        connectionstyle="arc3,rad=-0.12",
        arrowstyle='-|>',
        mutation_scale=13,
        color='black',
        lw=1.4
    )
    ax.add_patch(arrow)

    ax.set_xlim(-0.65, right_x + 1.8)
    ax.set_ylim(-0.12, 1.02)
    ax.axis('off')

    ax.legend(
        bars[::-1],
        list(row.index),
        bbox_to_anchor=(0.8, 1.0),
        loc='upper left',
        title="Taxa",
        frameon=True
    )

    plt.tight_layout()
    return fig


# ============================================================
# App UI
# ============================================================
st.title('Vaginal microbiome open day activity')
st.markdown("""
**Tomás Freire**  
 tomas.freire@tecnico.ulisboa.com
""")
st.write(
    'Explore a microbiome composition, see the model prediction, and try to move it toward a healthy state.'
)

st.markdown(
    '''
    **How to use this activity**
    1. Choose an example microbiome.
    2. The app predicts a health state from its composition.
    3. Move the sliders to change the microbiome.
    4. Try to reach **Healthy (Nugent 0–3)**.
    '''
)

X_freq, y = load_data()
model = load_model()

print("Model coefficients:")
for name, coef in zip(
    list(X_freq.columns) + [
        f"{X_freq.columns[i]} * {X_freq.columns[j]}"
        for i, j in combinations(range(X_freq.shape[1]), 2)
    ],
    model.coef_
):
    print(f"{name}: {coef}")
    
    
st.subheader("How the model makes a prediction")

st.markdown(
    """
The model uses the microbiome composition to predict how easy it is for harmful bacteria to invade.

Each taxon can affect the prediction in two ways:

- by its **own effect**
- by its **combined effect with another taxon**

So the model is not only looking at *how much of each group is present*, but also at *how different groups act together*.
"""
)

st.latex(r"""
r_{\mathrm{inv}}
=
\sum_{j=1}^N \beta_j z_j
\;-\;
\sum_{k<j}^N \beta_{jk} z_k z_j
""")

st.markdown(r"""
Here:

- $z_j$ is the relative abundance of taxon $j$
- $\beta_{j}$ measures the **effect of one taxon on its own**
- $\beta_{jk}$ measures the **pairwise effect of two taxa together**

The model then uses these effects to predict the Nugent score class.
""")
   
st.subheader("What the coefficients mean")

st.image("coefficients.png", caption="Model coefficients", use_container_width=True)

st.markdown(
    """
The coefficient figure summarizes what the model learned from the data - a study with 394 women.

- The **linear coefficients** show the effect of each taxon by itself.
- The **interaction coefficients** show how pairs of taxa affect the prediction together.

In simple terms:

- some taxa are linked to **healthier states**
- others are linked to **less healthy states**
- and some pairs of taxa have an extra effect when they appear together

This helps explain *why* changing the microbiome composition changes the prediction.
"""
)


sample_names = [f'Sample {i}' for i in range(len(X_freq))]
sample_idx = st.selectbox(
    'Choose a printed sample / QR-linked sample',
    range(len(X_freq)),
    format_func=lambda i: sample_names[i]
)

colA, colB = st.columns([1, 1])
with colA:
    if st.button('Load this sample'):
        reset_state_from_sample(X_freq.iloc[sample_idx])
with colB:
    if st.button('Reset sliders to selected sample'):
        reset_state_from_sample(X_freq.iloc[sample_idx])

if 'comp' not in st.session_state:
    reset_state_from_sample(X_freq.iloc[sample_idx])

def plot_composition_bar(comp_df):
    row = comp_df.iloc[0]

    fig, ax = plt.subplots(figsize=(1.5, 3.4))

    bottom = 0
    for taxon in row.index[::-1]:
        value = row[taxon]
        ax.bar(
            0,
            value,
            bottom=bottom,
            color=TAXA_COLORS[taxon],
            width=0.6
        )
        bottom += value

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Composition", fontsize=13, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig
    

    

st.subheader('Adjust the microbiome composition')
st.caption('Adjust the sliders. The composition bar updates immediately. Click Update prediction to refresh the model output and figure.')

# Initialize preview state if needed
if 'preview_comp' not in st.session_state:
    st.session_state.preview_comp = st.session_state.comp.copy()

adjust_col, preview_col = st.columns([2.4, 1])

with adjust_col:
    slider_cols = st.columns(2)
    new_values = {}

    for idx, taxon in enumerate(X_freq.columns):
        with slider_cols[idx % 2]:
            current_val = float(st.session_state.preview_comp[taxon])

            new_percent = st.slider(
                f'{taxon} (%)',
                min_value=0.0,
                max_value=100.0,
                value=float(round(current_val * 100, 1)),
                step=0.1,
                key=f'slider_{taxon}'
            )

            new_values[taxon] = new_percent / 100.0

    # Update preview composition immediately
    st.session_state.preview_comp = normalize_composition(new_values)

    if st.button("Update prediction"):
        st.session_state.comp = st.session_state.preview_comp.copy()
        st.rerun()

with preview_col:
    preview_comp_df = composition_to_df(st.session_state.preview_comp)
    preview_fig = plot_composition_bar(preview_comp_df)
    st.pyplot(preview_fig, use_container_width=False)
    
comp_df = composition_to_df(st.session_state.comp)
continuous_N = predict_nugent_continuous(model, comp_df, N0=TRAIN_N0, c=TRAIN_C)[0]
continuous_score = get_display_score_from_continuous(continuous_N)
predicted_class = classify_score(continuous_score)

info = CLASS_INFO[predicted_class]

st.subheader('Prediction')
metric_cols = st.columns(2)
metric_cols[0].metric('Nugent score prediction', str(continuous_score))
metric_cols[1].metric('Nugent score class', info['label'])

if predicted_class == 'healthy':
    st.success('You reached a healthy state.')
elif predicted_class == 'intermediate':
    st.warning('This composition is in the intermediate range.')
else:
    st.error('This composition is in the unhealthy range.')

fig = illustrate_prediction_replicator_ball_live(comp_df, predicted_class)
st.pyplot(fig, use_container_width=True)

st.markdown("""
---

### Want to see what a real research paper looks like?

This project comes from actual research — you can explore the full preprint here:  
🔗 [Read the paper](https://www.biorxiv.org/content/10.64898/2026.02.20.707042v1)

Feel free to scroll through it — even just the figures tell a story!
""")


