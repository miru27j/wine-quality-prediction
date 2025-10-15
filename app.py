# wine_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pickle
from io import BytesIO
import base64

st.set_page_config(page_title="Wine Quality Prediction", layout="wide", page_icon="🍷")

st.title("🍷 Wine Quality Prediction — Streamlit App")
st.markdown(
    """
    Upload the UCI Wine Quality dataset (CSV) or click **Load sample data**.
    This app trains a simple classifier to predict whether a wine is 'good' (quality >= 7) or 'bad' (quality < 7).
    """
)

@st.cache_data
def load_default_data(kind="red"):
    # UCI Wine Quality dataset (CSV uses semicolon delimiter)
    if kind == "red":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, sep=';')
    return df

# Sidebar - data load
st.sidebar.header("Data & Options")
uploaded_file = st.sidebar.file_uploader("Upload winequality CSV (semicolon or comma separated)", type=['csv'])
use_sample = st.sidebar.button("Load sample (UCI red wine)")

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')  # try autodetect separator
        st.success("Uploaded dataset loaded.")
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
elif use_sample:
    st.sidebar.info("Loading UCI red wine dataset from UCI repository...")
    try:
        df = load_default_data("red")
        st.sidebar.success("Sample dataset loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not load sample dataset: {e}")

if df is None:
    st.info("Upload a dataset or press *Load sample (UCI red wine)* to begin.")
    st.stop()

# Basic EDA
st.subheader("Dataset preview")
st.dataframe(df.head())

st.subheader("Dataset summary")
col1, col2 = st.columns(2)
with col1:
    st.write("Shape:")
    st.write(df.shape)
    st.write("Columns:")
    st.write(list(df.columns))
with col2:
    st.write("Statistics:")
    st.dataframe(df.describe().T)

# Target creation: binary classification
st.markdown("---")
st.subheader("Model target")
target_type = st.radio("Target framing", options=["Binary classification (good vs bad)", "Use raw quality as regression (NOT implemented)"], index=0)
if target_type.startswith("Binary"):
    threshold = st.slider("Quality threshold for 'good' wine (>= threshold => good)", 5, 9, 7)
    df['good'] = (df['quality'] >= threshold).astype(int)
    target_col = 'good'
    st.write(f"Binary target created: 1 = good (quality >= {threshold}), 0 = bad")
    st.write(df[target_col].value_counts())
else:
    st.warning("Regression mode is not implemented in this demo. Please use Binary classification.")

# Feature & target selection
features = [c for c in df.columns if c not in ['quality', target_col]]
st.subheader("Feature selection")
selected_features = st.multiselect("Select features to use for training", options=features, default=features)

if not selected_features:
    st.error("Select at least one feature to continue.")
    st.stop()

X = df[selected_features].copy()
y = df[target_col].copy()

# Preprocessing choices
st.subheader("Preprocessing")
scale_data = st.checkbox("Standardize features (recommended)", value=True)
if scale_data:
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)

# Train/test split
st.subheader("Train / Test split")
test_size = st.slider("Test set size (%)", 10, 50, 25)
random_state = st.number_input("Random state (seed)", min_value=0, max_value=9999, value=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size/100.0), random_state=int(random_state), stratify=y)

st.write(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Model selection and hyperparameters
st.subheader("Model selection")
model_choice = st.selectbox("Choose model", ["Random Forest", "Logistic Regression"])
if model_choice == "Random Forest":
    n_estimators = st.slider("Number of trees (n_estimators)", 10, 500, 100)
    max_depth = st.slider("Max depth (0 = None)", 0, 50, 8)
    if max_depth == 0:
        max_depth = None
    model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth, random_state=int(random_state))
else:
    c = st.number_input("Inverse of regularization strength (C)", 0.01, 10.0, 1.0, step=0.01)
    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
    model = LogisticRegression(C=float(c), max_iter=500, solver=solver, random_state=int(random_state))

# Train model
st.markdown("---")
if st.button("▶️ Train model"):
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained — Accuracy on test set: {acc:.3f}")

        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        cm_fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted Bad (0)', 'Predicted Good (1)'],
            y=['Actual Bad (0)', 'Actual Good (1)'],
            annotation_text=cm.astype(str),
            colorscale='Blues'
        )
        cm_fig.update_layout(height=400, width=500, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(cm_fig)

        # ROC AUC (if probabilities available)
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                st.subheader(f"ROC AUC: {auc:.3f}")
                # Plot ROC with sklearn's resp (we can compute FPR, TPR)
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=600, height=400)
                st.plotly_chart(roc_fig)
            except Exception:
                pass

        # Feature importances
        st.subheader("Feature importances / coefficients")
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": selected_features,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            st.dataframe(fi)
            fig_fi = px.bar(fi, x='feature', y='importance', title='Feature importances')
            st.plotly_chart(fig_fi)
        elif hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            fi = pd.DataFrame({"feature": selected_features, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
            st.dataframe(fi)
            fig_coef = px.bar(fi, x='feature', y='coef', title='Model coefficients')
            st.plotly_chart(fig_coef)
        else:
            st.write("Model does not expose feature importances or coefficients.")

        # Save model to session state for predictions / download
        st.session_state['trained_model'] = model
        if scale_data:
            st.session_state['scaler'] = scaler
        st.session_state['selected_features'] = selected_features
        st.session_state['threshold'] = threshold
        st.session_state['target_type'] = target_type

        # Allow model download
        buffer = BytesIO()
        pickle.dump({'model': model, 'scaler': st.session_state.get('scaler', None), 'features': selected_features}, buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="wine_model.pkl">⬇️ Download trained model (pickle)</a>'
        st.markdown(href, unsafe_allow_html=True)

# Single-sample prediction
st.markdown("---")
st.subheader("Single sample prediction")
if 'trained_model' not in st.session_state:
    st.info("Train a model first to enable single-sample predictions.")
else:
    model = st.session_state['trained_model']
    sf = st.session_state['selected_features']

    st.write("Enter feature values for prediction (use same units as dataset):")
    sample_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(sf):
        with cols[i % 3]:
            # We'll set reasonable slider ranges based on dataset values when available
            col_min = float(df[feat].min())
            col_max = float(df[feat].max())
            col_mean = float(df[feat].mean())
            sample_vals[feat] = st.number_input(feat, value=float(col_mean))

    sample_df = pd.DataFrame([sample_vals])
    if scale_data and 'scaler' in st.session_state:
        scaled_sample = st.session_state['scaler'].transform(sample_df)
        sample_for_pred = pd.DataFrame(scaled_sample, columns=sf)
    else:
        sample_for_pred = sample_df

    if st.button("🔮 Predict quality (good=1 / bad=0)"):
        pred = model.predict(sample_for_pred)[0]
        proba = model.predict_proba(sample_for_pred)[0][1] if hasattr(model, "predict_proba") else None
        st.write("Prediction:", "**Good (1)**" if pred == 1 else "**Bad (0)**")
        if proba is not None:
            st.write(f"Probability of being 'Good': {proba:.3f}")

# Footer & run instructions
st.markdown("---")
st.markdown(
    """
    **How to run locally**
    1. `pip install -r requirements.txt` (see below)
    2. `streamlit run wine_app.py`

    **Recommended requirements.txt**
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    ```
    """
)
