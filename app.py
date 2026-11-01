from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
EVALUATION_PATH = ROOT / "reports" / "evaluation_results.json"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"
PROBABILITY_PATH = ROOT / "reports" / "predicted_probability_analysis.csv"

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 750;
        color: #17365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        color: #17365D;
        font-weight: 700;
    }
    .disclaimer {
        padding: 14px;
        border-left: 4px solid #D99A27;
        background-color: rgba(217, 154, 39, 0.10);
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


try:
    model = load_model()
    metadata = load_json(METADATA_PATH)
    evaluation = load_json(EVALUATION_PATH)
except Exception as exc:
    st.error(f"Unable to load the model or its configuration: {exc}")
    st.stop()


FEATURES = metadata["features"]
NUMERICAL_FEATURES = metadata["numerical_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]
THRESHOLD = float(metadata["decision_threshold"])

CATEGORY_OPTIONS = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "apr", "aug", "dec", "feb", "jan", "jul",
        "jun", "mar", "may", "nov", "oct", "sep",
    ],
    "poutcome": ["failure", "other", "success", "unknown"],
}


# ============================================================
# 3. COMMON FUNCTIONS
# ============================================================

def predict_customers(data):
    """Generate predictions using the saved preprocessing pipeline."""

    missing = [col for col in FEATURES if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = data[FEATURES].copy()

    for col in NUMERICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isnull().any().any():
        raise ValueError(
            "Input contains missing or invalid values. "
            "Please check all required customer fields."
        )

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    return probabilities, predictions


def priority_label(probability):
    if probability < 0.20:
        return "Very Low"
    elif probability < 0.40:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    elif probability < 0.80:
        return "High"
    return "Very High"


def add_prediction_columns(data):
    """Add prediction results to a copy of the original input."""

    probabilities, predictions = predict_customers(data)

    result = data.copy()
    result["subscription_probability"] = probabilities
    result["predicted_class"] = predictions
    result["predicted_subscription"] = np.where(
        predictions == 1, "yes", "no"
    )
    result["marketing_priority"] = [
        priority_label(p) for p in probabilities
    ]

    return result


def show_disclaimer():
    st.markdown(
        """
        <div class="disclaimer">
        <b>Disclaimer:</b> Predictions are decision-support estimates,
        not guarantees. Actual customer decisions may differ.
        Use predictions responsibly and maintain appropriate human
        oversight when planning marketing campaigns.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_page_header(title, description):
    st.markdown(f'<div class="main-header">{title}</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">{description}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🏦 Bank Marketing ML")
st.sidebar.caption("Subscription Prediction System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Prediction",
        "Batch Prediction",
        "Model Performance",
        "Business Insights",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.write(f"**Model:** {metadata['model_name']}")
st.sidebar.write(f"**Features:** {len(FEATURES)}")
st.sidebar.write(f"**Threshold:** {THRESHOLD:.4f}")
st.sidebar.caption("B.Tech AIML | Machine Learning PBL")


# ============================================================
# 5. HOME
# ============================================================

if page == "Home":

    show_page_header(
        "Bank Marketing Subscription Prediction",
        "An intelligent machine learning application for customer "
        "subscription prediction.",
    )

    st.image(
        "https://images.unsplash.com/photo-1563013544-824ae1b704d3"
        "?auto=format&fit=crop&w=1400&q=80",
        use_container_width=True,
    )

    st.markdown("## Project Overview")

    st.write(
        """
        This project uses supervised machine learning to predict
        whether a bank customer is likely to subscribe to a term deposit.

        The system analyzes demographic information, financial
        attributes and previous marketing campaign history to support
        better customer targeting.
        """
    )

    st.markdown("## Project Objectives")

    c1, c2, c3 = st.columns(3)

    c1.metric("Training Records", "4,521")
    c2.metric("Input Features", "15")
    c3.metric("Selected Model", "Gradient Boosting")

    st.markdown("## Application Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Individual Prediction")
        st.write("Predict subscription likelihood for an individual customer.")

        st.markdown("### 📁 Batch Prediction")
        st.write("Upload a customer CSV and generate bulk predictions.")

    with col2:
        st.markdown("### 📊 Model Performance")
        st.write("Explore validation metrics and model evaluation results.")

        st.markdown("### 💡 Business Insights")
        st.write("Explore feature importance and customer probability segments.")

    st.markdown("## Technology Stack")

    st.write(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Imbalanced-learn (SMOTENC)
        - Matplotlib
        - Streamlit
        - Joblib
        """
    )

    show_disclaimer()


# ============================================================
# 6. INDIVIDUAL CUSTOMER PREDICTION
# ============================================================

elif page == "Customer Prediction":

    show_page_header(
        "Individual Customer Prediction",
        "Enter customer details to estimate term deposit subscription likelihood.",
    )

    st.info(
        f"The model uses a decision threshold of **{THRESHOLD:.4f}**. "
        "Predictions at or above this threshold are classified as yes."
    )

    with st.form("customer_prediction_form"):

        st.markdown("### 1. Customer Demographics")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=35
            )
            job = st.selectbox("Job", CATEGORY_OPTIONS["job"])

        with col2:
            marital = st.selectbox(
                "Marital Status", CATEGORY_OPTIONS["marital"]
            )
            education = st.selectbox(
                "Education", CATEGORY_OPTIONS["education"]
            )

        with col3:
            balance = st.number_input(
                "Account Balance",
                min_value=-1000000,
                max_value=10000000,
                value=1000,
            )
            default = st.selectbox(
                "Credit in Default", CATEGORY_OPTIONS["default"]
            )

        st.markdown("### 2. Loan and Contact Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            housing = st.selectbox(
                "Housing Loan", CATEGORY_OPTIONS["housing"]
            )
            loan = st.selectbox(
                "Personal Loan", CATEGORY_OPTIONS["loan"]
            )

        with col2:
            contact = st.selectbox(
                "Contact Type", CATEGORY_OPTIONS["contact"]
            )
            day = st.number_input(
                "Last Contact Day",
                min_value=1,
                max_value=31,
                value=15,
            )

        with col3:
            month = st.selectbox(
                "Last Contact Month", CATEGORY_OPTIONS["month"]
            )
            campaign = st.number_input(
                "Contacts During Current Campaign",
                min_value=1,
                max_value=100,
                value=2,
            )

        st.markdown("### 3. Previous Campaign Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            pdays = st.number_input(
                "Days Since Previous Contact",
                min_value=-1,
                max_value=1000,
                value=-1,
                help="-1 means the customer was not previously contacted.",
            )

        with col2:
            previous = st.number_input(
                "Previous Contacts",
                min_value=0,
                max_value=1000,
                value=0,
            )

        with col3:
            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                CATEGORY_OPTIONS["poutcome"],
            )

        submitted = st.form_submit_button(
            "Predict Subscription", use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "age": age,
            "balance": balance,
            "day": day,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "poutcome": poutcome,
        }])[FEATURES]

        try:
            probabilities, predictions = predict_customers(customer)
            probability = float(probabilities[0])
            prediction = int(predictions[0])

            st.divider()
            st.markdown("## Prediction Result")

            result_col, probability_col = st.columns(2)

            with result_col:
                if prediction == 1:
                    st.success("Predicted Outcome: YES")
                    st.write("This customer meets the model's subscription threshold.")
                else:
                    st.warning("Predicted Outcome: NO")
                    st.write("This customer is below the model's subscription threshold.")

            with probability_col:
                st.metric(
                    "Subscription Probability",
                    f"{probability * 100:.2f}%",
                )
                st.progress(probability)

            st.metric("Decision Threshold", f"{THRESHOLD * 100:.2f}%")
            st.write(
                f"Marketing Priority: **{priority_label(probability)}**"
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    show_disclaimer()


# ============================================================
# 7. BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    show_page_header(
        "Batch Customer Prediction",
        "Upload a CSV file to predict subscription likelihood for multiple customers.",
    )

    st.markdown("### Required CSV Features")

    st.write(
        "Your CSV must contain all 15 model features. "
        "Extra columns such as Id are ignored. The duration column "
        "is excluded from prediction."
    )

    with st.expander("View required feature names"):
        st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Upload a comma-separated CSV file.",
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Data Preview")
            st.dataframe(input_df.head(10), use_container_width=True)

            st.write(f"Uploaded records: **{len(input_df)}**")

            # Remove non-model columns.
            removable = [
                col for col in ["Id", "duration", "y", "subscribed"]
                if col in input_df.columns
            ]

            model_input = input_df.drop(columns=removable)

            missing = [
                col for col in FEATURES if col not in model_input.columns
            ]

            if missing:
                st.error(f"Missing required columns: {missing}")

            elif input_df.empty:
                st.error("The uploaded CSV contains no customer records.")

            else:
                if st.button(
                    "Generate Batch Predictions",
                    use_container_width=True,
                ):
                    try:
                        result = add_prediction_columns(model_input)

                        st.session_state["batch_result"] = result

                        st.success(
                            f"Successfully generated {len(result)} predictions."
                        )

                    except Exception as exc:
                        st.error(f"Unable to generate predictions: {exc}")

        except Exception as exc:
            st.error(f"Unable to read CSV file: {exc}")

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]

        st.divider()
        st.markdown("## Batch Prediction Results")

        total = len(result)
        yes_count = int((result["predicted_subscription"] == "yes").sum())
        no_count = total - yes_count

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", total)
        c2.metric("Predicted Yes", yes_count)
        c3.metric("Predicted No", no_count)

        st.dataframe(result, use_container_width=True)

        csv_data = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions CSV",
            data=csv_data,
            file_name="bank_marketing_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    show_disclaimer()


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    show_page_header(
        "Model Performance",
        "Evaluation results of the selected Gradient Boosting model.",
    )

    default = evaluation["metrics_at_default_threshold"]
    selected = evaluation["metrics_at_selected_threshold"]

    st.markdown("### Selected Model")
    st.info(
        f"Model: **{metadata['model_name']}** | "
        f"Decision Threshold: **{THRESHOLD:.4f}** | "
        f"Validation Records: **{evaluation['validation_size']}**"
    )

    st.markdown("### Performance at Selected Threshold")

    metric_columns = st.columns(3)

    metric_data = [
        ("Accuracy", selected["accuracy"]),
        ("Precision", selected["precision"]),
        ("Recall", selected["recall"]),
        ("F1-score", selected["f1_score"]),
        ("ROC-AUC", selected["roc_auc"]),
        ("PR-AUC", selected["pr_auc"]),
    ]

    for index, (name, value) in enumerate(metric_data):
        metric_columns[index % 3].metric(
            name, f"{value * 100:.2f}%"
        )

    st.markdown("### Default vs Selected Threshold")

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Default 0.50": [
            default["accuracy"],
            default["precision"],
            default["recall"],
            default["f1_score"],
        ],
        "Selected 0.7419": [
            selected["accuracy"],
            selected["precision"],
            selected["recall"],
            selected["f1_score"],
        ],
    }).set_index("Metric")

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(comparison.index))
    width = 0.35

    ax.bar(
        x - width / 2,
        comparison["Default 0.50"],
        width,
        label="Default Threshold (0.50)",
    )

    ax.bar(
        x + width / 2,
        comparison["Selected 0.7419"],
        width,
        label="Selected Threshold (0.7419)",
    )

    ax.set_ylabel("Score")
    ax.set_title("Default vs Selected Threshold Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.index)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Confusion Matrix")

    confusion = pd.DataFrame(
        [
            [selected["true_negatives"], selected["false_positives"]],
            [selected["false_negatives"], selected["true_positives"]],
        ],
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"],
    )

    st.dataframe(confusion, use_container_width=True)

    st.markdown("### Model Details")

    st.write(f"**Algorithm:** {metadata['model_type']}")
    st.write(f"**Preprocessing:** {', '.join(metadata['preprocessing'])}")
    st.write(f"**Training Records:** {metadata['training_dataset_rows']}")
    st.write(f"**Features:** {len(FEATURES)}")
    st.write(f"**Excluded Feature:** duration")

    st.caption(
        "All displayed performance metrics come from the validation split. "
        "The saved pipeline was subsequently retrained on the full labelled dataset."
    )

    show_disclaimer()


# ============================================================
# 9. BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    show_page_header(
        "Business Insights",
        "Understand model feature importance and customer subscription patterns.",
    )

    st.markdown("### 1. Feature Importance")

    try:
        importance = load_csv(IMPORTANCE_PATH)
        importance = importance.sort_values(
            "Importance", ascending=False
        ).head(10)

        st.bar_chart(
            importance.set_index("Feature")["Importance"],
            horizontal=True,
        )

        st.caption(
            "Feature importance indicates how much features contribute "
            "to the trained model. It does not establish causation."
        )

    except Exception as exc:
        st.warning(f"Could not load feature importance: {exc}")

    st.divider()
    st.markdown("### 2. Customer Probability Analysis")

    try:
        probability_data = load_csv(PROBABILITY_PATH)

        st.dataframe(
            probability_data,
            use_container_width=True,
            hide_index=True,
        )

        chart_data = probability_data.set_index("Probability_Group")[
            "Actual_Subscription_Rate"
        ]

        st.bar_chart(chart_data)

        st.caption(
            "These figures describe the validation dataset, not guaranteed "
            "subscription rates for future campaigns."
        )

    except Exception as exc:
        st.warning(f"Could not load probability analysis: {exc}")

    st.divider()
    st.markdown("### 3. Marketing Recommendations")

    st.markdown(
        """
        1. **Prioritize promising customers:** Use model probabilities
           to help plan campaign contact lists.
        2. **Allocate resources carefully:** Consider communication costs
           and the expected benefit of contacting each segment.
        3. **Review customer segments:** Previous campaign outcomes and
           contact characteristics can help inform marketing analysis.
        4. **Maintain human oversight:** Model predictions should support,
           not replace, responsible marketing decisions.
        5. **Monitor model performance:** Evaluate predictions on new
           labelled data before making operational decisions.
        """
    )

    st.warning(
        "Feature importance and associations are predictive observations. "
        "They do not prove that a customer characteristic causes subscription."
    )

    show_disclaimer()


# ============================================================
# 10. FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("Bank Marketing Subscription Prediction")
st.sidebar.caption("Machine Learning PBL | B.Tech AIML")
st.sidebar.caption("Built with Python and Streamlit")

