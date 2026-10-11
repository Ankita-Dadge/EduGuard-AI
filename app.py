
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix
)

# Page settings
st.set_page_config(
    page_title="EduGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🛡️ EduGuard AI")
st.sidebar.write("Intelligent Proctoring System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset Summary",
        "Model Evaluation",
        "Sample Prediction"
    ]
)

# Load dataset
df = pd.read_csv("proctoring_data.csv")

# Check required columns
required_columns = [
    "eye_movement",
    "head_movement",
    "multiple_faces",
    "suspicious"
]

if not all(column in df.columns for column in required_columns):
    st.error("The dataset is missing one or more required columns.")
    st.stop()

# Prepare data and train model
X = df[
    ["eye_movement", "head_movement", "multiple_faces"]
]
y = df["suspicious"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate using held-out test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Common page heading
st.title("🛡️ EduGuard AI")
st.caption(
    "Intelligent Proctoring System | Machine-learning demonstration"
)
st.info(
    "Educational demo using artificial data. "
    "A flag is not proof of cheating."
)

# PAGE 1: PROJECT OVERVIEW
if page == "Project Overview":
    st.header("Project Overview")

    st.write(
        "EduGuard AI is a machine-learning demonstration "
        "that studies sample observations during online exams."
    )

    st.subheader("Main Features")
    st.markdown("""
    - Sample dataset analysis
    - Machine-learning predictions
    - Accuracy and precision evaluation
    - Confusion matrix visualization
    - Sample prediction testing
    """)

    st.subheader("Ethical Safeguards")
    st.markdown("""
    - A flag is not proof of cheating.
    - Every flag requires human review.
    - The dataset contains artificial data.
    - Never penalize a student based only on this model.
    """)

    st.download_button(
        "📥 Download Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="proctoring_data.csv",
        mime="text/csv"
    )

# PAGE 2: DATASET SUMMARY
elif page == "Dataset Summary":
    st.header("📊 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric(
        "No Flag Samples",
        int((df["suspicious"] == 0).sum())
    )
    col3.metric(
        "Flagged Samples",
        int((df["suspicious"] == 1).sum())
    )

    st.download_button(
        "📥 Download Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="proctoring_data.csv",
        mime="text/csv"
    )

    st.subheader("View and Filter Records")

    filter_option = st.selectbox(
        "Filter records",
        ["All Records", "No Flag", "Flagged"]
    )

    if filter_option == "No Flag":
        filtered_df = df[df["suspicious"] == 0].copy()
    elif filter_option == "Flagged":
        filtered_df = df[df["suspicious"] == 1].copy()
    else:
        filtered_df = df.copy()

    search_text = st.text_input(
        "🔎 Search dataset records",
        key="dataset_search"
    )

    if search_text:
        matches = filtered_df.astype(str).apply(
            lambda row: row.str.contains(
                search_text,
                case=False,
                regex=False
            ).any(),
            axis=1
        )
        filtered_df = filtered_df[matches]

    st.dataframe(filtered_df, use_container_width=True)

    st.caption(f"Records displayed: {len(filtered_df)}")
    st.caption(
        "Artificial demonstration data only. "
        "A flag is not proof of cheating."
    )

# PAGE 3: MODEL EVALUATION
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Samples", len(X_test))
    col2.metric("Accuracy", f"{accuracy:.2%}")
    col3.metric("Precision", f"{precision:.2%}")

    st.subheader("Model Performance")

    performance_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision"],
        "Score": [accuracy, precision]
    })

    st.bar_chart(
        performance_df.set_index("Metric")
    )

    st.subheader("Held-out Confusion Matrix")

    st.write("Rows: Actual labels | Columns: Predicted labels")
    st.write("Label 0: No flag | Label 1: Flagged")

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(cm, cmap="Blues")

    ax.set_title("Held-out Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No flag (0)", "Flag (1)"])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No flag (0)", "Flag (1)"])

    fig.colorbar(image, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    st.pyplot(fig)
    plt.close(fig)

    report = f"""
EduGuard AI - Model Evaluation Report
=====================================

Model: Decision Tree Classifier
Dataset: Artificial sample data
Training samples: {len(X_train)}
Test samples: {len(X_test)}
Accuracy: {accuracy:.2%}
Precision: {precision:.2%}

Important Notice:
These results use artificial data.
They do not establish real-world cheating detection performance.
Every flag requires human review.
"""

    st.download_button(
        "📄 Download Evaluation Report",
        data=report,
        file_name="EduGuard_AI_Evaluation_Report.txt",
        mime="text/plain"
    )

    st.warning(
        "Evaluation scores on artificial data do not "
        "represent real-world performance."
    )

# PAGE 4: SAMPLE PREDICTION
elif page == "Sample Prediction":
    st.header("🔎 Try a Sample Prediction")

    st.warning(
        "Educational demo only. A prediction is not proof "
        "of cheating and must not be used alone to penalize a student."
    )

    # Set default values before creating the widgets
    if "eye_slider" not in st.session_state:
        st.session_state["eye_slider"] = 3

    if "head_slider" not in st.session_state:
        st.session_state["head_slider"] = 2

    if "faces_select" not in st.session_state:
        st.session_state["faces_select"] = 0

    def reset_sample_values():
        st.session_state["eye_slider"] = 3
        st.session_state["head_slider"] = 2
        st.session_state["faces_select"] = 0

    st.button(
        "🔄 Reset Sample Values",
        on_click=reset_sample_values
    )

    st.write("Adjust the sample observation values below.")

    col1, col2 = st.columns(2)

    with col1:
        eye = st.slider(
            "Eye movement score",
            min_value=0,
            max_value=10,
            key="eye_slider"
        )

    with col2:
        head = st.slider(
            "Head movement score",
            min_value=0,
            max_value=10,
            key="head_slider"
        )

    faces = st.selectbox(
        "Additional faces detected",
        [0, 1],
        key="faces_select"
    )

    if st.button("Run Demo", type="primary"):
        sample = pd.DataFrame(
            [[eye, head, faces]],
            columns=[
                "eye_movement",
                "head_movement",
                "multiple_faces"
            ]
        )

        prediction = model.predict(sample)[0]

        st.subheader("Demo Result")

        if prediction == 1:
            st.warning(
                "Flagged for human review. "
                "This is only a model output, not evidence of cheating."
            )
        else:
            st.success(
                "No flag generated for this sample. "
                "This does not prove that no misconduct occurred."
            )

        st.caption(
            "This is a demonstration using artificial data. "
            "All decisions require appropriate human review."
        )

    st.divider()
    st.subheader("Project Conclusion")

    st.write(
        "EduGuard AI demonstrates how machine learning can "
        "classify sample exam observations. Its predictions "
        "are educational outputs, not proof of cheating."
    )