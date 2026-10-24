import os
import joblib

from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = "models"

DISEASES = {
    "heart": {
        "title": "Heart Disease Prediction",
        "file": "heart_model.pkl"
    },
    "diabetes": {
        "title": "Diabetes Prediction",
        "file": "diabetes_model.pkl"
    },
    "cancer": {
        "title": "Breast Cancer Prediction",
        "file": "cancer_model.pkl"
    }
}


def load_disease_model(disease):
    filename = DISEASES[disease]["file"]
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        return None

    return joblib.load(path)


@app.route("/", methods=["GET", "POST"])
def index():

    selected_disease = request.values.get(
        "disease", "heart"
    )

    if selected_disease not in DISEASES:
        selected_disease = "heart"

    bundle = load_disease_model(selected_disease)

    result = None
    probability = None
    error = None
    fields = []

    if bundle is not None:

        fields = bundle["features"]

    if request.method == "POST":

        if bundle is None:
            error = (
                "Model not found. Please run "
                "python train_model.py first."
            )

        else:
            try:
                import pandas as pd

                input_values = []

                for feature in fields:
                    raw_value = request.form.get(feature, "").strip()

                    if not raw_value:
                        raise ValueError(
                            f"Please enter a value for {feature}."
                        )

                    value = float(raw_value)

                    if not pd.notna(value):
                        raise ValueError(
                            f"Invalid value for {feature}."
                        )

                    input_values.append(value)

                # Preserve feature names and order
                patient_data = pd.DataFrame(
                    [input_values],
                    columns=fields
                )

                model = bundle["model"]

                prediction = model.predict(patient_data)[0]

                predicted_class = str(prediction)

                result = bundle.get(
                    "target_names", {}
                ).get(
                    predicted_class,
                    f"Predicted class: {predicted_class}"
                )

                # Display model-estimated probability
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        patient_data
                    )[0]

                    class_labels = [
                        str(c) for c in model.classes_
                    ]

                    class_index = class_labels.index(
                        predicted_class
                    )

                    probability = round(
                        float(probabilities[class_index]) * 100,
                        2
                    )

            except (ValueError, TypeError) as e:
                error = str(e)

            except Exception:
                error = (
                    "Prediction could not be completed. "
                    "Check the input values and model files."
                )

    return render_template(
        "index.html",
        diseases=DISEASES,
        selected_disease=selected_disease,
        title=DISEASES[selected_disease]["title"],
        fields=fields,
        result=result,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)