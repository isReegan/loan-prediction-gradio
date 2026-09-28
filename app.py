import os
import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_logistic_model.pkl")


def predict_loan(income, age, loan_amount):

    data = pd.DataFrame({
        "person_income": [income],
        "person_age": [age],
        "loan_amnt": [loan_amount]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        result = "Loan Approved"
    else:
        result = "Loan Not Approved"

    return result, f"Approval Probability: {probability:.2f}%"


demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Annual Income"),
        gr.Number(label="Age"),
        gr.Number(label="Loan Amount")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability")
    ],
    title="Loan Approval Prediction",
    description="Enter applicant details to predict loan approval."
)

# Render provides the PORT environment variable
port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
