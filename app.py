import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import plotly.express as px

st.set_page_config(page_title="Surgical Complication Predictor", layout="centered")

@st.cache_resource
def load_models():
    base_path = "C:/Users/CompuGiza/3D Objects/prooject"
    with open(os.path.join(base_path, "complication_model.pkl"), "rb") as f:
        model_risk = pickle.load(f)
    with open(os.path.join(base_path, "complication_type_model.pkl"), "rb") as f:
        model_type = pickle.load(f)
    with open(os.path.join(base_path, "recovery_model.pkl"), "rb") as f:
        model_recovery = pickle.load(f)
    with open(os.path.join(base_path, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(base_path, "complication_type_encoder.pkl"), "rb") as f:
        type_encoder = pickle.load(f)
    return model_risk, model_type, model_recovery, label_encoders, type_encoder

model_risk, model_type, model_recovery, label_encoders, type_encoder = load_models()

language = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])

texts = {
    "English": {
        "title": "Surgical Complication Predictor",
        "input_section": "Patient Information",
        "predict_btn": "Predict",
        "result_section": "Prediction Result",
        "complication": "Complication Risk:",
        "type": "Complication Type:",
        "recovery": "Estimated Recovery Days:",
        "progress": "Analyzing Patient Information..."
    },
    "العربية": {
        "title": "نظام التنبؤ بالمضاعفات الجراحية",
        "input_section": "معلومات المريض",
        "predict_btn": "توقّع",
        "result_section": "نتيجة التنبؤ",
        "complication": "هل يوجد مضاعفات:",
        "type": "نوع المضاعفة:",
        "recovery": "مدة التعافي المتوقعة (أيام):",
        "progress": "جارٍ تحليل بيانات المريض..."
    }
}

T = texts[language]

st.title(T["title"])
st.subheader(T["input_section"])

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age / العمر", 18, 100, 30)
    gender = st.selectbox("Gender / الجنس", label_encoders["Gender"].classes_)
    bmi = st.number_input("BMI / مؤشر كتلة الجسم", 10.0, 60.0, 22.0)
    smoking = st.selectbox("Smoking Status / التدخين", label_encoders["Smoking_Status"].classes_)
    conditions = st.selectbox("Pre-existing Conditions / أمراض مزمنة", label_encoders["Pre_existing_Conditions"].classes_)

with col2:
    surgery_type = st.selectbox("Surgery Type / نوع الجراحة", label_encoders["Surgery_Type"].classes_)
    duration = st.slider("Surgery Duration (Minutes) / مدة العملية", 10, 1000, 90)

if st.button(T["predict_btn"]):
    with st.spinner(T["progress"]):
        input_data = {
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "Smoking_Status": smoking,
            "Surgery_Type": surgery_type,
            "Surgery_Duration_Minutes": duration,
            "Pre_existing_Conditions": conditions
        }

        input_encoded = input_data.copy()
        try:
            for col, encoder in label_encoders.items():
                if col in input_encoded:
                    input_encoded[col] = encoder.transform([input_encoded[col]])[0]
        except ValueError as ve:
            st.error(f"🚫 Invalid input for '{col}': {input_encoded[col]}")
            st.stop()

        input_df = pd.DataFrame([input_encoded])

        prediction = model_risk.predict(input_df)[0]
        complication_type = model_type.predict(input_df)[0] if prediction == 1 else "None"
        recovery_days = model_recovery.predict(input_df)[0]
        complication_type = type_encoder.inverse_transform([complication_type])[0] if prediction == 1 else "None"

        st.subheader(T["result_section"])
        st.success(f"{T['complication']} {'✅ Yes' if prediction == 1 else '❌ No'}")

        if prediction == 1:
            st.info(f"{T['type']} {complication_type}")

        st.write(f"{T['recovery']} {int(round(recovery_days))} يوم")

        # ✅ رسم بياني لتوضيح النتائج
        fig = px.bar(
            x=["Recovery Days"],
            y=[recovery_days],
            color_discrete_sequence=["#2E86C1"],
            labels={"x": "", "y": "Days"},
            title="Estimated Recovery Time"
        )
        fig.update_layout(yaxis_range=[0, max(20, int(recovery_days) + 5)])
        st.plotly_chart(fig)
