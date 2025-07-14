import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

# ✅ ضبط إعدادات الصفحة
st.set_page_config(page_title="Surgical Complication Predictor", layout="wide")

# ✅ تحميل النماذج والمشفرات
@st.cache_data
def load_models():
    base_path = "models"  # تأكد أن المجلد يحتوي على الملفات التالية:
    with open(os.path.join(base_path, "complication_model.pkl"), "rb") as f:
        model_risk = pickle.load(f)
    with open(os.path.join(base_path, "complication_type_model.pkl"), "rb") as f:
        model_type = pickle.load(f)
    with open(os.path.join(base_path, "recovery_model.pkl"), "rb") as f:
        model_recovery = pickle.load(f)
    with open(os.path.join(base_path, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    return model_risk, model_type, model_recovery, label_encoders

model_risk, model_type, model_recovery, label_encoders = load_models()

st.title("🔍 Surgical Complication Predictor")
st.markdown("أدخل بيانات المريض للتنبؤ بالمضاعفات ومدة التعافي المتوقعة.")

# ✅ الواجهة بلغة مزدوجة
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age / العمر", 0, 100, 30)
    gender = st.selectbox("Gender / الجنس", label_encoders["Gender"].classes_)
    bmi = st.slider("BMI / مؤشر كتلة الجسم", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status / التدخين", label_encoders["Smoking_Status"].classes_)

with col2:
    surgery = st.selectbox("Surgery Type / نوع العملية", label_encoders["Surgery_Type"].classes_)
    duration = st.slider("Surgery Duration (minutes) / مدة العملية بالدقائق", 10, 600, 90)
    condition = st.selectbox("Pre-existing Conditions / أمراض مزمنة", label_encoders["Pre_existing_Conditions"].classes_)

# ✅ تجهيز الداتا للإدخال
input_data = {
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Smoking_Status": smoking,
    "Surgery_Type": surgery,
    "Surgery_Duration_Minutes": duration,
    "Pre_existing_Conditions": condition
}

input_df = pd.DataFrame([input_data])

# ✅ تحويل النصوص إلى أرقام
input_encoded = input_df.copy()
for col in input_encoded.columns:
    if col in label_encoders:
        le = label_encoders[col]
        try:
            input_encoded[col] = le.transform([input_encoded[col]])[0]
        except:
            st.error(f"🚫 Invalid input for '{col}': {input_encoded[col]}")
            st.stop()

# ✅ التنبؤات
if st.button("🔮 Predict"):
    prediction = model_risk.predict(input_encoded)[0]
    complication = model_type.predict(input_encoded)[0]
    recovery_days = model_recovery.predict(input_encoded)[0]

    # ✅ تحويل القيم من أرقام إلى أسماء
    risk_label = "High Risk 🔴" if prediction == 1 else "Low Risk 🟢"
    complication_label = label_encoders["complication_type"].inverse_transform([complication])[0]

    st.subheader("📊 Prediction Result")
    st.success(f"Complication Risk: **{risk_label}**")
    st.info(f"Expected Complication Type: **{complication_label}**")
    st.warning(f"Estimated Recovery Time: **{int(recovery_days)} days**")

    # ✅ رسم بياني: توزيع مدة التعافي
    df = pd.read_csv("surgery_recovery_predection.csv")
    fig = px.histogram(df, x="Recovery_Duration_Days", nbins=30,
                       title="Distribution of Recovery Duration",
                       labels={"Recovery_Duration_Days": "Recovery Days"})
    st.plotly_chart(fig, use_container_width=True)
 