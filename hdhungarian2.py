import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

with open("hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)

df.replace(-9.0, np.NaN, inplace=True)

df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}

df_selected.rename(columns=column_mapping, inplace=True)

columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

df_clean = df_selected.fillna(value=fill_values)
df_clean.drop_duplicates(inplace=True)

X = df_clean.drop("target", axis=1)
y = df_clean['target']

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

model = pickle.load(open("xgb_model.pkl", 'rb'))

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y

# ===================================================================== #

# STREAMLIT
st.set_page_config(
    page_title = "Prediksi Penyakit Jantung",
    page_icon = ":ekg:"
)

st.title("Prediksi Heart Disease Dataset Hungarian")
st.write(f"**_Model's Accuracy_**: :green[**{accuracy}**]% (:red[_Do not copy outright_])")
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
    st.sidebar.header("**User Input** Sidebar")

    age = st.sidebar.number_input(label=":lilac[**Umur**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
    st.sidebar.write(f":blue[Min] value: :blue[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
    st.sidebar.write("")

    sex_sb = st.sidebar.selectbox(label=":lilac[**Jenis Kelamin**]", options=["Pria", "Wanita"])
    st.sidebar.write("")
    st.sidebar.write("")
    if sex_sb == "Pria":
        sex = 1
    elif sex_sb == "Wanita":
        sex = 0
    # -- Value 0: Wanita
    # -- Value 1: Pria
        
    cp_sb = st.sidebar.selectbox(label=":lilac[**Tipe Nyeri Dada**]", options=["Typical Angina", "Atypical Angina", "Non-anginal pain", "Asymptomatic"])
    st.sidebar.write("")
    st.sidebar.write("")
    if cp_sb == "Typical Angina":
        cp = 1
    elif cp_sb == "Atypical Angina":
        cp = 2 
    elif cp_sb == "Non-anginal Pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4
    # -- Value 1: Typical Angina
    # -- Value 2: Atypical Angina
    # -- Value 3: Non-anginal Pain
    # -- Value 4: Asymptomatic
        
    trestbps = st.sidebar.number_input(label=":lilac[**Gula Darah dalam Keadaan Istirahat** (dalam mm Hg saat masuk rumah sakit)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
    st.sidebar.write(f":blue[Min] value: :blue[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
    st.sidebar.write("")

    chol = st.sidebar.number_input(label=":lilac[**Kolesterol Serum** (dalam mg/dl)]", min_value=df_final['chol'].min(),  max_value=df_final['chol'].max())
    st.sidebar.write(f":blue[Min] value: :blue[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
    st.sidebar.write("")

    fbs_sb = st.sidebar.selectbox(label=":lilac[**Gula Darah Puasa > 120 mg/dl**]", options=["False", "True"])
    st.sidebar.write("")
    st.sidebar.write("")
    if fbs_sb == "False":
        fbs = 0
    elif fbs_sb == "True":
        fbs = 1
    # -- Value 0: False
    # -- Value 1: True
        
    restecg_sb = st.sidebar.selectbox(label=":lilac[**Hasil Resting Electrocardiographic**]", options=["Normal", "Mengalami kelainan gelombang ST-T", "Menunjukkan Hipertrofi Ventrikel Kiri"])
    st.sidebar.write("")
    st.sidebar.write("")
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "Mengalami Kelainan gelombang ST-T":
        restecg = 1
    elif restecg_sb == "Menunjukkan Hipertrofi Ventrikel Kiri":
        restecg = 2
    # -- Value 0: Normal
    # -- Value 1: Mengalami Kelainan Gelombang ST-T
    # -- Value 2: Menunjukkan Hipertrofi Ventrikel Kiri
    
    thalach = st.sidebar.number_input(label=":lilac[**Denyut Jantung Maksimum yang dicapai**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
    st.sidebar.write(f":blue[Min] value: :blue[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
    st.sidebar.write("")

    exang_sb = st.sidebar.selectbox(label=":lilac[**Angina akibat olahraga**]", options=["Tidak", "Iya"])
    st.sidebar.write("")
    st.sidebar.write("")
    if exang_sb == "Tidak":
        exang = 0
    elif exang_sb == "Iya":
        exang = 1
    # -- Value 0: Tidak
    # -- Value 1: Iya
        
    oldpeak = st.sidebar.number_input(label=":lilac[**Depresi ST disebabkan oleh olahraga dibandingkan istirahat**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
    st.sidebar.write(f":blue[Min] value: :blue[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
    st.sidebar.write("")

    data = {
        'Age': age,
        'Sex': sex_sb,
        'Chest pain type': cp_sb,
        'RPB': f"{trestbps} mm Hg",
        'Serum Cholestoral': f"{chol} mg/dl",
        'FBS > 120 mg/dl': fbs_sb,
        'Resting ECG': restecg_sb,
        'Maximum Heart Rate': thalach,
        'Exercise induced angina': exang_sb,
        'ST depression': oldpeak,
    }

    preview_df = pd.DataFrame(data, index=['input'])

    st.header("Input Pengguna sebagai DataFrame")
    st.write("")
    st.dataframe(preview_df.iloc[:, :6])
    st.write("")
    st.dataframe(preview_df.iloc[:, :6])
    st.write("")

    result = ":lilac[-]"

    predict_btn = st.button("**Prediksi**", type="primary")

    st.write("")
    if predict_btn:
        inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
        prediction = model.predict(inputs)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty

        if prediction == 0:
            result = ":pink[**Sehat**]"
        elif prediction == 1:
            result = ":yellow[**Sakit Jantung Level 1**]"
        elif prediction == 2:
            result = ":tangerine[**Sakit Jantung Level 2**]"
        elif prediction == 3:
            result = ":orange[**Sakit Jantung Level 3**]"
        elif prediction == 4:
            result = ":red[**Sakit Jantung Level 4**]"
    
    st.write("")
    st.write("")
    st.subheader("Prediksi:")
    st.subheader(result)

with tab2:
    st.header("Predict Multiple Data:")

    sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload file CSV", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        prediction_arr = model.predict(uploaded_df)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            if prediction == 0:
                result = "Sehat"
            elif prediction == 1:
                result = "Sakit Jantung Level 1"
            elif prediction == 2:
                result = "Sakit Jantung Level 2"
            elif prediction == 3:
                result = "Sakit Jantung Level 3"
            elif prediction == 4:
                result = "Sakit Jantung Level 4"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)