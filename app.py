import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

st.set_page_config(page_title="Loan Approval Prediction & EDA", layout="wide")
st.title("Loan Approval Prediction & Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("LoanApprovalPrediction.csv")

data = load_data()

with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Single Prediction ---
st.header("Predict Loan Approval for Single Applicant")
with st.form("loan_form", clear_on_submit=False):
    cols = st.columns(3)
    gender = cols[0].selectbox("Gender", ("Male", "Female"))
    married = cols[1].selectbox("Married", ("Yes", "No"))
    dependents = cols[2].selectbox("Dependents", ("0", "1", "2", "3+"))
    education = cols[0].selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = cols[1].selectbox("Self Employed", ("Yes", "No"))
    applicant_income = cols[2].number_input("Applicant Income", min_value=0)
    coapplicant_income = cols[0].number_input("Coapplicant Income", min_value=0)
    loan_amount = cols[1].number_input("Loan Amount (K)", min_value=0)
    loan_term = cols[2].selectbox("Loan Term (months)", (360, 120, 240, 180, 60))
    credit_history = cols[0].selectbox("Credit History", (1.0, 0.0))
    property_area = cols[1].selectbox("Property Area", ("Urban", "Rural", "Semiurban"))
    submit = st.form_submit_button("Predict Loan Approval")

def preprocess_input():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 1 if education == "Graduate" else 0
    self_emp_val = 1 if self_employed == "Yes" else 0
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_val = property_map[property_area]
    dependents_val = 3 if dependents == "3+" else int(dependents)

    return np.array([[gender_val, married_val, dependents_val, education_val,
                      self_emp_val, applicant_income, coapplicant_income,
                      loan_amount, loan_term, credit_history, property_val]])

if submit:
    input_data = preprocess_input()
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.subheader(f"Result: Loan {result} {'✅' if result == 'Approved' else '❌'}")

    pred_report = pd.DataFrame([{
        'Gender': gender, 'Married': married, 'Dependents': dependents,
        'Education': education, 'Self Employed': self_employed,
        'Applicant Income': applicant_income, 'Coapplicant Income': coapplicant_income,
        'Loan Amount': loan_amount, 'Loan Term': loan_term,
        'Credit History': credit_history, 'Property Area': property_area,
        'Prediction': result
    }])
    csv = pred_report.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="loan_prediction.csv">📥 Download Prediction Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- Batch Prediction ---
st.markdown("---")
st.header("Batch Prediction by CSV Upload")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
use_demo = st.button("Use Demo Dataset")

def batch_preprocess(df):
    df['Dependents'] = df['Dependents'].replace('3+', 3)
    df['Dependents'] = df['Dependents'].fillna(0).astype(int)
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    df['Married'] = df['Married'].map({'Yes':1, 'No':0})
    df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
    df['Property_Area'] = df['Property_Area'].map({'Urban':2, 'Semiurban':1, 'Rural':0})
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    features = ['Gender','Married','Dependents','Education','Self_Employed',
                'ApplicantIncome','CoapplicantIncome','LoanAmount',
                'Loan_Amount_Term','Credit_History','Property_Area']
    return df[features]

def predict_batch(df):
    processed = batch_preprocess(df)
    preds = model.predict(processed)
    df['Prediction'] = np.where(preds == 1, 'Approved', 'Rejected')
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Sample")
    st.dataframe(df.head())

    if st.button("Predict on Uploaded Data"):
        pred_df = predict_batch(df)
        st.write("### Prediction Results")
        st.dataframe(pred_df.head())

        csv = pred_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="loan_predictions.csv">📥 Download All Predictions</a>'
        st.markdown(href, unsafe_allow_html=True)

# --- EDA Section triggered by demo data only ---
if use_demo:
    st.write("Using demo dataset for batch prediction")
    pred_df = predict_batch(data)
    st.dataframe(pred_df.head())


    # --- EDA Section ---
    st.markdown("---")
    st.header("Exploratory Data Analysis")

    # # Correlation Table with Gradient (Keep this only)
    # st.markdown("#### Correlation Table")
    # corr_df = data.corr(numeric_only=True).round(3)
    # st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'))

    # Loan Status distribution
    st.markdown("#### Loan Status Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Loan_Status', data=data, palette='Set2', ax=ax2)
    st.pyplot(fig2)

    # Applicant Income distribution
    st.markdown("#### Applicant Income Distribution")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.histplot(data['ApplicantIncome'], kde=True, bins=30, ax=ax3)
    st.pyplot(fig3)

    # Loan Status by Education
    st.markdown("#### Loan Status by Education")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x='Education', hue='Loan_Status', palette='pastel', ax=ax4)
    st.pyplot(fig4)

    # Loan Approval by Credit History
    st.markdown("#### Loan Approval by Credit History")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x='Credit_History', hue='Loan_Status', palette='Set1', ax=ax5)
    st.pyplot(fig5)

    # Loan Amount vs Loan Status
    st.markdown("#### Loan Amount vs Loan Status")
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Loan_Status', y='LoanAmount', data=data, palette='Accent', ax=ax6)
    st.pyplot(fig6)

    # Property Area vs Loan Status
    st.markdown("#### Property Area vs Loan Status")
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x='Property_Area', hue='Loan_Status', palette='cool', ax=ax7)
    st.pyplot(fig7)

    # Loan Approval by Gender (Pie Chart)
    st.markdown("#### Loan Approval by Gender")
    approval_by_gender = pd.crosstab(data['Gender'], data['Loan_Status'], normalize='index')['Y']
    fig8, ax8 = plt.subplots(figsize=(4, 4))
    approval_by_gender.plot.pie(autopct='%1.1f%%', labels=['Female', 'Male'],
                                colors=['#ff9999','#66b3ff'], startangle=90, ax=ax8)
    ax8.set_ylabel('')
    ax8.set_title('Loan Approval Rate by Gender')
    st.pyplot(fig8)

    st.markdown("#### Correlation Table")
    corr_df = data.corr(numeric_only=True).round(3)
    st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'))


    # Feature Importance (if model supports it)
    if hasattr(model, "feature_importances_"):
        st.markdown("#### Feature Importance")
        feat_importance = pd.Series(model.feature_importances_, index=[
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]).sort_values(ascending=False)

        fig9, ax9 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax9, palette='viridis')
        ax9.set_title("Feature Importance")
        st.pyplot(fig9)
    else:
        st.info("Model does not support feature importance visualization.")














    # st.markdown("#### Correlation Heatmap")
    # fig1, ax1 = plt.subplots(figsize=(10,6))
    # sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax1)
    # st.pyplot(fig1)

    # st.markdown("#### Loan Status Distribution")
    # fig2, ax2 = plt.subplots()
    # sns.countplot(x='Loan_Status', data=data, palette='Set2', ax=ax2)
    # st.pyplot(fig2)

    # st.markdown("#### Applicant Income Distribution")
    # fig3, ax3 = plt.subplots()
    # sns.histplot(data['ApplicantIncome'], kde=True, bins=30, ax=ax3)
    # st.pyplot(fig3)

    # st.markdown("#### Loan Status by Education")
    # fig4, ax4 = plt.subplots()
    # sns.countplot(data=data, x='Education', hue='Loan_Status', palette='pastel', ax=ax4)
    # st.pyplot(fig4)

    # st.markdown("#### Loan Approval by Credit History")
    # fig5, ax5 = plt.subplots()
    # sns.countplot(data=data, x='Credit_History', hue='Loan_Status', palette='Set1', ax=ax5)
    # st.pyplot(fig5)

    # st.markdown("#### Loan Amount vs Loan Status")
    # fig6, ax6 = plt.subplots()
    # sns.boxplot(x='Loan_Status', y='LoanAmount', data=data, palette='Accent', ax=ax6)
    # st.pyplot(fig6)

    # st.markdown("#### Property Area vs Loan Status")
    # fig7, ax7 = plt.subplots()
    # sns.countplot(data=data, x='Property_Area', hue='Loan_Status', palette='cool', ax=ax7)
    # st.pyplot(fig7)

    # st.markdown("#### Loan Approval by Gender")
    # approval_by_gender = pd.crosstab(data['Gender'], data['Loan_Status'], normalize='index')['Y']
    # fig8, ax8 = plt.subplots()
    # approval_by_gender.plot.pie(autopct='%1.1f%%', labels=['Female', 'Male'], colors=['#ff9999','#66b3ff'], startangle=90, ax=ax8)
    # ax8.set_ylabel('')
    # ax8.set_title('Loan Approval Rate by Gender')
    # st.pyplot(fig8)

    # st.markdown("#### Correlation Table")
    # corr_df = data.corr(numeric_only=True).round(3)
    # st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'))

    # if hasattr(model, "feature_importances_"):
    #     st.markdown("#### Feature Importance")
    #     feat_importance = pd.Series(model.feature_importances_, index=[
    #         'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    #         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    #         'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    #     ]).sort_values(ascending=False)

    #     fig9, ax9 = plt.subplots()
    #     sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax9, palette='viridis')
    #     ax9.set_title("Feature Importance")
    #     st.pyplot(fig9)
    # else:
    #     st.info("Model does not support feature importance visualization.")












# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64

# st.set_page_config(page_title="Loan Approval Prediction & EDA", layout="wide")

# st.title("Loan Approval Prediction & Exploratory Data Analysis")

# # Load your dataset and model
# @st.cache_data
# def load_data():
#     return pd.read_csv("LoanApprovalPrediction.csv")

# data = load_data()

# with open('loan_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # --- Single prediction form ---
# st.header("Predict Loan Approval for Single Applicant")

# with st.form("loan_form", clear_on_submit=False):
#     cols = st.columns(3)
#     gender = cols[0].selectbox("Gender", ("Male", "Female"))
#     married = cols[1].selectbox("Married", ("Yes", "No"))
#     dependents = cols[2].selectbox("Dependents", ("0", "1", "2", "3+"))
#     education = cols[0].selectbox("Education", ("Graduate", "Not Graduate"))
#     self_employed = cols[1].selectbox("Self Employed", ("Yes", "No"))
#     applicant_income = cols[2].number_input("Applicant Income", min_value=0)
#     coapplicant_income = cols[0].number_input("Coapplicant Income", min_value=0)
#     loan_amount = cols[1].number_input("Loan Amount (K)", min_value=0)
#     loan_term = cols[2].selectbox("Loan Term (months)", (360, 120, 240, 180, 60))
#     credit_history = cols[0].selectbox("Credit History", (1.0, 0.0))
#     property_area = cols[1].selectbox("Property Area", ("Urban", "Rural", "Semiurban"))
#     submit = st.form_submit_button("Predict Loan Approval")

# def preprocess_input():
#     gender_val = 1 if gender == "Male" else 0
#     married_val = 1 if married == "Yes" else 0
#     education_val = 1 if education == "Graduate" else 0
#     self_emp_val = 1 if self_employed == "Yes" else 0
#     property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
#     property_val = property_map[property_area]
#     dependents_val = 3 if dependents == "3+" else int(dependents)

#     return np.array([[gender_val, married_val, dependents_val, education_val,
#                       self_emp_val, applicant_income, coapplicant_income,
#                       loan_amount, loan_term, credit_history, property_val]])

# if submit:
#     input_data = preprocess_input()
#     prediction = model.predict(input_data)
#     result = "Approved" if prediction[0] == 1 else "Rejected"
#     st.subheader(f"Result: Loan {result} {'✅' if result == 'Approved' else '❌'}")

#     # Show downloadable CSV report
#     pred_report = pd.DataFrame([{
#         'Gender': gender, 'Married': married, 'Dependents': dependents,
#         'Education': education, 'Self Employed': self_employed,
#         'Applicant Income': applicant_income, 'Coapplicant Income': coapplicant_income,
#         'Loan Amount': loan_amount, 'Loan Term': loan_term,
#         'Credit History': credit_history, 'Property Area': property_area,
#         'Prediction': result
#     }])
#     csv = pred_report.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="loan_prediction.csv">📥 Download Prediction Report</a>'
#     st.markdown(href, unsafe_allow_html=True)

# # --- Batch prediction ---
# st.markdown("---")
# st.header("Batch Prediction by CSV Upload")

# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
# use_demo = st.button("Use Demo Dataset")

# def batch_preprocess(df):
#     df['Dependents'] = df['Dependents'].replace('3+', 3)
#     df['Dependents'] = df['Dependents'].fillna(0).astype(int)
#     df = df.copy()
#     df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
#     df['Married'] = df['Married'].map({'Yes':1, 'No':0})
#     df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
#     df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
#     df['Property_Area'] = df['Property_Area'].map({'Urban':2, 'Semiurban':1, 'Rural':0})
#     df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
#     features = ['Gender','Married','Dependents','Education','Self_Employed',
#                 'ApplicantIncome','CoapplicantIncome','LoanAmount',
#                 'Loan_Amount_Term','Credit_History','Property_Area']
#     return df[features]

# def predict_batch(df):
#     processed = batch_preprocess(df)
#     preds = model.predict(processed)
#     df['Prediction'] = np.where(preds == 1, 'Approved', 'Rejected')
#     return df

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("### Uploaded Data Sample")
#     st.dataframe(df.head())

#     if st.button("Predict on Uploaded Data"):
#         pred_df = predict_batch(df)
#         st.write("### Prediction Results")
#         st.dataframe(pred_df.head())

#         csv = pred_df.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         href = f'<a href="data:file/csv;base64,{b64}" download="loan_predictions.csv">📥 Download All Predictions</a>'
#         st.markdown(href, unsafe_allow_html=True)

# elif use_demo:
#     st.write("Using demo dataset for batch prediction")
#     pred_df = predict_batch(data)
#     st.dataframe(pred_df.head())

# # --- EDA Section ---
# st.markdown("---")
# st.header("Exploratory Data Analysis")

# # Correlation heatmap
# st.markdown("#### Correlation Heatmap")
# fig1, ax1 = plt.subplots(figsize=(10,6))
# sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax1)
# st.pyplot(fig1)

# # Loan Status distribution
# st.markdown("#### Loan Status Distribution")
# fig2, ax2 = plt.subplots()
# sns.countplot(x='Loan_Status', data=data, palette='Set2', ax=ax2)
# st.pyplot(fig2)

# # Applicant Income distribution
# st.markdown("#### Applicant Income Distribution")
# fig3, ax3 = plt.subplots()
# sns.histplot(data['ApplicantIncome'], kde=True, bins=30, ax=ax3)
# st.pyplot(fig3)

# # Loan Status by Education
# st.markdown("#### Loan Status by Education")
# fig4, ax4 = plt.subplots()
# sns.countplot(data=data, x='Education', hue='Loan_Status', palette='pastel', ax=ax4)
# st.pyplot(fig4)

# # Loan Approval by Credit History
# st.markdown("#### Loan Approval by Credit History")
# fig5, ax5 = plt.subplots()
# sns.countplot(data=data, x='Credit_History', hue='Loan_Status', palette='Set1', ax=ax5)
# st.pyplot(fig5)

# # Loan Amount vs Loan Status
# st.markdown("#### Loan Amount vs Loan Status")
# fig6, ax6 = plt.subplots()
# sns.boxplot(x='Loan_Status', y='LoanAmount', data=data, palette='Accent', ax=ax6)
# st.pyplot(fig6)

# # Property Area vs Loan Status
# st.markdown("#### Property Area vs Loan Status")
# fig7, ax7 = plt.subplots()
# sns.countplot(data=data, x='Property_Area', hue='Loan_Status', palette='cool', ax=ax7)
# st.pyplot(fig7)

# # Loan Approval by Gender
# st.markdown("#### Loan Approval by Gender")
# approval_by_gender = pd.crosstab(data['Gender'], data['Loan_Status'], normalize='index')['Y']
# fig8, ax8 = plt.subplots()
# approval_by_gender.plot.pie(autopct='%1.1f%%', labels=['Female', 'Male'], colors=['#ff9999','#66b3ff'], startangle=90, ax=ax8)
# ax8.set_ylabel('')
# ax8.set_title('Loan Approval Rate by Gender')
# st.pyplot(fig8)

# # Correlation Table with Gradient
# st.markdown("#### Correlation Table")
# corr_df = data.corr(numeric_only=True).round(3)
# st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'))

# # Feature importance (if model supports)
# if hasattr(model, "feature_importances_"):
#     st.markdown("#### Feature Importance")
#     feat_importance = pd.Series(model.feature_importances_, index=[
#         'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
#         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#         'Loan_Amount_Term', 'Credit_History', 'Property_Area'
#     ]).sort_values(ascending=False)

#     fig9, ax9 = plt.subplots()
#     sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax9, palette='viridis')
#     ax9.set_title("Feature Importance")
#     st.pyplot(fig9)
# else:
#     st.info("Model does not support feature importance visualization.")






# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64

# st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")

# # --- Title and Description ---
# st.title("Loan Approval Prediction & Dashboard")
# st.markdown("""
# This app predicts whether a loan will be approved or rejected based on applicant details.  
# You can input individual details to check loan eligibility or upload a dataset to analyze multiple records at once.
# """)

# # Load trained model
# with open('loan_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# @st.cache_data
# def load_data():
#     return pd.read_csv("LoanApprovalPrediction.csv")

# data = load_data()

# # --- Input form for single prediction ---
# with st.form("loan_form", clear_on_submit=False):
#     st.subheader("Applicant Details")
    
#     cols = st.columns(3)
#     gender = cols[0].selectbox("Gender", ("Male", "Female"))
#     married = cols[1].selectbox("Married", ("Yes", "No"))
#     dependents = cols[2].selectbox("Dependents", ("0", "1", "2", "3+"))

#     education = cols[0].selectbox("Education", ("Graduate", "Not Graduate"))
#     self_employed = cols[1].selectbox("Self Employed", ("Yes", "No"))
#     applicant_income = cols[2].number_input("Applicant Income", min_value=0)

#     coapplicant_income = cols[0].number_input("Coapplicant Income", min_value=0)
#     loan_amount = cols[1].number_input("Loan Amount (K)", min_value=0)
#     loan_term = cols[2].selectbox("Loan Term (months)", (360, 120, 240, 180, 60))

#     credit_history = cols[0].selectbox("Credit History", (1.0, 0.0))
#     property_area = cols[1].selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

#     submit_button = st.form_submit_button("Predict Loan Approval")

# def preprocess():
#     gender_val = 1 if gender == "Male" else 0
#     married_val = 1 if married == "Yes" else 0
#     education_val = 1 if education == "Graduate" else 0
#     self_emp_val = 1 if self_employed == "Yes" else 0
#     property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
#     property_val = property_map[property_area]
#     dependents_val = 3 if dependents == "3+" else int(dependents)

#     return np.array([[gender_val, married_val, dependents_val, education_val,
#                       self_emp_val, applicant_income, coapplicant_income,
#                       loan_amount, loan_term, credit_history, property_val]])

# if submit_button:
#     input_data = preprocess()
#     prediction = model.predict(input_data)
#     result = "Approved" if prediction[0] == 1 else "Rejected"

#     if result == "Approved":
#         st.success(f"🎉 Loan {result}!")
#     else:
#         st.error(f"❌ Loan {result}.")

#     result_df = pd.DataFrame([{
#         'Gender': gender, 'Married': married, 'Dependents': dependents,
#         'Education': education, 'Self Employed': self_employed,
#         'Applicant Income': applicant_income, 'Coapplicant Income': coapplicant_income,
#         'Loan Amount': loan_amount, 'Loan Term': loan_term,
#         'Credit History': credit_history, 'Property Area': property_area,
#         'Prediction': result
#     }])

#     csv = result_df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="loan_prediction.csv">📥 Download Prediction Report</a>'
#     st.markdown(href, unsafe_allow_html=True)

# # --- Bulk prediction by uploading CSV ---
# st.markdown("---")
# st.subheader("Upload CSV file for batch prediction")

# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
# use_demo = st.button("Use Demo Dataset")

# def batch_preprocess(df):
#     # Map categorical variables same as in training
#     df = df.copy()
#     df['Dependents'] = df['Dependents'].replace('3+', 3)
#     df['Dependents'] = df['Dependents'].fillna(0).astype(int)
#     df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
#     df['Married'] = df['Married'].map({'Yes':1, 'No':0})
#     df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
#     df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
#     df['Property_Area'] = df['Property_Area'].map({'Urban':2, 'Semiurban':1, 'Rural':0})
#     df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    
#     features = ['Gender','Married','Dependents','Education','Self_Employed',
#                 'ApplicantIncome','CoapplicantIncome','LoanAmount',
#                 'Loan_Amount_Term','Credit_History','Property_Area']
#     return df[features]

# def predict_batch(df):
#     processed = batch_preprocess(df)
#     preds = model.predict(processed)
#     df['Prediction'] = np.where(preds == 1, 'Approved', 'Rejected')
#     return df

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("### Uploaded Data")
#     st.dataframe(df.head())

#     if st.button("Predict on Uploaded Data"):
#         pred_df = predict_batch(df)
#         st.write("### Prediction Results")
#         st.dataframe(pred_df.head())

#         csv = pred_df.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         href = f'<a href="data:file/csv;base64,{b64}" download="loan_predictions.csv">📥 Download All Predictions</a>'
#         st.markdown(href, unsafe_allow_html=True)

# elif use_demo:
#     st.write("Using demo dataset")
#     pred_df = predict_batch(data)
#     st.write(pred_df.head())

# # --- EDA Section ---
# st.markdown("---")
# st.header("Exploratory Data Analysis")

# def plot_histogram(df, column):
#     plt.figure(figsize=(6,4))
#     sns.histplot(df[column].dropna(), kde=True, color='skyblue')
#     plt.title(f'Distribution of {column}')
#     st.pyplot(plt.gcf())

# def plot_barplot(df, column):
#     plt.figure(figsize=(6,4))
#     sns.countplot(data=df, x=column, palette='viridis')
#     plt.title(f'Count of {column}')
#     st.pyplot(plt.gcf())

# def plot_correlation(df):
#     numeric_df = df.select_dtypes(include=['number'])  # only numeric columns
#     corr = numeric_df.corr()
#     plt.figure(figsize=(8,6))
#     corr = df.corr()
#     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#     plt.title("Correlation Heatmap")
#     st.pyplot(plt.gcf())

# # Select which data to analyze for EDA
# eda_data = None
# if uploaded_file is not None:
#     eda_data = df
# elif use_demo:
#     eda_data = data

# if eda_data is not None:
#     st.subheader("Dataset Summary")
#     st.write(eda_data.describe(include='all'))

#     num_cols = eda_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     cat_cols = eda_data.select_dtypes(include=['object']).columns.tolist()

#     st.subheader("Numerical Variable Distributions")
#     for col in num_cols:
#         plot_histogram(eda_data, col)

#     st.subheader("Categorical Variable Counts")
#     for col in cat_cols:
#         plot_barplot(eda_data, col)

#     st.subheader("Correlation Heatmap")
#     plot_correlation(eda_data)

# else:
#     st.info("Upload a dataset or use the demo dataset to see EDA visualizations here.")






# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64

# st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")
# st.title("Loan Approval Prediction & Dashboard")

# # Load trained model
# with open('loan_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Load dataset for EDA
# @st.cache_data
# def load_data():
#     return pd.read_csv("LoanApprovalPrediction.csv")  # Rename to your dataset filename

# data = load_data()

# # Top KPIs
# col1, col2, col3 = st.columns(3)
# col1.metric("Total Applicants", data.shape[0])
# col2.metric("Loan Approved %", f"{(data['Loan_Status'] == 'Y').mean()*100:.2f}%")
# col3.metric("Avg Loan Amt (K)", f"{data['LoanAmount'].mean():.2f}")

# # Sidebar inputs
# st.sidebar.header("Applicant Details")
# gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
# married = st.sidebar.selectbox("Married", ("Yes", "No"))
# dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
# education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
# self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
# applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
# coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
# loan_amount = st.sidebar.number_input("Loan Amount (K)", min_value=0)
# loan_term = st.sidebar.selectbox("Loan Term (months)", (360, 120, 240, 180, 60))
# credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
# property_area = st.sidebar.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

# # Data pre-processing for prediction
# def preprocess():
#     gender_val = 1 if gender == "Male" else 0
#     married_val = 1 if married == "Yes" else 0
#     education_val = 1 if education == "Graduate" else 0
#     self_emp_val = 1 if self_employed == "Yes" else 0
#     property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
#     property_val = property_map[property_area]
#     dependents_val = 3 if dependents == "3+" else int(dependents)

#     return np.array([[gender_val, married_val, dependents_val, education_val,
#                       self_emp_val, applicant_income, coapplicant_income,
#                       loan_amount, loan_term, credit_history, property_val]])

# # Prediction section
# if st.button("Predict Loan Approval"):
#     input_data = preprocess()
#     prediction = model.predict(input_data)
#     result = "Approved" if prediction[0] == 1 else "Rejected"

#     st.subheader(f"Result: Loan {result} ✅" if result == "Approved" else f"Result: Loan {result} ❌")

#     # Display prediction report and download
#     result_df = pd.DataFrame([{
#         'Gender': gender, 'Married': married, 'Dependents': dependents,
#         'Education': education, 'Self Employed': self_employed,
#         'Applicant Income': applicant_income, 'Coapplicant Income': coapplicant_income,
#         'Loan Amount': loan_amount, 'Loan Term': loan_term,
#         'Credit History': credit_history, 'Property Area': property_area,
#         'Prediction': result
#     }])

#     csv = result_df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="loan_prediction.csv">📥 Download Prediction Report</a>'
#     st.markdown(href, unsafe_allow_html=True)

# # EDA Section
# st.subheader("Exploratory Data Analysis")

# # Correlation heatmap
# st.markdown("#### Correlation Heatmap")
# fig1, ax1 = plt.subplots(figsize=(10,6))
# sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax1)
# st.pyplot(fig1)

# # Approval status count
# st.markdown("#### Loan Status Distribution")
# fig2, ax2 = plt.subplots()
# sns.countplot(x='Loan_Status', data=data, palette='Set2', ax=ax2)
# st.pyplot(fig2)

# # Income distribution
# st.markdown("#### Applicant Income Distribution")
# fig3, ax3 = plt.subplots()
# sns.histplot(data['ApplicantIncome'], kde=True, bins=30, ax=ax3)
# st.pyplot(fig3)

# # Loan status by education
# st.markdown("#### Loan Status by Education")
# fig4, ax4 = plt.subplots()
# sns.countplot(data=data, x='Education', hue='Loan_Status', palette='pastel', ax=ax4)
# st.pyplot(fig4)

# # Loan approval by credit history
# st.markdown("#### Loan Approval by Credit History")
# fig5, ax5 = plt.subplots()
# sns.countplot(data=data, x='Credit_History', hue='Loan_Status', palette='Set1', ax=ax5)
# st.pyplot(fig5)

# # Loan amount vs loan status
# st.markdown("#### Loan Amount vs Loan Status")
# fig6, ax6 = plt.subplots()
# sns.boxplot(x='Loan_Status', y='LoanAmount', data=data, palette='Accent', ax=ax6)
# st.pyplot(fig6)

# # Property area vs loan status
# st.markdown("#### Property Area vs Loan Status")
# fig7, ax7 = plt.subplots()
# sns.countplot(data=data, x='Property_Area', hue='Loan_Status', palette='cool', ax=ax7)
# st.pyplot(fig7)

# # Loan approval by gender
# st.markdown("#### Loan Approval by Gender")
# approval_by_gender = pd.crosstab(data['Gender'], data['Loan_Status'], normalize='index')['Y']
# fig8, ax8 = plt.subplots()
# approval_by_gender.plot.pie(autopct='%1.1f%%', labels=['Female', 'Male'], colors=['#ff9999','#66b3ff'], ax=ax8)
# ax8.set_ylabel('')
# st.pyplot(fig8)

# # Correlation table
# st.markdown("#### Feature Correlation Table")
# st.dataframe(data.corr(numeric_only=True).style.background_gradient(cmap='coolwarm'))

# # Feature importance (if model has attribute)
# if hasattr(model, 'feature_importances_'):
#     st.subheader("Feature Importance")
#     importance = model.feature_importances_
#     feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self Employed', 
#                      'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
#                      'Loan_Amount_Term', 'Credit_History', 'Property_Area']
#     imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
#     imp_df = imp_df.sort_values(by='Importance', ascending=False)
#     st.bar_chart(imp_df.set_index('Feature'))

# # Sidebar About section
# st.sidebar.markdown("---")
# st.sidebar.markdown("ℹ️ **About App**")
# st.sidebar.info("""
# This app predicts loan approval using a machine learning model trained on historical data. 
# Includes data analysis and visualization to extract meaningful insights.

# Built by: Your Name  
# Dataset: LoanApprovalPrediction.csv  
# """)





# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64


# st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")
# st.title("Loan Approval Prediction & Dashboard")

# # Load trained model
# with open('loan_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Load dataset for EDA
# @st.cache_data
# def load_data():
#     return pd.read_csv("LoanApprovalPrediction.csv")  # Rename to your dataset filename

# data = load_data()


# # Top KPIs
# col1, col2, col3 = st.columns(3)
# col1.metric("Total Applicants", data.shape[0])
# col2.metric("Loan Approved %", f"{(data['Loan_Status'] == 'Y').mean()*100:.2f}%")
# col3.metric("Avg Loan Amt (K)", f"{data['LoanAmount'].mean():.2f}")

# # Sidebar inputs
# st.sidebar.header("Applicant Details")
# gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
# married = st.sidebar.selectbox("Married", ("Yes", "No"))
# dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
# education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
# self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
# applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
# coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
# loan_amount = st.sidebar.number_input("Loan Amount (K)", min_value=0)
# loan_term = st.sidebar.selectbox("Loan Term (months)", (360, 120, 240, 180, 60))
# credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
# property_area = st.sidebar.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

# # Data pre-processing for prediction
# def preprocess():
#     gender_val = 1 if gender == "Male" else 0
#     married_val = 1 if married == "Yes" else 0
#     education_val = 1 if education == "Graduate" else 0
#     self_emp_val = 1 if self_employed == "Yes" else 0
#     property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
#     property_val = property_map[property_area]
#     dependents_val = 3 if dependents == "3+" else int(dependents)

#     return np.array([[gender_val, married_val, dependents_val, education_val,
#                       self_emp_val, applicant_income, coapplicant_income,
#                       loan_amount, loan_term, credit_history, property_val]])

# # Prediction section
# if st.button("Predict Loan Approval"):
#     input_data = preprocess()
#     prediction = model.predict(input_data)
#     result = "Approved" if prediction[0] == 1 else "Rejected"

#     st.subheader(f"Result: Loan {result} ✅" if result == "Approved" else f"Result: Loan {result} ❌")

#     # Display prediction report and download
#     result_df = pd.DataFrame([{
#         'Gender': gender, 'Married': married, 'Dependents': dependents,
#         'Education': education, 'Self Employed': self_employed,
#         'Applicant Income': applicant_income, 'Coapplicant Income': coapplicant_income,
#         'Loan Amount': loan_amount, 'Loan Term': loan_term,
#         'Credit History': credit_history, 'Property Area': property_area,
#         'Prediction': result
#     }])

#     csv = result_df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="loan_prediction.csv">📥 Download Prediction Report</a>'
#     st.markdown(href, unsafe_allow_html=True)

# # EDA Section
# st.subheader("Exploratory Data Analysis")

# # Correlation heatmap
# st.markdown("#### Correlation Heatmap")
# fig1, ax1 = plt.subplots(figsize=(10,6))
# sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax1)
# st.pyplot(fig1)

# # Approval status count
# st.markdown("#### Loan Status Distribution")
# fig2, ax2 = plt.subplots()
# sns.countplot(x='Loan_Status', data=data, palette='Set2', ax=ax2)
# st.pyplot(fig2)

# # Income distribution
# st.markdown("#### Applicant Income Distribution")
# fig3, ax3 = plt.subplots()
# sns.histplot(data['ApplicantIncome'], kde=True, bins=30, ax=ax3)
# st.pyplot(fig3)

# # Feature importance (if model has attribute)
# if hasattr(model, 'feature_importances_'):
#     st.subheader("Feature Importance")
#     importance = model.feature_importances_
#     feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self Employed', 
#                      'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
#                      'Loan_Amount_Term', 'Credit_History', 'Property_Area']
#     imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
#     imp_df = imp_df.sort_values(by='Importance', ascending=False)
#     st.bar_chart(imp_df.set_index('Feature'))





# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the trained model
# with open('loan_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Load dataset for EDA
# data = pd.read_csv('LoanApprovalPrediction.csv')

# st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
# st.title("Loan Approval Prediction App")

# # Sidebar options
# option = st.sidebar.radio("Choose an Option", ['Predict Loan Status', 'Explore Data (EDA)'])

# # -----------------------------------------
# # 🚀 1. Prediction Section
# # -----------------------------------------
# if option == 'Predict Loan Status':
#     st.sidebar.header("Applicant Details")

#     gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
#     married = st.sidebar.selectbox("Married", ("Yes", "No"))
#     dependents = st.sidebar.selectbox("Number of Dependents", ("0", "1", "2", "3+"))
#     education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
#     self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
#     applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
#     coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
#     loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0)
#     loan_term = st.sidebar.selectbox("Loan Term (in months)", (360, 120, 240, 180, 60))
#     credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
#     property_area = st.sidebar.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

#     def preprocess():
#         gender_val = 1 if gender == "Male" else 0
#         married_val = 1 if married == "Yes" else 0
#         education_val = 1 if education == "Graduate" else 0
#         self_emp_val = 1 if self_employed == "Yes" else 0
#         property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
#         property_val = property_map[property_area]
#         dependents_val = 3 if dependents == "3+" else int(dependents)

#         features = np.array([[gender_val, married_val, dependents_val, education_val,
#                               self_emp_val, applicant_income, coapplicant_income,
#                               loan_amount, loan_term, credit_history, property_val]])
#         return features

#     if st.button("Predict"):
#         input_data = preprocess()
#         prediction = model.predict(input_data)

#         if prediction[0] == 1:
#             st.success("✅ Loan Approved!")
#         else:
#             st.error("❌ Loan Rejected.")

# # -----------------------------------------
# # 📊 2. EDA Section
# # -----------------------------------------
# elif option == 'Explore Data (EDA)':
#     st.header("📈 Data Exploration")

#     st.subheader("Preview of Dataset")
#     st.dataframe(data.head())

#     st.subheader("Loan Status Distribution")
#     fig1, ax1 = plt.subplots()
#     sns.countplot(data=data, x='Loan_Status', ax=ax1)
#     st.pyplot(fig1)

#     st.subheader("Gender Distribution")
#     fig2, ax2 = plt.subplots()
#     sns.countplot(data=data, x='Gender', ax=ax2)
#     st.pyplot(fig2)

#     st.subheader("Education vs Loan Status")
#     fig3, ax3 = plt.subplots()
#     sns.countplot(data=data, x='Education', hue='Loan_Status', ax=ax3)
#     st.pyplot(fig3)

#     st.subheader("Applicant Income Distribution")
#     fig4, ax4 = plt.subplots()
#     sns.histplot(data['ApplicantIncome'], kde=True, ax=ax4)
#     st.pyplot(fig4)





# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model
# with open('loan_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# st.title("Loan Approval Prediction App")

# st.sidebar.header("Applicant Details")

# # Input fields
# gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
# married = st.sidebar.selectbox("Married", ("Yes", "No"))
# dependents = st.sidebar.selectbox("Number of Dependents", ("0", "1", "2", "3+"))
# education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
# self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
# applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
# coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
# loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0)
# loan_term = st.sidebar.selectbox("Loan Term (in months)", (360, 120, 240, 180, 60))
# credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
# property_area = st.sidebar.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

# # Convert input to model format
# def preprocess():
#     gender_val = 1 if gender == "Male" else 0
#     married_val = 1 if married == "Yes" else 0
#     education_val = 1 if education == "Graduate" else 0
#     self_emp_val = 1 if self_employed == "Yes" else 0
#     property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
#     property_val = property_map[property_area]
#     if dependents == "3+":
#         dependents_val = 3
#     else:
#         dependents_val = int(dependents)

#     features = np.array([[gender_val, married_val, dependents_val, education_val,
#                           self_emp_val, applicant_income, coapplicant_income,
#                           loan_amount, loan_term, credit_history, property_val]])
#     return features

# # Predict
# if st.button("Predict"):
#     input_data = preprocess()
#     prediction = model.predict(input_data)

#     if prediction[0] == 1:
#         st.success("Loan Approved!")
#     else:
#         st.error("Loan Rejected.")