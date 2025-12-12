# main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.title("HappyLens: The Happiness Predictor – User Survey")
st.write("The happiness project aims to help you predict what to prioritize in your future life! Fill out the survey as your future self, using what you imagine your life would be like at the age you would like to predict your happiness for.")
st.write("For example, if I wanted to predict how happy I will be at 50 years old, I would put my age in as 50 and answer the questions with how I think my life will look then")
st.write("Please answer the questions below:")

# ---------------------------
# USER INPUTS
# ---------------------------
age = st.number_input("What is your age?", min_value=18, max_value=100, value=30)
childs = st.number_input("How many children do you have?", min_value=0, max_value=20, value=0)
is_religious = st.selectbox("Do you consider yourself religious?", ["Yes", "No"])
is_religious = 1 if is_religious == "Yes" else 0

wrkstat = st.selectbox("Work status", [
    "Working full time", "Working part time", "With a job, but not at work",
    "In school", "Retired", "Keeping house", "Unemployed, laid off, looking for work",
    "Other", "NA"
])
marital = st.selectbox("Marital status", ["Married", "Never married", "Separated", "Widowed", "Divorced"])
race = st.selectbox("Race", ["White", "Black", "Other"])
reg16 = st.selectbox("Region", ["Midwest", "Northeast", "South", "West", "Foreign", "NA"])
family16 = st.selectbox("Family situation at 16 years old", [
    "Father only", "Mother only", "Father and stepparent", "Mother and stepparent",
    "Some other male relative", "Some other female relative", "Other arrangement with relatives",
    "Both own parents", "Other"
])
relig = st.selectbox("Religion", [
    "Catholic", "Christian", "Hinduism", "Inter-Nondenominational", "Jewish", "Muslim/Islam",
    "Native American", "Orthodox-Christian", "Other Eastern religions", "Protestant", "None", "Other"
])
reliten = st.selectbox("Religious intensity", ["Strong", "Somewhat strong", "Not very strong", "No religion"])
hapmar = st.selectbox("Marital happiness (if married)", ["VERY HAPPY", "PRETTY HAPPY", "NOT TOO HAPPY", "NA"])
health = st.selectbox("Health status", ["Excellent", "Good", "Fair", "Poor"])
satjob = st.selectbox("Job satisfaction", ["Very satisfied", "Moderately satisfied", "A little dissatisfied", "Very dissatisfied", "NA"])
satfin = st.selectbox("Financial satisfaction", ["Very satisfied", "Pretty well satisfied", "More or less satisfied", "Not satisfied at all", "NA"])
finalter = st.selectbox("Financial situation compared to last year", ["Better", "Stayed same", "Worse", "NA"])
gender1 = st.selectbox("Gender", ["Male", "Female", "Other"])
educ_cat = st.selectbox("Education level", ["elementary", "middle", "some_high_school", "high_school", "some_college", "college", "6+_college", "NA"])
life = st.selectbox("Life outlook", ["Exciting", "Routine", "Dull", "Other"])
income_mid = st.number_input("Total family income per month (income of all your future household members)", min_value=0, max_value=300000, value=50000)
rincome_mid = st.number_input("Your income per month(mid estimate of what you will make in your future job)", min_value=0, max_value=300000, value=50000)

# ---------------------------
# LOAD PRE-TRAINED MODEL
# ---------------------------
g_clf = joblib.load("g_clf.pkl")
training_columns = joblib.load("columns.pkl")  # list of columns the model expects

# ---------------------------
# BUTTON TO GET PREDICTION
# ---------------------------
if st.button("Get My Happiness Prediction"):

    # Default numeric values
    user_data = {
        'childs': childs,
        'age': age,
        'is_religious': is_religious,
        'income_mid': income_mid,
        'rincome_mid': rincome_mid,
        'wtssps': 1.0  # default weight if not asked from user
    }

    # Categorical answers from the survey
    cats = {
        'wrkstat': wrkstat,
        'marital': marital,
        'race': race,
        'reg16': reg16,
        'family16': family16,
        'relig': relig,
        'reliten': reliten,
        'hapmar': hapmar,
        'health': health,
        'satjob': satjob,
        'satfin': satfin,
        'finalter': finalter,
        'gender1': gender1,
        'educ_cat': educ_cat,
        'life': life
    }

    # Map survey 'NA' selections to '_nan' columns
    na_mapping = {
        "NA": True # example if needed, otherwise handled below
    }

    # One-hot encode exactly like training
    for prefix, answer in cats.items():
        mapped = False
        for col in training_columns:
            if col.startswith(prefix + "_"):
                # Handle NA selections
                if answer in na_mapping and "_nan" in col:
                    user_data[col] = 1
                    mapped = True
                elif col.split(prefix + "_")[1] == answer:
                    user_data[col] = 1
                    mapped = True
                else:
                    user_data[col] = 0
        # If nothing matched (rare), leave 0
        if not mapped:
            for col in training_columns:
                if col.startswith(prefix + "_") and col.endswith("_nan"):
                    user_data[col] = 1

    # Fill any remaining columns with 0
    for col in training_columns:
        if col not in user_data:
            user_data[col] = 0

    # Build user DataFrame in correct column order
    df_user = pd.DataFrame([user_data])[training_columns]

    st.write("### Your Input Data")
    st.dataframe(df_user)

    # Predict
    probs_user = g_clf.predict_proba(df_user)
    expected_happiness_user = np.dot(probs_user, [1, 5, 10])

    st.write("### Predicted Happiness Score (out of 10)")
    st.write(round(expected_happiness_user[0], 2))
