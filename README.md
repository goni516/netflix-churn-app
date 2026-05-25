# Netflix Customer Churn Prediction System

This Streamlit app predicts churn risk for OTT subscribers using a trained Random Forest model.

## Live App

Deploy this repository on Streamlit Community Cloud and set the main file path to:

```text
app.py
```

## Model

The model uses eight selected features:

- `avg_watch_time_per_day`
- `watch_hours`
- `last_login_days`
- `number_of_profiles`
- `payment_method`
- `subscription_type`
- `age`
- `monthly_fee`

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Files

- `app.py`: Streamlit frontend and prediction logic
- `netflix_churn_model.joblib`: trained Random Forest model
- `Netflix_Churn_Model.ipynb`: modeling notebook
- `Netflix_Churn_Project_Report.docx`: technical report
