# Medical Cost Prediction Project

This project analyzes and predicts medical costs using machine learning techniques.

## Live Demo

🔗 **[Try the Live Application](https://insurance-cost-prediction.streamlit.app/)**

The application is deployed on Streamlit Cloud and provides instant medical insurance cost predictions in both USD and INR.

## Setup Instructions

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `notebooks/Medical_Cost_Prediction.ipynb` in your browser and run the cells sequentially.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks with analysis
  - `Medical_Cost_Prediction.ipynb`: Main analysis notebook
- `app.py`: Streamlit web application for cost prediction
- `models/`: Contains trained machine learning models

## Data Description

The project analyzes medical cost data with features including:

- Age
- BMI (Body Mass Index)
- Number of children
- Medical charges

## Features

- Predicts medical insurance costs in USD and INR
- Takes into account:
  - Age
  - BMI
  - Smoking status
  - Number of children
  - Region
  - Gender

## Model

Currently using Linear Regression with features:

- Age
- BMI
- Smoking status

## Usage

You can either:

1. Use the [live application](https://insurance-cost-prediction.streamlit.app/)
2. Run locally using `streamlit run app.py`
