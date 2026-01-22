# Breast Cancer Prediction System

## Project Overview
This project is a Breast Cancer Prediction System using Logistic Regression and Streamlit.

## Setup Instructions

1.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model:**
    - Open `model/model_building.ipynb` in Jupyter Notebook or VS Code.
    - Run all cells to train the model and save `breast_cancer_model.pkl` and `scaler.pkl` in the `model/` directory.

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Files
- `app.py`: Main Streamlit application.
- `model/model_building.ipynb`: Jupyter notebook for model training.
- `model/`: Directory to store the trained model and scaler.
- `BreastCancer_hosted_webGUI_link.txt`: Submission details file.
