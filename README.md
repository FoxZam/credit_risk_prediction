# Credit Risk Prediction

This project implements a credit risk prediction model using deep learning techniques. The model is designed to predict the likelihood of a customer defaulting on a loan based on various financial and personal attributes.

## Project Structure

- `credit_risk_scoring_dl.py`: Main script for training and evaluating the deep learning model.
- `download_data.py`: Script to download and preprocess the German Credit Data.
- `pyproject.toml`: Project dependencies and configuration.
- `README.md`: Project documentation.

## Setup

### Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)
- uv package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FoxZam/credit_risk_prediction.git
   cd credit-risk-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the project dependencies:
   ```bash
   uv sync
   ```

## Usage

### Data Preparation

1. Run the `download_data.py` script to download and preprocess the German Credit Data:
   ```bash
   uv run download_data.py
   ```

### Model Training and Evaluation

1. Run the `credit_risk_scoring_dl.py` script to train and evaluate the model:
   ```bash
   uv run credit_risk_scoring_dl.py
   ```

2. The script will output the model's performance metrics, including accuracy, precision, recall, and F1-score.

3. The trained model and preprocessor will be saved to `german_credit_risk_model.pth`.

## Results

The model's performance is evaluated using accuracy, precision, recall, F1-score, and specificity. These metrics provide insights into the model's ability to correctly classify credit risk.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- The German Credit Data is sourced from the UCI Machine Learning Repository.
- This project utilizes PyTorch for deep learning model implementation.