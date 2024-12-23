# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from typing import List, Dict
import logging
import warnings
from download_data import load_german_credit_data

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Create a file handler for logging to a file
file_handler = logging.FileHandler('credit_risk_scoring.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Set basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter out warnings to keep output clean
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_device():
    """
    Determine the appropriate device for training (CPU/GPU).
    Returns:
        torch.device: The device to be used for training
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class CreditDataset(Dataset):
    """
    Custom Dataset class for handling credit risk data.
    Converts numpy arrays to PyTorch tensors and provides indexing functionality.
    """
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class RiskScorer(nn.Module):
    """
    Neural Network architecture for credit risk scoring.
    Features:
    - Layer normalization for better training stability
    - Residual connections where possible
    - Dropout for regularization
    - GELU activation functions
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int = 1):
        super(RiskScorer, self).__init__()

        # Feature extraction layers with layer normalization
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.feature_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),  # Normalize layer outputs
                    nn.BatchNorm1d(dim),  # Batch normalization
                    nn.GELU(),  # GELU activation
                    nn.Dropout(0.4)  # Dropout for regularization
                )
            )
            prev_dim = dim

        # Risk scoring layers with additional normalization
        self.risk_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        # Process through feature layers with residual connections
        for layer in self.feature_layers:
            identity = x if x.size() == layer(x).size() else None
            x = layer(x)
            if identity is not None:
                x = x + identity  # Residual connection

        # Final risk scoring
        return self.risk_layers(x)

class DeepCreditRiskScorer:
    """
    Main class for deep learning-based credit risk scoring.
    Handles the entire pipeline from data preprocessing to model evaluation.
    """
    def __init__(self, random_state: int = 42):
        """Initialize the Deep Credit Risk Scorer."""
        self.random_state = random_state
        self.preprocessor = None
        self.risk_scorer = None
        self.feature_names_ = None
        self.label_encoders = {}
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

    def _create_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features with focus on risk assessment.

        Features created include:
        - Financial ratios and metrics
        - Risk scores for categorical variables
        - Age-based risk factors
        - Composite risk scores
        - Interaction features
        """
        X = X.copy()

        # Core financial metrics - Calculate key financial ratios
        X['credit_income_ratio'] = X['credit_amount'] / np.maximum(X['installment_rate'], 1)
        X['monthly_burden'] = X['credit_amount'] / np.maximum(X['duration'], 1)
        X['debt_service_ratio'] = X['monthly_burden'] / X['installment_rate']
        X['credit_per_month'] = X['credit_amount'] / X['duration']
        X['payment_ratio'] = X['installment_rate'] / X['duration']

        # Risk mapping for account status - Higher score means higher risk
        status_risk = {
            'A11': 0.9,  # < 0 DM (high risk)
            'A12': 0.6,  # 0-200 DM
            'A13': 0.3,  # >= 200 DM
            'A14': 1.0   # no checking account (highest risk)
        }
        X['checking_risk'] = X['status'].map(status_risk).fillna(0.7)

        # Risk mapping for savings - Higher score means higher risk
        savings_risk = {
            'A61': 0.8,  # < 100 DM
            'A62': 0.6,  # 100-500 DM
            'A63': 0.4,  # 500-1000 DM
            'A64': 0.2,  # >= 1000 DM
            'A65': 1.0   # unknown/no savings
        }
        X['savings_risk'] = X['savings'].map(savings_risk).fillna(0.9)

        # Risk mapping for employment duration - Higher score means higher risk
        employment_risk = {
            '0': 1.0,    # unemployed
            '1': 0.8,    # < 1 year
            '2': 0.5,    # 1-4 years
            '3': 0.3,    # 4-7 years
            '4': 0.1     # >= 7 years
        }
        X['employment_risk'] = X['employment_duration'].map(employment_risk).fillna(0.7)

        # Risk mapping for loan purpose - Risk varies by purpose
        purpose_risk = {
            'A40': 0.5,  # car (new)
            'A41': 0.6,  # car (used)
            'A42': 0.4,  # furniture
            'A43': 0.3,  # television
            'A44': 0.7,  # appliances
            'A45': 0.4,  # repairs
            'A46': 0.2,  # education
            'A47': 0.8,  # vacation
            'A48': 0.5,  # retraining
            'A49': 0.6,  # business
            'A410': 0.5  # others
        }
        X['purpose_risk'] = X['purpose'].map(purpose_risk).fillna(0.5)

        # Risk mapping for credit history - Based on past credit behavior
        history_risk = {
            '0': 1.0,    # critical
            '1': 0.8,    # delayed
            '2': 0.4,    # existing paid
            '3': 0.1,    # all paid
            '4': 0.6     # no credits
        }
        X['history_risk'] = X['credit_history'].map(history_risk).fillna(0.7)

        # Age-based risk assessment
        X['age_risk'] = np.where(X['age'] < 25, 0.8,
                        np.where(X['age'] < 35, 0.6,
                        np.where(X['age'] < 50, 0.4,
                        np.where(X['age'] < 65, 0.3, 0.5))))

        # Composite risk scores - Combining multiple risk factors
        X['financial_risk'] = (
            X['checking_risk'] * 0.3 +
            X['savings_risk'] * 0.2 +
            X['credit_income_ratio'] / X['credit_income_ratio'].max() * 0.3 +
            X['debt_service_ratio'] / X['debt_service_ratio'].max() * 0.2
        )

        X['stability_risk'] = (
            X['employment_risk'] * 0.4 +
            X['age_risk'] * 0.3 +
            X['history_risk'] * 0.3
        )

        # Purpose impact weighted by credit amount
        X['purpose_impact'] = X['purpose_risk'] * X['credit_amount'] / X['credit_amount'].max()

        # Interaction features - Combining different risk aspects
        X['risk_exposure'] = X['financial_risk'] * X['credit_amount'] / X['duration']
        X['stability_factor'] = (1 - X['stability_risk']) * X['installment_rate']
        X['credit_confidence'] = (1 - X['history_risk']) * (1 - X['financial_risk'])

        # Binary indicators for key factors
        X['has_checking'] = (X['status'] != 'A14').astype(int)
        X['has_savings'] = (X['savings'] != 'A65').astype(int)
        X['is_employed'] = (X['employment_duration'] != '0').astype(int)
        X['good_history'] = (X['credit_history'].isin(['2', '3'])).astype(int)

        # Remove original categorical columns after feature engineering
        cols_to_drop = ['status', 'savings', 'employment_duration', 'purpose', 'credit_history']
        X = X.drop(cols_to_drop, axis=1)

        # Handle any remaining missing values in numeric columns
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        return X

    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline for both numeric and categorical features.
        Includes imputation and scaling/encoding steps.
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        return preprocessor

    def _train_risk_scorer(self, X_train: np.ndarray, y_train: np.ndarray,
                          batch_size: int = 32, epochs: int = 200,
                          hidden_dims: List[int] = [256, 128, 64]):
        """Train risk scorer with focused training strategy."""
        input_dim = X_train.shape[1]
        self.risk_scorer = RiskScorer(input_dim, hidden_dims).to(self.device)

        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train.astype(int))
        class_weights = torch.FloatTensor(len(y_train) / (2 * class_counts)).to(self.device)

        # Create balanced sampler
        class_count = np.bincount(y_train)
        weight = 1. / class_count
        sample_weights = torch.FloatTensor([weight[t] for t in y_train])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # Create data loader with balanced sampling
        train_dataset = CreditDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.risk_scorer.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Training loop
        logger.info("Training risk scorer...")
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.risk_scorer.train()
            total_loss = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                outputs = self.risk_scorer(batch_features)

                # Calculate weighted loss
                loss = criterion(outputs, batch_labels.view(-1, 1))
                loss = loss * class_weights[batch_labels.long()]
                loss = loss.mean()

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.risk_scorer.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step()

            # Early stopping with model selection
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = {
                    'state_dict': self.risk_scorer.state_dict(),
                    'loss': best_loss
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

                # Validation metrics
                self.risk_scorer.eval()
                with torch.no_grad():
                    val_outputs = self.risk_scorer(torch.FloatTensor(X_train).to(self.device))
                    val_preds = (val_outputs.cpu().numpy() >= 0.5).astype(int)
                    train_accuracy = accuracy_score(y_train, val_preds)
                    logger.info(f"Training Accuracy: {train_accuracy:.3f}")
                self.risk_scorer.train()

        # Load best model
        if best_model_state is not None:
            self.risk_scorer.load_state_dict(best_model_state['state_dict'])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit the model using deep learning approach with cross-validation."""
        logger.info("Starting model training...")

        # Create engineered features
        X = self._create_engineered_features(X)
        self.feature_names_ = list(X.columns)

        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor(X)
        X_transformed = self.preprocessor.fit_transform(X)

        # Implement k-fold cross-validation
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        val_scores = []
        best_val_score = -float('inf')
        best_model_state = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_transformed, y)):
            logger.info(f"\nTraining fold {fold+1}/{n_splits}")

            X_train, X_val = X_transformed[train_idx], X_transformed[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train risk scorer
            self._train_risk_scorer(X_train, y_train)

            # Evaluate and save best model
            self.risk_scorer.eval()
            with torch.no_grad():
                val_features = torch.FloatTensor(X_val).to(self.device)
                val_preds = self.risk_scorer(val_features)
                val_preds = (val_preds.cpu().numpy() >= 0.5).astype(int)
                val_accuracy = accuracy_score(y_val, val_preds)
                val_scores.append(val_accuracy)
                logger.info(f"Fold {fold+1} Validation Accuracy: {val_accuracy:.3f}")

                if val_accuracy > best_val_score:
                    best_val_score = val_accuracy
                    best_model_state = self.risk_scorer.state_dict()

        mean_val_score = np.mean(val_scores)
        std_val_score = np.std(val_scores)
        logger.info(f"\nCross-validation results:")
        logger.info(f"Mean Validation Accuracy: {mean_val_score:.3f} (+/- {std_val_score:.3f})")

        # Load best model instead of retraining
        if best_model_state is not None:
            self.risk_scorer.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {best_val_score:.3f}")

        logger.info("Model training completed successfully")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of bad credit."""
        # Preprocess data
        X = self._create_engineered_features(X)
        X_transformed = self.preprocessor.transform(X)

        # Get predictions
        with torch.no_grad():
            self.risk_scorer.eval()
            features = torch.FloatTensor(X_transformed).to(self.device)
            risk_scores = self.risk_scorer(features)
            return risk_scores.cpu().numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict credit risk (1 for bad credit, 0 for good credit)."""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)

    def evaluate_model(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict:
        """Evaluate model performance."""
        # Get predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Log results
        logger.info(f"Model Accuracy: {accuracy:.3f}")
        logger.info(f"Specificity: {specificity:.3f}")
        logger.info("\nClassification Report:")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1-score: {f1:.3f}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        }

if __name__ == "__main__":
    # Load German Credit Data
    df = load_german_credit_data()
    print("Dataset shape:", df.shape)

    # Split features and target
    y = df['credit_risk'].values
    X = df.drop('credit_risk', axis=1)

    # Split data into train and test sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=42,
        stratify=y
    )

    # Initialize and train model
    scorer = DeepCreditRiskScorer()
    scorer.fit(X_train, y_train)

    # Evaluate model on test set
    print("\nModel Evaluation on Test Set:")
    evaluation_metrics = scorer.evaluate_model(X_test, y_test)

    # Save model state and preprocessor
    torch.save({
        'model_state': scorer.risk_scorer.state_dict(),
        'preprocessor': scorer.preprocessor,
        'feature_names': scorer.feature_names_,
        'device': str(scorer.device)
    }, 'german_credit_risk_model.pth')
    print("\nModel saved to 'german_credit_risk_model.pth'")