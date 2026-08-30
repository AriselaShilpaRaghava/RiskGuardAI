\# 🛡️ RiskGuard AI



\## AI-Powered Transaction Risk Assessment and Explanation System



RiskGuard AI is a machine-learning-based prototype designed to assess transaction fraud risk and provide an explanation for the prediction.



The system goes beyond simple fraud prediction by providing:

\- Fraud probability

\- Risk score

\- Risk level

\- SHAP-based explanation

\- Explanation stability analysis



\---



\## 🎯 Problem Statement



Traditional fraud detection systems may provide a fraud prediction without clearly explaining why a transaction was considered risky.



RiskGuard AI aims to provide a more interpretable transaction risk assessment by combining machine-learning prediction with explainable AI and explanation reliability testing.



\---



\## 💡 Proposed Solution



RiskGuard AI follows this pipeline:



Transaction Data

&#x20;       ↓

Machine Learning Model

&#x20;       ↓

Fraud Probability

&#x20;       ↓

Risk Score

&#x20;       ↓

Risk Level

&#x20;       ↓

SHAP Explanation

&#x20;       ↓

Explanation Stability Test



\---



\## 🚀 Features



\### 1. Fraud Risk Prediction

The trained ML model estimates the probability that a transaction is fraudulent.



\### 2. Risk Score

The probability is converted into a 0–100 risk score.



\### 3. Risk Classification



\- 0–29 → LOW

\- 30–69 → MEDIUM

\- 70–100 → HIGH



\### 4. Explainable AI



SHAP is used to identify which transaction features contribute toward higher or lower model risk.



\### 5. Explanation Reliability



A feature-change experiment is used to evaluate whether model behavior is directionally consistent.



\---



\## 🧠 Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- SHAP

\- Streamlit

\- Joblib



\---



\## 📊 Model Results



Current experimental results:



| Metric | Result |

|---|---:|

| Accuracy | 77.25% |

| Precision | 31.63% |

| Recall | 29.12% |

| F1 Score | 30.32% |

| ROC-AUC | 72.14% |



\### Explanation Stability



\- Total feature-change tests: 500

\- Consistent tests: 374

\- Inconsistent tests: 126

\- Preliminary stability score: 74.80%



> Note: These results are from a synthetic dataset and represent a prototype experiment, not production fraud-detection performance.



\---



\## 🔍 Example



Example transaction:



\- Transaction amount: ₹25,000

\- Transaction hour: 2

\- New beneficiary: Yes

\- New device: Yes

\- Location changed: Yes

\- Recent transaction velocity: 6

\- Previous risk: 0.75

\- Account age: 100 days



Example output:



\- Risk Score: 70/100

\- Fraud Probability: 70%

\- Risk Level: HIGH



Important SHAP contributors included:



\- Location change

\- Transaction velocity

\- New beneficiary

\- Transaction hour

\- Previous risk



\---



\## 📁 Project Structure



```text

RiskGuardAI/

│

├── app.py

├── data\_generator.py

├── train\_model.py

├── risk\_score.py

├── explain\_risk.py

├── stability\_test.py

├── synthetic\_transactions.csv

├── risk\_model.pkl

└── README.md



**Installation**



Install the required Python packages:

pip install pandas numpy scikit-learn joblib shap streamlit



**Run the Project**

Generate dataset:

python data\_generator.py



**Train model**

python train\_model.py



**Test risk score**

python risk\_score.py



**Test SHAP explanation**

python explain\_risk.py



**Test explanation stability**

python stability\_test.py



**Launch dashboard**

python -m streamlit run app.py

The dashboard will open in the browser.



**Limitations**

The current dataset is synthetic.

The model has not been validated on real banking transaction data.

The current stability analysis is a preliminary directional consistency test.

The system should not be treated as a production fraud-detection system.



🔮 **Future Improvements**

Evaluate on real-world or benchmark fraud datasets.

Compare multiple ML models.

Improve fraud-class recall and precision.

Add real-time transaction monitoring.

Add stronger explanation validation methods.

Add historical user behavior and transaction context.

Deploy the system as a secure web service.







**Project Status**



**Prototype completed**



Current pipeline:



Prediction → Risk Score → Explainability → Explanation Reliability



