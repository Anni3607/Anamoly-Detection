CAUSAL-XAD: Causal, Uncertainty-Aware, Explainable Anomaly Detection

This project implements an end-to-end anomaly detection system that goes beyond simple detection and focuses on understanding why anomalies occur and how to fix them.

Instead of just labeling data points as anomalous, the system provides causal-style explanations, confidence estimates, and actionable recommendations. It is designed to be practical, interpretable, and suitable for real-world usage.

What this project does

The system takes time series data as input and performs the following:

Detects anomalies using a machine learning model
Classifies anomalies into types such as spike, drift, or seasonal
Generates counterfactual explanations to identify root causes
Suggests how to modify feature values to reduce anomaly risk
Estimates confidence using model disagreement
Allows user feedback to adjust detection behavior
Provides a visual dashboard through a Streamlit app
Key idea

Most anomaly detection systems answer this question:

“What is important?”

This system answers a more useful question:

“What change would make this normal again?”

This is done using counterfactual reasoning. Instead of assigning importance scores, the system modifies input features and observes how the anomaly score changes.

Features used

The model expects the following features:

value
lag_1
lag_2
rolling_mean
rolling_std

If these are not present, they can be generated from the value column.

How to run the project
Clone the repository
git clone https://github.com/your-username/causal-xad.git
cd causal-xad
Install dependencies
pip install -r requirements.txt
Run the Streamlit app
streamlit run app.py
Using the app
Upload a CSV file or use the provided sample data
Navigate through tabs for overview, trends, detection, explanation, feature impact, and uncertainty
Select an anomaly to view its explanation
Observe how changing feature values affects anomaly score
Example explanation

The system produces outputs like:

Anomaly Type: DRIFT
Root Cause: value is driving the anomaly
Fix Recommendation: change value from 39.8 to 63.2
Confidence: 99 percent

Evaluation of explanations

The system evaluates explanation quality using:

Faithfulness: whether applying the suggested fix reduces anomaly score
Stability: whether explanations remain consistent under small noise
Actionability: whether the anomaly can actually be corrected
Limitations
Assumes features are independent during counterfactual changes
Uses Isolation Forest, which is not inherently causal
Confidence is based on ensemble disagreement, not calibrated probability
Possible improvements
Use true causal models instead of feature perturbation
Add multivariate and real-world datasets
Improve uncertainty estimation with proper probabilistic methods
Deploy on cloud for public access
Why this project stands out

Most anomaly detection projects stop at detection or basic feature importance.

This system integrates detection, temporal reasoning, causal-style explanation, uncertainty estimation, and user interaction into one pipeline.

Author

Anirudha Pujari
Third Year CSBS
Interested in Machine Learning and Data Science
