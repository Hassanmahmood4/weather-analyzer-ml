🌦️ Weather Analyzer ML

An end-to-end Machine Learning project that combines historical weather data (Kaggle) with live weather from OpenWeather API to deliver an interactive Streamlit app.

🔁 Two Modes
	•	City → Conditions: Enter a city and get:
	•	🤖 ML-predicted temperature
	•	☁️ Current weather summary
	•	🌧️ Precipitation type
	•	💨 Wind category
	•	Conditions → Closest City: Enter weather conditions and find the closest matching city based on live weather similarity.
<img width="874" height="686" alt="image" src="https://github.com/user-attachments/assets/cec4dd96-1213-4176-b882-e3c11384050e" />
<img width="786" height="613" alt="image" src="https://github.com/user-attachments/assets/b6a1448f-391c-431b-bb7f-511c4421ad0e" />
<img width="782" height="510" alt="image" src="https://github.com/user-attachments/assets/bc584ae2-afa1-4bac-a7f6-4c55c0db72e6" />



✨ Features
	•	📊 EDA on Kaggle weather dataset
	•	🧼 Data preprocessing & feature engineering
	•	🤖 RandomForest regression model for temperature prediction
	•	🛡️ Feature leakage prevention (excluded “Apparent Temperature”)
	•	🌐 Live weather integration (OpenWeather API)
	•	🖥️ Streamlit UI with two interactive modes
	•	🔍 Similarity-based city recommendation


🗂️ Project Structure

weather-analyzer-ml/
├── app.py                  # Streamlit app
├── src/
│   ├── eda.py              # EDA
│   ├── preprocess.py      # Data cleaning & features
│   ├── train_model.py     # Model training
│   ├── predict.py         # Inference script
│   ├── weather_api.py     # OpenWeather API helper
│   └── city_recommender.py# City similarity logic
├── requirements.txt
├── .gitignore
└── README.md

Note: data/, models/, .env, and .venv are intentionally ignored from Git for security and size reasons.


📦 Dataset
	•	Kaggle – Weather History Dataset
Hourly historical weather observations (temperature, humidity, wind, pressure, visibility, etc.)

Place the dataset at:

data/raw/weatherHistory.csv



🔑 API Setup (OpenWeather)
	1.	Create an API key: https://openweathermap.org/api
	2.	Create a .env file in project root:

OPENWEATHER_API_KEY=YOUR_API_KEY



🛠️ Installation

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt



▶️ Run the Pipeline

1) Preprocess data

python src/preprocess.py

2) Train model

python src/train_model.py

3) Test prediction

python src/predict.py

4) Launch the app

streamlit run app.py

Open: http://localhost:8501


📈 Model Performance

After removing feature leakage:
	•	MAE: ~1.43 °C
	•	RMSE: ~1.95 °C

These are realistic metrics for temperature prediction.


🧩 Tech Stack
	•	Python
	•	Pandas, NumPy
	•	scikit-learn
	•	Streamlit
	•	OpenWeather API


🚀 Future Improvements
	•	🗺️ Add map visualization for closest city
	•	☁️ Deploy on Streamlit Cloud
	•	📈 Add model explainability (SHAP)
	•	⚡ Try XGBoost / LightGBM for improved accuracy


👤 Author

Hassan Mahmood
GitHub: https://github.com/Hassanmahmood4

