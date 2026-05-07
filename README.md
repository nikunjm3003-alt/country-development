\# 🌍 Country Development Predictor 
**APP**
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://country-development-97mdmzappppexrvefryzbnuz.streamlit.app/)

A machine learning web application that classifies countries into development 

categories based on socio-economic and health indicators using unsupervised learning.



\## 📌 Objective

To categorize countries using factors like child mortality, GDP per capita, 

life expectancy, income, and trade balance to determine their overall 

development status.



\## 🗂️ Project Structure

country\_development/

│

├── data/

│   └── Country-data.csv        # raw dataset

│

├── model/

│   └── country\_model.pkl       # trained model + scaler

│

├── notebook/

│   └── country\_development.ipynb  # EDA and experimentation

│

├── src/

│   ├── preprocess.py           # data preprocessing and feature engineering

│   ├── train.py                # model training script

│   └── predict.py              # prediction logic

│

├── app.py                      # streamlit web app

├── README.md

└── requirements.txt



\## ⚙️ Tech Stack

\- \*\*Python\*\*

\- \*\*Scikit-learn\*\* — Agglomerative Clustering, StandardScaler

\- \*\*Streamlit\*\* — web app UI

\- \*\*Pandas \& NumPy\*\* — data manipulation

\- \*\*Joblib\*\* — model serialization



\## 🧠 ML Approach

\- \*\*Algorithm\*\* : Agglomerative Clustering (Hierarchical)

\- \*\*Metric\*\* : Silhouette Score

\- \*\*Feature Engineering\*\* :

&#x20; - `trade\_balance` = exports - imports

&#x20; - `income\_to\_gdp` = income / gdpp

&#x20; - `real\_income` = income / (1 + inflation/100)



\## 📊 Dataset

\- \*\*Source\*\* : Country-data.csv

\- \*\*Rows\*\* : 167 countries

\- \*\*Features\*\* : child mortality, exports, health spending, imports, 

&#x20; income, inflation, life expectancy, fertility rate, GDP per capita



\## 🚀 How to Run



\*\*1. Clone the repository\*\*

```bash

git clone https://github.com/yourusername/country-development.git

cd country-development

```



\*\*2. Create and activate virtual environment\*\*

```bash

python -m venv venv

venv\\Scripts\\Activate        # Windows PowerShell

```



\*\*3. Install dependencies\*\*

```bash

pip install -r requirements.txt

```



\*\*4. Train the model\*\*

```bash

python src/train.py

```



\*\*5. Run the app\*\*

```bash

streamlit run app.py

```



\## 📈 Development Categories

| Cluster | Status | Description |

|---|---|---|

| 0 | 🟢 Developed | High income, low child mortality, high life expectancy |

| 1 | 🟡 Developing | Moderate income and health indicators |

| 2 | 🟠 Underdeveloped | Low income, high child mortality |

| 3 | 🔴 Critical | Very low income, needs urgent aid |

