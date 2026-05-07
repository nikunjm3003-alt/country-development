import joblib
from preprocess import transform_input
import pandas as pd

def load_model(model_path):
    artifacts = joblib.load(model_path)
    return artifacts['model'],artifacts['scaler']

def predict(input_dict,
            model_path = r'C:\Users\HP\OneDrive\Desktop\country_development\model\country_model.pkl'):
    
    model , scaler = load_model(model_path)
    df = pd.DataFrame([input_dict])
    X_scaled = transform_input(df,scaler)

    cluster = model.predict(X_scaled)[0]

    cluster_labels = {
        0: 'Developed',
        1: 'Developing',
        2: 'Underdeveloped',
        3: 'Critical'
    }

    return {
        'cluster' : int(cluster),
        'label' : cluster_labels.get(cluster,f'Cluster {cluster}')
    }

if __name__ == '__main__':
    sample_input = {
        'child_mort': 90,
        'exports': 10,
        'health': 7.5,
        'imports': 20,
        'income': 1500,
        'inflation': 5.0,
        'life_expec': 60,
        'total_fer': 3.5,
        'gdpp': 700
    }

result = predict(sample_input)
print(f"Cluster : {result['cluster']}")
print(f"Label   : {result['label']}")
