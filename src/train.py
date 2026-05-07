# training the model
import joblib
import os
from sklearn.cluster import KMeans
from preprocess import load_and_preprocess
from sklearn.metrics import silhouette_score

def train_and_save(data_path,model_save_path):
    print("Loading the data!!")
    X_scaled, scaler = load_and_preprocess(data_path)

    print("Training The Model")
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled) 
    labels = model.labels_

    # finding the score
    score = silhouette_score(X_scaled,labels)
    print(f"Silhouette Score : {round(score, 4)}")

    # creating a pickle file 
    os.makedirs(os.path.dirname(model_save_path),exist_ok=True)
    joblib.dump({'model':model,'scaler':scaler},model_save_path)
    print(f"The Model Saved to : {model_save_path}")

if __name__ == '__main__':
    train_and_save(
        data_path= r'C:\Users\HP\OneDrive\Desktop\country_development\data\Country-data.csv',
        model_save_path= r'C:\Users\HP\OneDrive\Desktop\country_development\model\country_model.pkl'
    )



