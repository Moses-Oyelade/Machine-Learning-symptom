import pandas as pd
import zipfile

# Open ZIP file
with zipfile.ZipFile("./disease-symptom-description-dataset.zip", "r") as z:
    # List files inside ZIP
    print(z.namelist())  # Check available files
    # Read a specific CSV inside the ZIP
    with z.open("dataset.csv") as f:
        df = pd.read_csv(f)
    with z.open("symptom_Description.csv") as f:
        df1 = pd.read_csv(f)
    with z.open("symptom_precaution.csv") as f:
        df2 = pd.read_csv(f)

# Convert disease names to lowercase for uniformity
df1["Disease"] = df1["Disease"].str.lower()
df2["Disease"] = df2["Disease"].str.lower()

# Show first few rows
print(df.head())

# Preprocess the Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fill missing values
df.fillna("", inplace=True)

# print(df.describe())
# print(type(df))

# Merge all symptoms into a single string
df["all_symptoms"] = df.iloc[:, 1:].apply(lambda x: " ".join(x.dropna()), axis=1)



# Encode diseases into numbers
label_encoder = LabelEncoder()
df["encoded_disease"] = label_encoder.fit_transform(df["Disease"])


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["all_symptoms"], df["encoded_disease"], test_size=0.2, random_state=42)

# Train a Simple ML Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Convert text to numeric vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Dataframe for disease description
def get_disease_description(disease_name):
    disease_name = disease_name.lower()
    result = df1[df1["Disease"] == disease_name]["Description"]
      
    if not result.empty:
        return result.values[0]
    else:
        return "Disease not found in database."

# Dataframe for Precaution to Disease
def get_precautions(disease):
    disease = disease.lower().strip()
    result = df2[df2["Disease"] == disease].iloc[:, 1:].apply(lambda x: ", ".join(x.dropna()), axis=1)  # Select only precaution columns
    
    if not result.empty:
        return result.values.tolist()[0]  # Return list of precautions
    else:
        return "No matching disease found in database."

 
# Save the Model for Deployment
import joblib

joblib.dump(model, "symptom_disease_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Save the dataframe as a lookup model
joblib.dump(df1, "disease_description_model.pkl")

# Save the dataframe as a lookup model
joblib.dump(df2, "disease_precaution_model.pkl")


from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware # Import the CORS module
from pydantic import BaseModel
import joblib

# Load the trained model and vectorizer
model = joblib.load('symptom_disease_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load("label_encoder.pkl")

df1 = joblib.load("disease_description_model.pkl")
df2 = joblib.load("disease_precaution_model.pkl")

app = FastAPI()

class SymptomRequest(BaseModel):
    symptoms: str

@app.get("/")
async def home():
    return {"message": "Disease Symptom Prediction API"}

@app.post('/predict')
async def predict_disease(request: SymptomRequest):
    # Vectorize the input symptoms
    input_vector = vectorizer.transform([request.symptoms])
    # Predict the disease
    prediction = model.predict(input_vector)

    # return {'predicted_disease': prediction[0]}

    # Decode the disease label
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    

    description = get_disease_description(predicted_disease)
    print(f"Disease: {predicted_disease}\nDescription: {description}")
    
    precautions = get_precautions(predicted_disease)
    print(f"Precautions for {predicted_disease}: {precautions}")
    
    return {'predicted_disease': predicted_disease, 'Description': description, 'Precaution': precautions}
    # return {'predicted_disease': predicted_disease}