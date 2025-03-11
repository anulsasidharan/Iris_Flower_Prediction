import streamlit as st
import time
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# MongoDB connectivity
uri = "mongodb+srv://anu:tiger@cluster0.57jxgvp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new MongoDB client and connect to the Database server
client = MongoClient(uri, server_api = ServerApi('1'))
db = client['Iris']
collection = db['Iris_Flower_data']

# Step1: Module for Loading the required model
def load_model(model_name):
    """
    Loads the saved model and its scaler from a pickle file.

        Parametrs:
        ----------
            model_name:str
                Name of the model contains the model and its scaler

        Returns:
        --------
            model:The selected model
                The loaded model

            scaler:sklearn.preprocessing
                The loaded Scaler
    """
    with open(model_name, 'rb') as file:
        model, scaler = joblib.load(file)
    return model, scaler

# Step2 : Custome Module to proces the input data
def preprocessing_input_data(data, scaler):
    """
    Preprocess the input data and prepare it for the model to ingest for processing

    The input data is converted into  DataFrame,
    Then the data is transformed using scaler to standardize the features. 

    Parameters:
    -----------
        data: dict
            A dictionary containing the input data.
        scaler: object
            scaler object from sklearn.

    Returns:
    --------
        df_transformed: array
            The transformed data ready for prediction.
    """
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

# Step3: Module for predection of Diabetes progression
def predict_data(data, model_name):
    """
    Predict the output based on the input data given. 

    The input data is prossed through "preprocessing_input_data".
    and pass to "prediction" module to predict the output through loaded model. 

    Parameters:
    -----------
        data: dict
            input is a dictionary containing the flower details.

        model_name: str
            The name of the model containing the saved model and scaler.

    Returns:
    --------
        perdiction: array
            It contains the predected output.

    """
    model,scaler = load_model(model_name)
    processed_data = preprocessing_input_data(data, scaler)
    prediction = model.predict(processed_data)
    return prediction


#---------------------------------------------------------------------#
# Define the main function for the streamlit application
def main():
    st.set_page_config(layout="wide")
    st.title("Iris Flower Prediction Application :blossom:")


    # Sidebar layout for entering the data
    st.sidebar.header(":gear: **Model Selection**")
    model_choice = st.sidebar.radio("**Chose The Model**", [
        "SVM Binary", 
        "SVM Multiclass", 
        "Logistic Binary", 
        "Logistic Multiclass OVR", 
        "Logistic Multiclass Multinomial"
    ]
    )
    st.sidebar.markdown("---") # Divider Line


    # Get the flower data input
    st.sidebar.write("**The Iris Flower Data**")
    st.sidebar.write(":pencil: Enter the details of the flower to predict the species.")
    st.sidebar.markdown("---") # Divider Line

    sepal_length = st.sidebar.slider("Enter the sepal length: :straight_ruler: :blossom: ", 4.0,8.0,5.0)
    sepal_width = st.sidebar.slider("Enter the sepal width: :triangular_ruler: :blossom: ", 1.5,5.5,2.5)
    petal_length = st.sidebar.slider("Enter the petal length: :straight_ruler: :blossom: ", 1.0,7.0,2.0)
    petal_width = st.sidebar.slider("Enter the petal width: :triangular_ruler: :blossom: ", 0.1,2.5,0.5)

    # Data Mapping
    flower_data = {
        "sepal length (cm)":sepal_length,
        "sepal width (cm)":sepal_width,
        "petal length (cm)":petal_length,
        "petal width (cm)":petal_width,
    }

    # Submit button to predict the species
    if st.sidebar.button("Predict"): 
        with st.spinner(":stopwatch: Processing the flower details... Please wait!.."):
            time.sleep(2)

            # Map the model selection
            model_mapping = {
                "SVM Binary": "svm_binary.pkl",
                "SVM Multiclass": "svm_multi.pkl",
                "Logistic Binary": "logistic_binary.pkl",
                "Logistic Multiclass OVR": "logistic_ovr.pkl",
                "Logistic Multiclass Multinomial": "logistic_multinomial.pkl",
            }

            model_name =  model_mapping.get(model_choice, None)

            # Map the model ID for storing it in MongoDB
            model_ID_mapping = {
                "SVM Binary": "SVM Binary",
                "SVM Multiclass": "SVM Multiclass",
                "Logistic Binary": "Logistic Binary",
                "Logistic Multiclass OVR": "Logistic Multiclass OVR",
                "Logistic Multiclass Multinomial": "Logistic Multiclass Multinomial",
            }
            model_ID = model_ID_mapping.get(model_choice, None)


            # Make Prediction
            def prediction():
                species = predict_data(flower_data, model_name)
                if species == 0:
                    return 'setosa'
                if species == 1:
                    return 'versicolor'
                else:
                    return 'virginica'
                    
                
            # Dictionary mapping species to image path
            species_images = {
                "setosa": "images/setosa.jpg",
                "versicolor": "images/versicolor.jpg",
                "virginica": "images/virginica.jpg"
            }

            # Store data to MongoDB
            species_name = prediction()
            flower_data["Iris_Flower_predection_result"] = species_name
            flower_data["Model Name"] = model_ID
            st.markdown("**The processed input and predicted output:**")
            st.write(flower_data)
            document = {key: int(value) if isinstance(value,np.integer) else 
                        float(value) if isinstance(value, np.floating) else
                        value for key, value in flower_data.items()}
            
            collection.insert_one(document)


        # Display Result
        st.success(f"The predicted Species is {species_name}")

        # Display corresponding image
        if species_name in species_images:
            image = Image.open(species_images[species_name])
            st.image(image, caption=species_name, width=300)


    # Footer -- APP OWNER
    st.markdown(
        "<br><hr><center>Created by **Anu L Sasidharan** | Using Streamlit. </center></hr></br>",
        unsafe_allow_html=True
    )
    st.markdown("---") # Divider line

if __name__ == "__main__":
    # Run the streamlit app
    main()