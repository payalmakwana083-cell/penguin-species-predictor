import streamlit as st
import joblib
import pandas as pd

model = joblib.load('penguin_model.pkl')

st.title("🐧 Penguin Species Predictor")
st.write("This app uses Machine Learning to predict the species of a Palmer Penguin.")


st.sidebar.header("User Input Features")

def user_input_features():
    island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
    sex = st.sidebar.selectbox("Sex", ("MALE", "FEMALE"))
    culmen_l = st.sidebar.slider("Culmen Length (mm)", 32.1, 59.6, 43.9)
    culmen_d = st.sidebar.slider("Culmen Depth (mm)", 13.1, 21.5, 17.2)
    flipper = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
    body_m = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4200.0)
    
    
    island_map = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    sex_map = {'MALE': 0, 'FEMALE': 1}
    
    data = {
        'island': island_map[island],
        'culmen_length_mm': culmen_l,
        'culmen_depth_mm': culmen_d,
        'flipper_length_mm': flipper,
        'body_mass_g': body_m,
        'sex': sex_map[sex]
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()


st.subheader("Your Input Parameters")
st.write(input_df)


if st.button("Predict Species"):
    prediction = model.predict(input_df)
    
    st.subheader("Prediction Result")
    # Using a nice big display for the result
    st.success(f"The penguin is likely an **{prediction[0]}** species.")
