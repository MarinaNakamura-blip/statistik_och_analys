import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Ladda in den tränade modellen och kolumnnamnen som jag sparade i Notebooken.
model = joblib.load("car_price_model.joblib")
train_columns = joblib.load("car_price_columns.joblib")

# Jag sätter en centered layout för att det är lättare att se för användaren.
st.set_page_config(page_title="Car Price Predictor", page_icon=":car:", layout="centered")

st.title("Welcome to the Car Price Predictor!🚗💨")
st.write(
    """
    Let's find out how much your car is likely worth!
         
    Please fill in the details below:
    """
    )
st.subheader("Car information")


# ================================================================================================================
# Användares input om bilen. 
# Jag har rangordnat inputen på samma sätt som i outputen från Notebooken så att jag inte glömmer variabler.
brand = st.selectbox("Brand", ["Ford", "Audi", "Volkswagen", "Honda", "Chevrolet", "BMW", "Hyundai", "Kia", "Toyota", "Mercedes"])

# Här kan användaren välja året med plus och minus knapparna mellan 2000 och 2023 såsom i datan, och jag har satt default värdet till 2010.
year = st.number_input("Year", min_value=2000, max_value=2023, value=2010)

# Jag valde att sätta step=0.5 och ber användaren att välja det närmaste numret mellan 1 och 5 för att det är samma range som i datan.
engine_size = st.number_input("Engine size: Please choose the closest number from 1 to 5", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

fuel_type = st.selectbox("Fuel type", ["Electric", "Diesel", "Hybrid", "Petrol"])

transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Automatic"])

mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000, step=1000)

doors = st.selectbox("Doors", [2, 3, 4, 5])

owner_count = st.selectbox("Number of previous owners", [1, 2, 3, 4, 5])
# ================================================================================================================


# Nu visar vi en sammanfattning av bilens information som användaren har matat in, så att de kan dubbelkolla att allt är korrekt innan vi gör prediktionen.
st.subheader("Summary of your car's information:")
st.info(
    f"""
    **Brand:** {brand}  
    **Year:** {year}  
    **Engine size:** {engine_size}  
    **Fuel type:** {fuel_type}  
    **Transmission:** {transmission}  
    **Mileage:** {mileage:,} km  
    **Doors:** {doors}  
    **Owner count:** {owner_count}
    """
)

# När användaren klickar på knappen "Predict the price!", skapas en DataFrame med samma kolumner som i träningen av modellen.
# Det är för att modellen förväntar sig en tabell form av input.
if st.button("Predict the price now!"):
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Year": year,
        "Engine_Size": engine_size,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Mileage": mileage,
        "Doors": doors,
        "Owner_Count": owner_count
    }])
    
    # Input_df har fortfarande de kategoriska variablerna, så här görs om dem till numeriska variabler.
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # Detta tvunger input_encoded att ha samma kolumner som i träningen av modellen, och om det saknas någon kolumn så fylls den med 0.
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)
    
    # Nu predikterar vi priset. 0 här är för att modellen returnerar en array, och vi vill ha det första (och enda) elementet i den arrayen.
    pred = model.predict(input_encoded)[0]
    
    # success är en grön box som visar resultatet på ett fint sätt.
    st.subheader("Predicted price")
    st.success(f"Your car is likely worth around {pred:,.0f} USD!")