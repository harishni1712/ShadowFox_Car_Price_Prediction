import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("car_selling_price_model.pkl")
feature_columns = joblib.load("car_selling_price_features.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Car Selling Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
    }

    .result-price {
        font-size: 36px;
        font-weight: 700;
    }

    .info-text {
        text-align: center;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">🚗 Car Selling Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Estimate the approximate selling price of your car using Machine Learning</div>',
    unsafe_allow_html=True
)


st.divider()


# --------------------------------------------------
# Car Details
# --------------------------------------------------

st.subheader("Enter Car Details")


# First row
col1, col2 = st.columns(2)

with col1:
    car_age = st.number_input(
        "Car Age (Years)",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )

with col2:
    present_price = st.number_input(
        "Present Price (Lakhs)",
        min_value=0.1,
        max_value=100.0,
        value=8.5,
        step=0.1
    )


# Second row
col3, col4 = st.columns(2)

with col3:
    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=1000000,
        value=30000,
        step=1000
    )

with col4:
    owner = st.selectbox(
        "Previous Owners",
        [0, 1, 2, 3]
    )


# Third row
col5, col6 = st.columns(2)

with col5:
    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

with col6:
    seller_type = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual"]
    )


# Fourth row
col7, col8 = st.columns(2)

with col7:
    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

with col8:
    st.write("")
    st.write("")
    st.caption("All prices are in Indian Rupees.")


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Selling Price",
    use_container_width=True
):

    # Create input data
    input_data = pd.DataFrame({
        "Car_Age": [car_age],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Owner": [owner],
        "Fuel_Type_Diesel": [fuel_type == "Diesel"],
        "Fuel_Type_Petrol": [fuel_type == "Petrol"],
        "Seller_Type_Individual": [seller_type == "Individual"],
        "Transmission_Manual": [transmission == "Manual"]
    })

    # Ensure correct feature order
    input_data = input_data[feature_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Prevent negative price display
    prediction = max(prediction, 0)

    # Display result
    st.success("Prediction generated successfully!")

    st.markdown(
        f"""
        <div class="result-box">
            <div>Estimated Selling Price</div>
            <div class="result-price">₹{prediction:.2f} Lakhs</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-text">This is an approximate ML-based prediction and may vary from the actual market price.</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.divider()

st.subheader("About the Model")

st.write(
    """
    This application uses a **Gradient Boosting Regression model**
    trained on car-related features such as present price, car age,
    kilometers driven, fuel type, seller type, transmission, and
    previous ownership.
    """
)

st.caption(
    "Developed as part of the ShadowFox Machine Learning Engineer Internship."
)