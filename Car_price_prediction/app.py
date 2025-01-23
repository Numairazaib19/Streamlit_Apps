# import libraries
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from st_aggrid import AgGrid
import joblib

# Get the current date and time
date_time = datetime.datetime.now()

# Load the XGBoost model from a saved file
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

# Define the main function
def main(): 
    # Apply custom CSS styles to the web app
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 0rem;
                }
                .css-1544g2n.e1akgbir4 {
                margin-top: -75px;
                }
        </style>
        """, unsafe_allow_html=True)
    
    # Create a title section with a background color and centered text
    html_temp="""
     <div style = "background-color:#5453A6;padding:16px">
     <h2 style="color:#ffffff;text-align:center;"> CAR PRICE PREDICTION </h2>
     </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
   # Create a sidebar with input fields for car price prediction
    st.sidebar.markdown("##### Are you planning to sell your car ?\n##### So let's try evaluating the price..")
    st.write('')
    st.write('')
    p1 = st.sidebar.number_input('What is the current ex-showroom price of the car ?  (In Lakhs)',2.5,25.0,step=1.0) 
    p2 = st.sidebar.number_input('What is distance completed by the car in Kilometers ?',100,50000000,step=100)

    # Dropdown for selecting fuel type
    s1 = st.sidebar.selectbox('What is the fuel type of the car ?',('Petrol','Diesel', 'CNG'))
    if s1=="Petrol":
        p3=0
    elif s1=="Diesel":
        p3=1
    elif s1=="CNG":
        p3=2
        
    # Dropdown for selecting seller type    
    s2 = st.sidebar.selectbox('Are you a dealer or an individual ?', ('Dealer','Individual'))
    if s2=="Dealer":
        p4=0
    elif s2=="Indivisual":
        p4=1
        
    # Dropdown for selecting transmission type     
    s3 = st.sidebar.selectbox('What is the Transmission Type ?', ('Manual','Automatic'))
    if s3=="Manual":
        p5=0
    elif s3=="Automatic":
        p5=1
        
    # Slider for selecting the number of owners the car previously had    
    p6 = st.sidebar.slider("Number of Owners the car previously had",0,3)
    
    # Number input for the year the car was purchased
    years = st.sidebar.number_input('In which year car was purchased ?',1990,date_time.year,step=1)
    p7 = date_time.year-years
    
    # Create a DataFrame with user inputs
    data_new = pd.DataFrame({
    'Present_Price':p1,
    'Kms_Driven':p2,
    'Fuel_Type':p3,
    'Seller_Type':p4,
    'Transmission':p5,
    'Owner':p6,
    'Age':p7
},index=[0])
    try: 
        # If the "Predict" button is clicked, make a prediction
        if st.sidebar.button('Predict'):
            prediction = model.predict(data_new)
            if prediction>0:
                st.balloons()
                st.success('You can sell the car for {:.2f} lakhs'.format(prediction[0]))
            else:
                st.warning("You will be not able to sell this car !!")
    except:
        st.warning("Opps!! Something went wrong\nTry again")

if __name__ == '__main__':
    main()
    
# Read a CSV file into a Pandas DataFrame    
df = pd.read_csv("car data.csv")
#st.dataframe(df.style)

# Display the data using the AgGrid interactive data table
AgGrid(df)


# Read the car data from a CSV file
car_data = pd.read_csv("car data.csv")

# Set Matplotlib style
style.use('ggplot')

# Create a Streamlit app section for visualizing categorical data columns
st.subheader('Visualizing Categorical Data Columns')

# Define the list of categorical columns you want to visualize
selected_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']

# Create a figure
fig, axes = plt.subplots(1, len(selected_columns), figsize=(15, 5))

# Iterate through selected columns and plot
for i, column in enumerate(selected_columns):
    ax = axes[i]
    ax.bar(car_data[column], car_data['Selling_Price'], color='royalblue')
    ax.set_xlabel(column)
    ax.set_ylabel('Selling Price')
    ax.set_title(f'{column} vs Selling Price')

# Display the figure using Streamlit
st.pyplot(fig)


file_path = 'car data.csv'  # Replace with the actual file path if you want
car_data = pd.read_csv(file_path)

# Display the original data
#st.subheader("Original Data")
#st.write(car_data)

# Manual encoding
fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
car_data['Fuel_Type'].replace(fuel_type_mapping, inplace=True)

# One-hot encoding
car_data = pd.get_dummies(car_data, columns=['Seller_Type', 'Transmission'], drop_first=True)

# Display the encoded data
#st.subheader("Encoded Data")
#st.write(car_data)

# Create a correlation heatmap
st.subheader("Correlation Heatmap")
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(10, 7))
sns.heatmap(car_data.corr(), annot=True)
plt.title('Correlation between the columns')
st.pyplot()
