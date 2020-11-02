import streamlit as st        ##Streamlit is used to create beautiful, custom web apps for machine learning and data science.
import pandas as pd           ##used as open source data analysis and manipulation tool
import numpy as np            ##used for working with array
from fbprophet import Prophet ##Forcasting library(open source) released by facebook
st.title('Final project Slot for Python Developers')
st.set_option('deprecation.showfileUploaderEncoding',False)  ##now we are setting up the file uploader for uploading the csv file for prediction...
df = st.file_uploader('Import the time series csv file here. Columns must be labeled ds and y. The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.', type='csv', encoding='auto')
if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce')

    st.write(data) ###we are reading the csv file and we are just showing the whole data set and finding the max value of the date and printing that just after the dataset
    max_date = data['ds'].max()
    st.write(max_date)


periods_input = st.number_input('How many inputs you want to forecast into the future?',min_value = 1, max_value = 365)##How many days of prediction that you want
if df is not None:
    m = Prophet()##Initializing the model
    m.fit(data)##fitting the model

if df is not None:
    future = m.make_future_dataframe(periods = periods_input)##creating future dates for prediction

    forecast = m.predict(future) #This is the prediction
    fcst = forecast[['ds','yhat','yhat_lower','yhat_upper']] ##and we have predicted the lower ,upper and the predicted value....

    fcst_filtered = fcst[fcst['ds']> max_date]
    st.write(fcst_filtered)##Printing the values for predicted dates 
    fig1 = m.plot(forecast)##now we are plotting the predicted values
    st.write(fig1)

    fig2 = m.plot_components(forecast)##visualizing the components(trends and all)
    st.write(fig2)
