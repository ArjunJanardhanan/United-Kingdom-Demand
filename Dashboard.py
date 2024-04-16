import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc



import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pickle
from sklearn import  metrics
import numpy as np


# External CSS stylesheets
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']  # Update path to your CSS file


app = dash.Dash(__name__, external_stylesheets = external_stylesheets ,suppress_callback_exceptions=True)
server=app.server


# Define global styles
global_styles = {
    'backgroundColor': 'rgba(0, 0, 0, 0)',  # Transparent background
    'color': 'white',  # White font color
    'borderColor': 'white'  # White border color (optional)
}

drop_styles = {
    'backgroundColor': 'rgba(0, 0, 0, 0)',  # Transparent background
    'color': 'royalblue',  # White font color
    'borderColor': 'white'  # White border color (optional)
}



# Apply global styles to specific components
input_style = {**global_styles, 'border': '1px solid white'}
dropdown_style = global_styles


df = pd.read_csv('data.csv')
feature = df.columns[1:]


#reading weatherdata


import openmeteo_requests

import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 51.5085,
	"longitude": -0.1257,
	"past_days": 92,
	"forecast_days": 16,
	"hourly": "temperature_2m",
    "timezone": "Europe/London"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {"Date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["LONDON_TEMPERATURE"] = hourly_temperature_2m
hourly_london = pd.DataFrame(data = hourly_data)

params = {
	"latitude": 51.48,
	"longitude": -3.18,
	"past_days": 92,
	"forecast_days": 16,
	"hourly": "temperature_2m",
    "timezone": "Europe/London"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {"Date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["CARDIFF_TEMPERATURE"] = hourly_temperature_2m
hourly_cardiff = pd.DataFrame(data = hourly_data)



params = {
	"latitude": 55.8651,
	"longitude": -4.2576,
	"past_days": 92,
	"forecast_days": 16,
	"hourly": "temperature_2m",
    "timezone": "Europe/London"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {"Date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["GLASGOW_TEMPERATURE"] = hourly_temperature_2m
hourly_glasgow = pd.DataFrame(data = hourly_data)

from functools import reduce
# Merge weather data using reduce and lambda function
dfs = [hourly_london, hourly_cardiff, hourly_glasgow]
merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='inner'), dfs)


merged_df = merged_df.set_index ('Date', drop = True)
merged_df.index = merged_df.index.tz_convert(None)


# Get current datetime rounded to the last hour
current_datetime_utc = pd.Timestamp.utcnow().floor('h')

# Convert the UTC datetime to a naive datetime (without timezone information)
current_datetime_naive = current_datetime_utc.tz_convert(None)

# Get row based on current datetime
current_weather= merged_df.loc[current_datetime_naive]

temperature_values = current_weather.values.tolist()

temp_data = {
    'City': ['London','Cardiff','Glasgow'],
    'Lat': [51.5085, 51.48, 55.8651],
    'Lon': [-0.1257, -3.18, -4.2576]
}
temp_data['Temp(°C)'] = temperature_values
temp_df = pd.DataFrame(temp_data)






#reading demand data

import requests

# API endpoint URL
url = 'https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=177f6fa4-ae49-4182-81ea-0c6b35f26ca6'
limit = 100  # Number of records per page
offset = 0   # Initial offset

# List to store data frames
dfs = []

# Loop until all rows are fetched
while True:
    # Construct the URL with offset and limit parameters
    params = {'offset': offset, 'limit': limit}
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Read the JSON response
        json_data = response.json()
        
        # Extract records from the JSON response
        records = json_data.get('result', {}).get('records', [])
        
        # If no more records, exit the loop
        if not records:
            break
        
        # Convert records to DataFrame
        df_ = pd.DataFrame(records)
        
        # Append the DataFrame to the list
        dfs.append(df_)
        
        # Increment the offset for the next page
        offset += limit
    else:
        print(f"Error: {response.status_code} - {response.text}")
        break

if dfs:  # Check if dfs is not empty
    final_df = pd.concat(dfs, ignore_index=True)
    df_final = final_df.iloc[:, [1,2,3]]
else:
    final_df = pd.read_csv('demanddataupdate.csv')
    df_final = final_df.iloc[:, [0,1,2]]
 

df_final = df_final.drop(df_final[df_final['SETTLEMENT_PERIOD'] > 48].index)


# Define a function to convert half-hourly periods to time format
def convert_to_time(period):
    # Calculate hours and minutes based on the half-hourly period
    if period==48:
        hours=0
        minutes=0
    else:
        hours = period // 2
        minutes = (period % 2) * 30
    # Format the time as HH:MM
    return f'{hours:02d}:{minutes:02d}'

# Apply the conversion function to the 'HalfHourlyPeriod' column
df_final['Time'] = df_final['SETTLEMENT_PERIOD'].apply(convert_to_time)

# Concatenate SETTLEMENT_DATE and Time columns and convert to datetime
df_final['Date'] = pd.to_datetime(df_final['SETTLEMENT_DATE'] + ' ' + df_final['Time']+':00', format='%Y-%m-%d %H:%M:%S')
# Shift the datetime for rows where Time is midnight (00:00:00)
df_final.loc[df_final['Time'] == '00:00', 'Date'] += pd.DateOffset(days=1)
df_final = df_final.set_index ('Date', drop = True)
df_final = df_final.drop(['SETTLEMENT_PERIOD', 'SETTLEMENT_DATE','Time'], axis=1)

df_hourly = df_final.resample('h').sum()
df_hourly.rename(columns={'ND': 'NATIONAL_DEMAND'}, inplace=True)


df_data=pd.merge(df_hourly,merged_df, left_index=True, right_index=True)

#Hour of the day
df_data['HOUR']=df_data.index.hour

#Day of the week
df_data['DAY_OF_WEEK']=df_data.index.dayofweek

# Previous hour demand
df_data['NATIONAL_DEMAND-1']=df_data['NATIONAL_DEMAND'].shift(1)
df_data=df_data.dropna()


# First row where 'NATIONAL_DEMAND' is zero
first_zero_demand_row = df_data[df_data['NATIONAL_DEMAND'] == 0].iloc[0].iloc[1:]


next_hour=first_zero_demand_row.values

# Reshape the input array to be 2D with a single row for regression models
next_hour= next_hour.reshape(1, -1)
#print(next_hour)

df_data.drop(index=df_data[df_data['NATIONAL_DEMAND'] == 0].index, inplace=True)

last_hour = df_data.tail(1).index[0]

df_data.to_csv('2024data.csv')





#modeling and regression
feature1 = df_data.columns[0:]
df2=df_data.iloc[:,1:]
X2=df2.values
feature2 = df2.columns[0:]
y2=df_data['NATIONAL_DEMAND'].values

from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# Define feature selection methods
def filter_method(X2, y2):
    features = SelectKBest(k=5, score_func=mutual_info_regression)
    fit = features.fit(X2, y2)
    return fit.scores_

def wrapper_method(X2, y2):
    model = LinearRegression() 
    rfe = RFE(model,n_features_to_select=3)
    fit = rfe.fit(X2, y2)
    return fit.ranking_

def ensemble_method(X2, y2):
    model = RandomForestRegressor()
    model.fit(X2, y2)
    return model.feature_importances_





#Load and run LR model
with open('Lregr.pkl','rb') as file:
    LR_model=pickle.load(file)

y2_pred_LR = LR_model.predict(X2)
y_next_hour_LR= LR_model.predict(next_hour)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MBE_LR=np.mean(y2-y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)
NMBE_LR=MBE_LR/np.mean(y2)

#Load RF model
with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)

y2_pred_RF = RF_model.predict(X2)
y_next_hour_RF= RF_model.predict(next_hour)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MBE_RF=np.mean(y2-y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)
NMBE_RF=MBE_RF/np.mean(y2)

#Load and run SV model
with open('Sregr.pkl','rb') as file:
    SV_model=pickle.load(file)

y2_pred_SV = SV_model.predict(X2)
y_next_hour_SV= SV_model.predict(next_hour)

#Evaluate errors
MAE_SV=metrics.mean_absolute_error(y2,y2_pred_SV) 
MBE_SV=np.mean(y2-y2_pred_SV)
MSE_SV=metrics.mean_squared_error(y2,y2_pred_SV)  
RMSE_SV= np.sqrt(metrics.mean_squared_error(y2,y2_pred_SV))
cvRMSE_SV=RMSE_SV/np.mean(y2)
NMBE_SV=MBE_SV/np.mean(y2)

#Load and run DT model
with open('DT_regr_model.pkl','rb') as file:
    DT_model=pickle.load(file)

y2_pred_DT = DT_model.predict(X2)
y_next_hour_DT= DT_model.predict(next_hour)

#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y2,y2_pred_DT) 
MBE_DT=np.mean(y2-y2_pred_DT)
MSE_DT=metrics.mean_squared_error(y2,y2_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y2)
NMBE_DT=MBE_DT/np.mean(y2)

#Load and run GB model
with open('GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)

y2_pred_GB = GB_model.predict(X2)
y_next_hour_GB= GB_model.predict(next_hour)

#Evaluate errors
MAE_GB=metrics.mean_absolute_error(y2,y2_pred_GB) 
MBE_GB=np.mean(y2-y2_pred_GB)
MSE_GB=metrics.mean_squared_error(y2,y2_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y2)
NMBE_GB=MBE_GB/np.mean(y2)

#Load and run XGB model
with open('XGB_model.pkl','rb') as file:
    XGB_model=pickle.load(file)

y2_pred_XGB = XGB_model.predict(X2)
y_next_hour_XGB= XGB_model.predict(next_hour)

#Evaluate errors
MAE_XGB=metrics.mean_absolute_error(y2,y2_pred_XGB) 
MBE_XGB=np.mean(y2-y2_pred_XGB)
MSE_XGB=metrics.mean_squared_error(y2,y2_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y2)
NMBE_XGB=MBE_XGB/np.mean(y2)

#Load and run BT model
with open('BT_model.pkl','rb') as file:
    BT_model=pickle.load(file)

y2_pred_BT = BT_model.predict(X2)
y_next_hour_BT= BT_model.predict(next_hour)

#Evaluate errors
MAE_BT=metrics.mean_absolute_error(y2,y2_pred_BT) 
MBE_BT=np.mean(y2-y2_pred_BT)
MSE_BT=metrics.mean_squared_error(y2,y2_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y2)
NMBE_BT=MBE_BT/np.mean(y2)


#Load and run NN model
with open('NN_model.pkl','rb') as file:
    NN_model=pickle.load(file)

y2_pred_NN = NN_model.predict(X2)
y_next_hour_NN= NN_model.predict(next_hour)

#Evaluate errors
MAE_NN=metrics.mean_absolute_error(y2,y2_pred_NN) 
MBE_NN=np.mean(y2-y2_pred_NN)
MSE_NN=metrics.mean_squared_error(y2,y2_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y2,y2_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y2)
NMBE_NN=MBE_NN/np.mean(y2)

# Create data frames with predictin results and error metrics 
d = {'Methods': ['Linear Regression','Random Forest Regression','Support Vector Regression','Decision Tree Regression','Gradient Boosting Regression','Extreme Gradient Boosting Regression','Bootstrapping Regression', 'Neural Network Regression'], 'MAE': [MAE_LR, MAE_RF, MAE_SV, MAE_DT, MAE_GB, MAE_XGB, MAE_BT, MAE_NN],'MBE': [MBE_LR, MBE_RF, MBE_SV, MBE_DT, MBE_GB, MBE_XGB, MBE_BT, MBE_NN], 'MSE': [MSE_LR, MSE_RF, MSE_SV, MSE_DT, MSE_GB, MSE_XGB, MSE_BT, MSE_NN], 'RMSE': [RMSE_LR, RMSE_RF, RMSE_SV, RMSE_DT, RMSE_GB, RMSE_XGB, RMSE_BT, RMSE_NN],'cvMSE': [cvRMSE_LR, cvRMSE_RF, cvRMSE_SV, cvRMSE_DT, cvRMSE_GB, cvRMSE_XGB, cvRMSE_BT, cvRMSE_NN],'NMBE': [NMBE_LR, NMBE_RF, NMBE_SV, NMBE_DT, NMBE_GB, NMBE_XGB, NMBE_BT, NMBE_NN]}
df_metrics = pd.DataFrame(data=d)
df_metrics.set_index('Methods', inplace=True)


d={'Linear Regression': y2_pred_LR,'Random Forest Regression': y2_pred_RF,'Support Vector Regression': y2_pred_SV,'Decision Tree Regression': y2_pred_DT,'Gradient Boosting Regression': y2_pred_GB,'Extreme Gradient Boosting Regression': y2_pred_XGB,'Bootstrapping Regression': y2_pred_BT, 'Neural Network Regression': y2_pred_NN}
d_next_hour={'Linear Regression': y_next_hour_LR,'Random Forest Regression': y_next_hour_RF,'Support Vector Regression': y_next_hour_SV,'Decision Tree Regression': y_next_hour_DT,'Gradient Boosting Regression': y_next_hour_GB,'Extreme Gradient Boosting Regression': y_next_hour_XGB,'Bootstrapping Regression': y_next_hour_BT, 'Neural Network Regression': y_next_hour_NN}

df_forecast = pd.DataFrame(data=d, index=df_data.index)

regression = df_forecast.columns

# merge real and forecast results and creates a figure with it
df_data=pd.merge(df_data, df_forecast, left_index=True, right_index=True)
df_data=df_data.dropna()



# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ],style={'margin': 'auto'})



# Create the heatmap
heatmap_fig = px.imshow(
    df_metrics,
    color_continuous_scale='Viridis',  # Choose the colorscale (e.g., 'Viridis')
    color_continuous_midpoint=0,  # Set the midpoint for the color range

)

# Customize hover label format
heatmap_fig.update_traces(hovertemplate='Method: %{y}<br>Metric: %{x}<br>Value: %{z}')
# Set axis labels
heatmap_fig.update_layout(xaxis_title='Error type', yaxis_title='Methods', plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                                    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
                                    font=dict(color='white'))  # Font color)







# Define sidebar content
sidebar = html.Div(
    [   html.H4("MENU", style={"color": "white","margin-bottom": "60px", 'text-align': 'center'}),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("HOME", id="tab-1-link", href="#")),
                dbc.NavItem(dbc.NavLink("ORGANIZED DATA", id="tab-2-link", href="#")),
                dbc.NavItem(dbc.NavLink("FORECASTING NATIONAL DEMAND", id="tab-3-link", href="#")),
                dbc.NavItem(dbc.NavLink("NEXT HOUR PREDICTION", id="tab-4-link", href="#")),
                dbc.NavItem(dbc.NavLink("ERROR METRICS OF REGRESSION", id="tab-5-link", href="#")),
                dbc.NavItem(dbc.NavLink("EXPLORATORY DATA ANALYSIS", id="tab-6-link", href="#")),
                dbc.NavItem(dbc.NavLink("FEATURE SELECTION", id="tab-7-link", href="#")),
                dbc.NavItem(dbc.NavLink("ABOUT", id="tab-8-link", href="#")),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"background-color": "#050505", "height": "100vh", "padding": "20px"},
)

# Define main content
content = html.Div(
    id="page-content",
    children=[
        html.Div(
            [
                dbc.Button(
                    "☰ Toggle Sidebar",
                    id="sidebar-toggle",
                    color="dark",
                    className="mb-3",
                    style={"width": "100%","display": "flex", "justify-content": "center"},
                ),
            ],
            style={"padding": "20px"},
        ),
        html.Div("Content loading.....Choose an option in menu", id="main-content"),
        
    ],
)


# Define app layout with sidebar and main content
app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    sidebar,
                    width=3,
                    id="sidebar-col",
                    style={"height": "100vh", "overflowY": "auto"},
                ),
                dbc.Col(
                    content,
                    style={"height": "100vh", "overflowY": "auto"},
                ),
            ],
            style={"margin": "0px"},
        )
    ], style={"background-color": "#121212", "color": "white", "height": "100vh", "font-family": "Arial, sans-serif"}
)

# Callback to toggle sidebar visibility
@app.callback(
    Output("sidebar-col", "style"),
    [Input("sidebar-toggle", "n_clicks")],
)
def toggle_sidebar_visibility(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "none"}  # Hide the sidebar
    else:
        return {"display": "block"}  # Show the sidebar

# Callback to handle tab navigation
@app.callback(
    Output("main-content", "children"),
    [Input("tab-1-link", "n_clicks"), Input("tab-2-link", "n_clicks"), Input("tab-3-link", "n_clicks"), Input("tab-4-link", "n_clicks"), Input("tab-5-link", "n_clicks"), Input("tab-6-link", "n_clicks"), Input("tab-7-link", "n_clicks"), Input("tab-8-link", "n_clicks")],
)
def display_tab_content(*args):
    triggered_button = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    content_style = {"display": "flex", "justify-content": "center"}
    
    if triggered_button == "tab-1-link":
        return html.Div([
                            html.H1('United Kingdom Hourly Electricity Demand',style=content_style),
                            html.H6("Explore electricity demand in UK with weather",style=content_style),
                            html.Div("Historic Demand Data (2021 to 2023) and weather data (London, Cardiff & Glasgow) is used to predict National Demand",style=content_style),
                            html.Div("Go to another tab in sidebar to explore ", style={"font-style": "italic",'text-align': 'center'}),
                            html.Br(),
                            html.H6("Hover to see real-time temperature in the 3 cities", style={'text-align': 'center'}),
                            dcc.Graph(
                                id='uk-map',
                                figure=px.scatter_mapbox(
                                    temp_df,
                                    lat='Lat',
                                    lon='Lon',
                                    hover_name='City',
                                    hover_data={'City': False, 'Lat': True, 'Lon': True,'Temp(°C)': True },
                                    zoom=5,
                                ).update_layout(
                                    mapbox_style="carto-darkmatter",  # Other styles: "carto-positron", "carto-darkmatter", etc.
                                    mapbox_zoom=4,
                                    mapbox_center={"lat": 54, "lon": -2},
                                    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                                    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
                                    font=dict(color='white'),  # Font color
                                ),
                            ),
                            
                        ])
    elif triggered_button == "tab-2-link":
        return html.Div([
                            html.H4('United Kingdom Hourly Electricity & Weather Data (2021 to 2023)', style=content_style),
                            html.Label('Select the data-set to visualize in the plot below:',style=content_style),
                            dcc.Checklist(
                                id='feature_checklist',
                                options=[{'label': i, 'value': i} for i in feature],
                                value=['NATIONAL_DEMAND'],
                                inline=True,
                                style={'text-align': 'center'}
                            ),
                            dcc.Graph(
                                id='yearly-data',
                                
                            ),
                            html.Div("Note: Electric power is in MW & temperature in degree celcius", style={'text-align': 'center',"font-style": "italic"})
                        ])
    elif triggered_button == "tab-3-link":
        return  html.Div([
                            html.H4('Predicted and Actual National Demand in 2024',  style=content_style),
                            html.Label('Select the regression method to forecast and visualize in the plot below:', style=content_style),
                            dcc.Checklist(
                                id='forecast_checklist',
                                options=[{'label': i, 'value': i} for i in regression],
                                value=['Random Forest Regression'],
                                inline=True,
                                style={'text-align': 'center'}
                            ),
                            html.Br(),
                            html.Label('Select the real data or features to add in the plot below:', style=content_style),
                            dcc.Checklist(
                                id='real_checklist',
                                options=[{'label': i, 'value': i} for i in feature1],
                                value=['NATIONAL_DEMAND'],
                                inline=True,
                                style={'text-align': 'center'}
                            ),
                            dcc.Graph(
                                id='forecast_data'
                                # figure=fig2,
                                ),
                            html.Div("Note: Electric power is in MW & temperature in degree celcius (Real-time updation)", style={'text-align': 'center',"font-style": "italic"})
                        ])
    elif triggered_button == "tab-4-link":
        return html.Div([
                            html.H4('Next Hour National Demand Prediction', style=content_style),
                            html.Div([
                                        html.Span("Last recorded time is "),
				    	html.Span(" "),
                                        html.Span("(Real-time updation from "),
                                        html.Span(" "),
                                        html.A("National Grid ESO", href="https://www.nationalgrideso.com/", target="_blank"),  # Opens link in a new tab
                                        html.Span(f") : {last_hour}")
                                    ], style=content_style),
                            html.Br(),
                            html.Label('Select the regression method:', style=content_style),
                            dcc.RadioItems(
                                id='forecast_radio',
                                options=[{'label': i, 'value': i} for i in regression],
                                value='Random Forest Regression',
                                inline=True,
                                style={'text-align': 'center'}
                            ),
                            html.Br(),
                            html.Br(),
                            html.Div(id='predicted_value_output', style={'text-align': 'center',"font-weight": "bold"})  # This is where the predicted value will be displayed
                        ])
    elif triggered_button == "tab-5-link":
        return html.Div([
                            html.H4('Error metrics for different regression methods', style={'text-align': 'center'}),
                            dcc.Graph(figure=heatmap_fig),
                            html.Div("Note: Hover to see actual values", style={'text-align': 'center',"font-style": "italic"})
                        ])
    elif triggered_button == "tab-6-link":
        return html.Div([
                            html.H4('Exploratory analysis of real data', style={'text-align': 'center'}),
                            html.Label('Select the variable:'),
                            dcc.RadioItems(
                                id='feature-radio',
                                options=[{'label': f, 'value': f} for f in feature],
                                value=feature[0],  # Default value
                                inline=True
                            ),
                            html.Br(),
                            html.Label('Select the plot type:'),
                            dcc.Dropdown(id='plot-type',
                                         options=[{'label': 'Histogram', 'value': 'histogram'},
                                                  {'label': 'Boxplot', 'value': 'boxplot'}],
                                         value='histogram',style=drop_styles), 
                            dcc.Graph(id='plot')
                        ])
    elif triggered_button == "tab-7-link":
        return html.Div([
                            html.H4('Visualizing the significance of features in modeling',style={'text-align': 'center'}),
                            html.Label('Select the feature selection method:'),           
                            dcc.Dropdown(
                                id='feature-method-dropdown',
                                options=[
                                    {'label': 'Filter Method (KBest)', 'value': 'filter'},
                                    {'label': 'Wrapper Method (RFE)', 'value': 'wrapper'},
                                    {'label': 'Ensemble Method', 'value': 'ensemble'}
                                ],
                                value='filter',
                                style=drop_styles,
                            ),
                            html.Div(id='feature-selection-results'),
                            html.Br(),
                            html.Div("Different features where analyzed for regression models, above six features where used for prediction and National Demand in the past hour had highest influence", style={'text-align': 'center',"font-style": "italic"})
                            
                        ])
    elif triggered_button == "tab-8-link":
        return html.Div([   html.Div("Weather data has a significant role in national demand and was collected from capital cities."),
                            html.Div("2020 data was not considered due to Covid lockdown."),
                            html.Div("Demand data was available since 2006 but data from 2021 was explored considering similar capacity and generation."),
                            html.Div([
                                        html.Span("National Demand Data is obtained real-time from "),
                                        " ",
                                        html.A(" National Grid ESO", href="https://www.nationalgrideso.com/", target="_blank"),  # Opens link in a new tab
                                        "."
                                    ]),
                            
                            html.Div([
                                        html.Span("Weather Data is obtained real-time from "),
                                        " ",
                                        html.A(" Open-meteo", href="https://open-meteo.com/", target="_blank"),  # Opens link in a new tab
                                        "."
                                    ]),
                            html.Div([
                                        html.Span("Visit"),
                                        " ",
                                        html.A("GitHub repository", href="https://github.com/ArjunJanardhanan/United-Kingdom-Demand", target="_blank"),
                                        " ",
                                        html.Span("for the source code and Jupyter notebook used to for data preperation and regression.")
                                    ]),
                            html.Div("Developed as part of Energy services course project at IST Lisbon by Arjun Janardhanan", style={"font-style": "italic"})
                        ])
    else:
        return html.Div([
                            html.H1('United kingdom Hourly Electricity Demand',style=content_style),
                            html.H6("Explore electricity demand in UK with weather",style=content_style),
                            html.Div("Historic Demand Data (2021 to 2023) and weather data (London, Cardiff & Glasgow) is used to predict National Demand",style=content_style),
                            html.Div("Go to another tab in sidebar to explore ", style={"font-style": "italic",'text-align': 'center'}),
                            html.Br(),
                            html.H6("Hover to see real-time temperature in the 3 cities", style={'text-align': 'center'}),
                            dcc.Graph(
                                id='uk-map',
                                figure=px.scatter_mapbox(
                                    temp_df,
                                    lat='Lat',
                                    lon='Lon',
                                    hover_name='City',
                                    hover_data={'City': False, 'Lat': True, 'Lon': True,'Temp(°C)': True },
                                    zoom=5,
                                ).update_layout(
                                    mapbox_style="carto-darkmatter",  # Other styles: "carto-positron", "carto-darkmatter", etc.
                                    mapbox_zoom=4,
                                    mapbox_center={"lat": 54, "lon": -2},
                                    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                                    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
                                    font=dict(color='white')  # Font color
                                ),
                            ),
                            
                        ])

    
@app.callback(
    Output('yearly-data', 'figure'),
    [Input('feature_checklist', 'value')]
)
def update_graph1(selected_features):
    traces = []
    for f in selected_features:
        # trace = px.line(df, x="date", y=selected_features)
        trace = go.Scatter(x=df['date'], y=df[f], mode='lines', name=f)
        traces.append(trace)
    # Create the figure object
    fig = {'data': traces, 'layout': {'title': 'Selected plots',
                    'font': 'white',  # Apply global styles to font
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Transparent plot background
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'}}  # Transparent paper background
    return fig


@app.callback(
    dash.dependencies.Output('forecast_data', 'figure'),
    [dash.dependencies.Input('forecast_checklist', 'value'),
     dash.dependencies.Input('real_checklist', 'value'),
      ]
)
def update_graph2(selected_features,selected_feature):
    traces = []
    for f in selected_features:
        trace = go.Scatter(x=df_data.index, y=df_data[f], mode='lines', name=f)
        traces.append(trace)      
    for f in selected_feature:
        trace = go.Scatter(x=df_data.index, y=df_data[f], mode='lines', name=f)
        traces.append(trace)   
    # Create the figure object
    fig = {'data': traces, 'layout': {'title': 'Selected plots',
        'font': 'white',  # Apply global styles to font
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Transparent plot background
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'} } # Transparent paper background
    return fig


# Define callback to update the predicted value display
@app.callback(
    Output('predicted_value_output', 'children'),
    [Input('forecast_radio', 'value')]
)
def update_predicted_value(selected_method):
    if selected_method in d_next_hour:
        predicted_value = d_next_hour[selected_method]
        return f'Predicted National Demand for next hour using {selected_method}: {predicted_value} MW'
    else:
        return 'DATA UNAVAILABLE'  # Return empty string if the selected method is not found in the dictionary



# Callback to update histogram and boxplot based on selected feature
@app.callback(
    Output('plot', 'figure'),
    [Input('feature-radio', 'value'),
     Input('plot-type', 'value')]
)
def update_plots(selected_feature, plot_type):
    
    if plot_type == 'histogram':
        # Create histogram
        fig3 = px.histogram(df, x=selected_feature, title=f'Histogram for {selected_feature}')
    else:
        # Create boxplot
        fig3 = px.box(df, y=selected_feature, title=f'Boxplot for {selected_feature}')
        
        # Apply global styles to the entire figure
    fig3.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        font=dict(color='white'),  # Font color
    )
    
    return fig3

@app.callback(
    Output('feature-selection-results', 'children'),
    Input('feature-method-dropdown', 'value')
)
def perform_feature_selection(method):
    if method == 'filter':
        scores = filter_method(X2, y2)
        return html.Div([
            dcc.Graph(figure={
                'data': [go.Bar(x=feature2, y=scores)],
                'layout': {'title': 'Feature Importance Scores for Filter Method',
                    'font': 'white',  # Apply global styles to font
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Transparent plot background
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'}  # Transparent paper background
            })
        ])
    elif method == 'wrapper':
        rankings = wrapper_method(X2, y2)
        return html.Div([
            dcc.Graph(figure={
                'data': [go.Bar(x=feature2, y=rankings)],
                'layout': {'title': 'Feature Rankings for Wrapper Method',
                    'font': 'white',  # Apply global styles to font
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Transparent plot background
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)' } # Transparent paper background
            })
        ])
       
    elif method == 'ensemble':
        importances = ensemble_method(X2, y2)
        return html.Div([
            dcc.Graph(figure={
                'data': [go.Bar(x=feature2, y=importances)],
                'layout': {'title': 'Feature Importances for Ensemble Method',
                    'font': 'white',  # Apply global styles to font
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Transparent plot background
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'}  # Transparent paper background
            })
        ])



if __name__ == "__main__":
    app.run_server(debug=False)
