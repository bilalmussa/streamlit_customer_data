# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:11:01 2021

@author: bmussa
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
import base64
from io import BytesIO
import smtplib

def get_table_download_link_csv(df, message):
    csv = df.to_csv(index=False).encode('utf-8-sig')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{df.name}.csv" target="_blank">{message}</a>'
    return href

st.set_page_config(layout="wide")

matplotlib.use("agg")

sns.set_style('white')

col_spacer1, col1, col_spacer2, col2, col_spacer3 = st.columns(
    (.1, 2, .2, 2, .1))

col1.title('Analyse your customer data')
col2.subheader('https://www.linkedin.com/in/bilalmussa/ ')

st.markdown("Hello, this is Bilal, and welcome to my Quick Analysis app. We just need 4 columns from your transactional data for (ideally) the last 3 years, and the app will automaically profile your customers and will display various statistics and charts for your analysis.")
st.markdown("The system reads the data, analyses it, and displays the results. No data is stored during the process.")
 
st.write("We accept the data in a CSV format with the following column headers:")
st.write("- OrderDateTime: Date and Time of the order in DD/MM/YYYY format")
st.write("- ItemCost: Cost of the transaction in a float format")
st.write("- CustomerID: Unique ID of the customer - can be email address or customer key")
st.write("- OrderID: Unique ID of the order - can be text or numeric or mixed")

example_dict = {'OrderDateTime': ['31/05/2021','29/05/2021','05/05/2021'],
                'ItemCost': [5.6,7.7,10],
                'CustomerID': [1111,222222,333333],
                'OrderID' : ['abc1', 'efg3', 'hij4']
                }

example_data = pd.DataFrame(example_dict)
example_data.name = 'example_data'

st.dataframe(example_data)
st.markdown(get_table_download_link_csv(example_data,"Click here to download a sample CSV template"), unsafe_allow_html=True)
st.markdown("The app is preloaded with the above example data")


@st.cache_data
def tidy_data(data):
    #clean up some of the data where needed
    data['OrderDate'] = pd.to_datetime(data['OrderDateTime'], dayfirst=True).dt.date
    #fill blank order IDs with 1
    data['OrderID'] = data['OrderID'].replace('nan', np.nan).fillna(1)
    data['ItemCost'] = pd.to_numeric(data['ItemCost'],errors='coerce')
    return data


user_input = st.file_uploader("Upload CSV",type=['csv'])

# Create a text element and let the reader know the data is loading.
if not user_input:
    trans_data = tidy_data(example_data)
else:
    with st.spinner('Waiting to load data...'):
        trans_data = tidy_data(pd.read_csv(user_input, parse_dates=['OrderDateTime'], dayfirst=True))
        # Notify the reader that the data was successfully loaded.
        st.success('Loading data...done!')

#get max date from data series
max_date = trans_data['OrderDate'].max()
last_year = (datetime.strptime(str(trans_data['OrderDate'].max()),"%Y-%m-%d")+ timedelta(days=-365)).date()
last_year_1 = (datetime.strptime(str(trans_data['OrderDate'].max()),"%Y-%m-%d")+ timedelta(days=-730)).date()


@st.cache_data
def data_calcs(trans_data):
    last12data = trans_data[trans_data['OrderDate']>=last_year]
    
    last12data['months'] = pd.to_datetime(last12data['OrderDate']).dt.strftime('%Y-%m')
    
    last12months_pivot = pd.pivot_table(data = last12data, values='OrderID', index='CustomerID', columns='months', aggfunc='count')
    last12months_pivot = last12months_pivot.replace('nan', np.nan).fillna(0)
    
    list_of_cols = list(last12months_pivot.columns)
    last12months_pivot= pd.DataFrame(last12months_pivot)
    for col in list_of_cols:
        last12months_pivot[col] = last12months_pivot[col].apply(lambda x: 1 if x>0 else 0).astype(np.uint8)
        
    agg_data = trans_data.groupby(['CustomerID']).agg({'ItemCost': ['sum']
                                                     ,'OrderID': ['count']
                                                     ,'OrderDate': ['min','max']}
                                                     ).reset_index()
    
    agg_data.columns = ['CustomerID', 'TotalSpend', 'TotalOrders', 'FirstOrder','LastOrder']
    
    agg_data['ATV'] = agg_data['TotalSpend']/agg_data['TotalOrders']
    
    agg_data['AvgTimePOrder'] = ((agg_data['LastOrder'] -agg_data['FirstOrder'])/np.timedelta64(1,'D'))/agg_data['TotalOrders']
    
    condlist = [(agg_data.FirstOrder>=last_year),
                (agg_data.LastOrder>=last_year)&(agg_data.FirstOrder<last_year_1),
                (agg_data.FirstOrder<last_year)&(agg_data.LastOrder>=last_year),
                (agg_data.LastOrder<last_year)
                ]
    
    choicelist = ['New'
                  , 'Reactivated'
                  , 'Active'
                  ,'Lapsed']
    
    agg_data['CustStatus'] = np.select(condlist, choicelist, default='unknown')
    
    condlist = [(agg_data.TotalOrders>1)&(agg_data.AvgTimePOrder<=7),
                (agg_data.TotalOrders>1)&(agg_data.AvgTimePOrder<=30),
                (agg_data.TotalOrders>1)&(agg_data.AvgTimePOrder<=180),
                (agg_data.TotalOrders>1)&(agg_data.AvgTimePOrder<=365),
                (agg_data.TotalOrders>1)&(agg_data.AvgTimePOrder>365),
                ]
    
    choicelist = ['1. Weekly'
                  , '2. Monthly'
                  ,'3. Bi Annually'
                  , '4. Annually'
                  , '5. More than a year']
    
    agg_data['AvgTimepOrderBand'] = np.select(condlist, choicelist, default='unknown')
    
    condlist = [(agg_data.ATV<20),
                (agg_data.ATV<50),
                (agg_data.ATV<=100),
                (agg_data.ATV<=250),
                (agg_data.ATV<=500),
                (agg_data.ATV>500),
                ]
    
    choicelist = ['1. < £20'
                  , '2. £20 - £50'
                  , '3. £50 - £100'
                  , '4. £100 - £250'
                  , '5. £250 - £500'
                  , '6. £500+']
    
    agg_data['ATVBand'] = np.select(condlist, choicelist, default='unknown')
    
    condlist = [(agg_data.TotalSpend<50),
                (agg_data.TotalSpend<100),
                (agg_data.TotalSpend<200),
                (agg_data.TotalSpend<500),
                (agg_data.TotalSpend<1000),
                ]
    
    choicelist = ['1. < £50'
                  , '2. £50 - £100'
                  , '3. £100 - £200'
                  , '4. £200 - £500'
                  , '5. £500 - £1000']
    
    agg_data['TotalSpendBand'] = np.select(condlist, choicelist, default='6. More than £1000')
    
    agg_data = pd.merge(agg_data, last12months_pivot, how='left', left_on='CustomerID', right_on='CustomerID')
    
    last12data_spend = last12data.groupby(last12data['CustomerID']).agg({'ItemCost': ['sum']}).reset_index()
    
    last12data_spend.columns = ['CustomerID', 'Spend12m']
    
    agg_data = agg_data.merge(last12data_spend,how='left', on='CustomerID')
    
    agg_data['monthsSpent'] = agg_data[list_of_cols].sum(axis=1)
    
    agg_data['monthsSpent'] = agg_data['monthsSpent'].replace('nan', np.nan).fillna(0)
    
    condlist = [(agg_data.CustStatus=='New'),
                (agg_data.monthsSpent>9),
                (agg_data.monthsSpent>3),
                ((agg_data.monthsSpent<=3) & (agg_data.monthsSpent>0)),
                ]
    
    choicelist = ['0. New Customer'
                  ,'1. High Loyal'
                  , '2. Med Loyal'
                  , '3. Low Loyal'
                  ]
    
    agg_data['LoyaltyBand'] = np.select(condlist, choicelist, default='4. No Spender')
    
    condlist = [(agg_data.Spend12m<250),
                (agg_data.Spend12m<1000),
                (agg_data.Spend12m>=1000),
                ]
    
    choicelist = ['1. < £250'
                  , '2. £250 - £1000'
                  , '3. >£1000'
                  ]
    
    agg_data['Last12mSpendBand'] = np.select(condlist, choicelist, default='7. No Spender')
    return agg_data

agg_data = data_calcs(trans_data)
