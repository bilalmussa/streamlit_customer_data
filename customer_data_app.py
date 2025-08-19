# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:11:01 2021
Last updated on Tue Aug 19 2025

@author: bmussa (Updated by Gemini)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Customer Data Analysis")

# --- Helper Functions ---

@st.cache_data
def tidy_data(df):
    """Cleans and preprocesses the raw dataframe."""
    df['OrderDate'] = pd.to_datetime(df['OrderDateTime'], dayfirst=True, errors='coerce')
    #df['OrderDate'] = pd.to_datetime(df['OrderDateTime'], dayfirst=True).dt.date
    df['OrderID'] = df['OrderID'].replace('nan', np.nan).fillna(1)
    df['ItemCost'] = pd.to_numeric(df['ItemCost'], errors='coerce')
    return df

@st.cache_data
def perform_customer_analysis(df):
    """Performs all the calculations and customer segmentation."""
    max_date = df['OrderDate'].max()
    last_year_start = max_date - timedelta(days=365)
    two_years_ago_start = max_date - timedelta(days=730)

    # --- Last 12 Months Activity ---
    last12data = df[df['OrderDate'] >= last_year_start].copy()
    last12data['months'] = pd.to_datetime(last12data['OrderDate']).dt.strftime('%Y-%m')
    
    last12months_pivot = pd.pivot_table(
        data=last12data,
        values='OrderID',
        index='CustomerID',
        columns='months',
        aggfunc='count',
        fill_value=0
    )
    # Ensure all columns are binary (0 or 1)
    last12months_pivot = last12months_pivot.applymap(lambda x: 1 if x > 0 else 0).astype(np.uint8)
    month_cols = list(last12months_pivot.columns)

    # --- Aggregate Customer Data ---
    agg_data = df.groupby('CustomerID').agg(
        TotalSpend=('ItemCost', 'sum'),
        TotalOrders=('OrderID', 'count'),
        FirstOrder=('OrderDate', 'min'),
        LastOrder=('OrderDate', 'max')
    ).reset_index()

    agg_data['ATV'] = agg_data['TotalSpend'] / agg_data['TotalOrders']
    # Avoid division by zero for single-order customers
    time_diff = (agg_data['LastOrder'] - agg_data['FirstOrder']).dt.days
    agg_data['AvgTimePOrder'] = np.where(agg_data['TotalOrders'] > 1, time_diff / (agg_data['TotalOrders'] - 1), 0)


    # --- Customer Segmentation (Status, Frequency, Value) ---
    cond_status = [
        (agg_data.FirstOrder >= last_year_start),
        (agg_data.LastOrder >= last_year_start) & (agg_data.FirstOrder < two_years_ago_start),
        (agg_data.FirstOrder < last_year_start) & (agg_data.LastOrder >= last_year_start),
        (agg_data.LastOrder < last_year_start)
    ]
    choice_status = ['New', 'Reactivated', 'Active', 'Lapsed']
    agg_data['CustStatus'] = np.select(cond_status, choice_status, default='Unknown')

    cond_freq = [
        (agg_data.TotalOrders > 1) & (agg_data.AvgTimePOrder <= 7),
        (agg_data.TotalOrders > 1) & (agg_data.AvgTimePOrder <= 30),
        (agg_data.TotalOrders > 1) & (agg_data.AvgTimePOrder <= 180),
        (agg_data.TotalOrders > 1) & (agg_data.AvgTimePOrder <= 365),
        (agg_data.TotalOrders > 1) & (agg_data.AvgTimePOrder > 365)
    ]
    choice_freq = ['1. Weekly', '2. Monthly', '3. Bi-Annually', '4. Annually', '5. > Annually']
    agg_data['AvgTimepOrderBand'] = np.select(cond_freq, choice_freq, default='Single Order')
    
    # Define bands for ATV and Total Spend
    atv_bins = [-np.inf, 20, 50, 100, 250, 500, np.inf]
    atv_labels = ['1. < £20', '2. £20-£50', '3. £50-£100', '4. £100-£250', '5. £250-£500', '6. £500+']
    agg_data['ATVBand'] = pd.cut(agg_data['ATV'], bins=atv_bins, labels=atv_labels, right=False)

    spend_bins = [-np.inf, 50, 100, 200, 500, 1000, np.inf]
    spend_labels = ['1. < £50', '2. £50-£100', '3. £100-£200', '4. £200-£500', '5. £500-£1000', '6. > £1000']
    agg_data['TotalSpendBand'] = pd.cut(agg_data['TotalSpend'], bins=spend_bins, labels=spend_labels, right=False)

    # --- Merge Monthly Data and Finalize ---
    agg_data = pd.merge(agg_data, last12months_pivot, on='CustomerID', how='left')
    
    last12data_spend = last12data.groupby('CustomerID')['ItemCost'].sum().reset_index(name='Spend12m')
    agg_data = agg_data.merge(last12data_spend, on='CustomerID', how='left')
    agg_data['Spend12m'] = agg_data['Spend12m'].fillna(0)

    agg_data['monthsSpent'] = agg_data[month_cols].sum(axis=1)

    cond_loyalty = [
        (agg_data.CustStatus == 'New'),
        (agg_data.monthsSpent > 9),
        (agg_data.monthsSpent > 3),
        (agg_data.monthsSpent > 0)
    ]
    choice_loyalty = ['0. New Customer', '1. High Loyal', '2. Med Loyal', '3. Low Loyal']
    agg_data['LoyaltyBand'] = np.select(cond_loyalty, choice_loyalty, default='4. No Spender (L12M)')

    spend12m_bins = [-np.inf, 250, 1000, np.inf]
    spend12m_labels = ['1. < £250', '2. £250-£1000', '3. > £1000']
    agg_data['Last12mSpendBand'] = pd.cut(agg_data['Spend12m'], bins=spend12m_bins, labels=spend12m_labels, right=False)
    agg_data['Last12mSpendBand'] = agg_data['Last12mSpendBand'].cat.add_categories('0. No Spend').fillna('0. No Spend')
    
    # Fill NA for month columns for customers with no spend in the last 12 months
    agg_data[month_cols] = agg_data[month_cols].fillna(0).astype(int)

    return agg_data

def create_bar_chart(data, x_col, y_col, title):
    """Generates a styled bar chart using Seaborn."""
    fig, ax = plt.subplots()
    sns.barplot(data=data, x=x_col, y=y_col, color="goldenrod", ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45, ha='right')
    return fig

# --- App Layout ---

# -- Header --
col1, col2 = st.columns([3, 1])
with col1:
    st.title('Analyse Your Customer Data')
with col2:
    st.subheader('By Bilal Mussa')
    st.write("[LinkedIn Profile](https://www.linkedin.com/in/bilalmussa/)")

st.markdown("""
Hello! This app automatically profiles your customers from your transactional data. 
Just provide a CSV with four specific columns, and the app will generate key statistics and charts. 
**No data is stored in the process.**
""")

# -- Instructions and Template --
with st.expander("Click here for data format instructions and to download a template", expanded=True):
    st.markdown("""
    Please upload a CSV file with the following **exact** column headers:
    - **OrderDateTime**: Date and time of the order (e.g., `DD/MM/YYYY HH:MM` or `DD/MM/YYYY`).
    - **ItemCost**: The transaction value (numeric format, e.g., `10.50`).
    - **CustomerID**: A unique identifier for the customer.
    - **OrderID**: A unique identifier for the order.
    """)
    
    example_dict = {
        'OrderDateTime': ['31/05/2021', '29/05/2021', '05/05/2021'],
        'ItemCost': [5.6, 7.7, 10],
        'CustomerID': [1111, 222222, 333333],
        'OrderID': ['abc1', 'efg3', 'hij4']
    }
    example_df = pd.DataFrame(example_dict)
    st.dataframe(example_df)
    
    st.download_button(
       label="Download CSV Template",
       data=example_df.to_csv(index=False).encode('utf-8'),
       file_name='example_template.csv',
       mime='text/csv',
    )

# --- Data Upload and Processing ---
user_input = st.file_uploader("Upload your CSV file here", type=['csv'])

if user_input:
    trans_data = tidy_data(pd.read_csv(user_input))
else:
    st.info("Using preloaded example data. Upload your own CSV file to analyse.")
    trans_data = tidy_data(example_df)

# Perform the main analysis
agg_data = perform_customer_analysis(trans_data)

# --- Display Initial Stats ---
st.header("Data Overview")
with st.expander("View Raw Data and Summary Statistics"):
    max_date = trans_data['OrderDate'].max()
    st.subheader('Raw Data')
    st.dataframe(trans_data)
    
    st.subheader('Data Summary')
    st.write(f"There are **{len(trans_data):,}** records in the data from **{trans_data['CustomerID'].nunique():,}** unique customers.")
    st.write(f"The latest order date is **{max_date.strftime('%d %B %Y')}**.")
    st.write("Below is a summary of the **ItemCost** column:")
    
    data_description = trans_data['ItemCost'].agg(['count', 'mean', 'sum', 'min', 'max', 'median']).reset_index()
    data_description.columns = ['Metric', 'Value']
    st.dataframe(data_description)

# --- Interactive Analysis ---
st.header("Interactive Customer Analysis")
st.write("Select a dimension to segment your customer base and see how key metrics change across different groups.")

option = st.selectbox(
    'Which dimension would you like to analyze?',
    ('LoyaltyBand', 'TotalSpendBand', 'CustStatus', 'Last12mSpendBand', 'AvgTimepOrderBand', 'ATVBand'),
    key='dimension_select'
)

# --- Data Cuts Table ---
st.subheader(f"Data Cut by: {option}")
data_cut = agg_data.groupby(option).agg(
    Counts=('CustomerID', 'count'),
    TotalSpend=('TotalSpend', 'sum'),
    TotalOrders=('TotalOrders', 'sum'),
    Spend12m=('Spend12m', 'sum')
).reset_index()

data_cut['Avg Spend'] = data_cut['TotalSpend'] / data_cut['Counts']
data_cut['Avg Orders'] = data_cut['TotalOrders'] / data_cut['Counts']
data_cut['Avg Spend Last12m'] = data_cut['Spend12m'] / data_cut['Counts']
data_cut['ATV'] = data_cut['TotalSpend'] / data_cut['TotalOrders']

st.dataframe(data_cut.style.format({
    'Counts': '{:,.0f}',
    'TotalSpend': '£{:,.2f}',
    'TotalOrders': '{:,.0f}',
    'Spend12m': '£{:,.2f}',
    'Avg Spend': '£{:,.2f}',
    'Avg Orders': '{:,.2f}',
    'Avg Spend Last12m': '£{:,.2f}',
    'ATV': '£{:,.2f}'
}))

# --- Charts ---
st.subheader(f"Charts by: {option}")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.pyplot(create_bar_chart(data_cut, option, 'Counts', f'Customer Count by {option}'))
    st.pyplot(create_bar_chart(data_cut, option, 'TotalOrders', f'Total Orders by {option}'))
    st.pyplot(create_bar_chart(data_cut, option, 'Avg Spend', f'Average Spend by {option}'))
    st.pyplot(create_bar_chart(data_cut, option, 'ATV', f'Average Transaction Value by {option}'))

with chart_col2:
    st.pyplot(create_bar_chart(data_cut, option, 'TotalSpend', f'Total Spend by {option}'))
    st.pyplot(create_bar_chart(data_cut, option, 'Avg Orders', f'Average Orders by {option}'))
    st.pyplot(create_bar_chart(data_cut, option, 'Spend12m', f'Total Spend (Last 12m) by {option}'))
    st.pyplot(create_bar_chart(data_cut, option, 'Avg Spend Last12m', f'Avg Spend (Last 12m) by {option}'))

# --- Final Data and Download ---
st.header("Aggregated Customer Data")
with st.expander("View the final aggregated dataset"):
    st.dataframe(agg_data)

csv_data = agg_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Aggregated Data as CSV",
    data=csv_data,
    file_name='aggregated_customer_data.csv',
    mime='text/csv'
)

# --- Contact Form ---
st.header("Let's Connect!")
st.write("If you have any questions or would like to discuss this further, please get in touch.")

with st.form(key='contact_form'):
    name = st.text_input(label='Enter your name', placeholder='Your Name')
    email = st.text_input(label='Enter your email address', placeholder='you@example.com')
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if not name or not email:
            st.warning('Please enter both your name and email address.')
        else:
            try:
                # --- IMPORTANT ---
                # This requires you to set up secrets in Streamlit Cloud:
                # [smtp]
                # user = "your_gmail_address@gmail.com"
                # password = "your_app_password" 
                
                from_addr = email
                to_addr = st.secrets["smtp"]["user"]
                username = st.secrets["smtp"]["user"]
                password = st.secrets["smtp"]["password"]
                
                msg_body = (
                    f"Subject: New Enquiry from Customer Analysis App\n\n"
                    f"You have a new enquiry from: {name}\n"
                    f"Email: {email}"
                )
                
                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    server.starttls()
                    server.login(username, password)
                    server.sendmail(from_addr, to_addr, msg_body)
                
                st.success('Thank you! Your message has been sent. I will be in touch shortly.')
            except Exception as e:
                st.error(f"Sorry, something went wrong. Could not send the email. Error: {e}")
