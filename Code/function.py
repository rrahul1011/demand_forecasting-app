import numpy as np 
import pandas as pd
import plotly.express as px 
import streamlit as st
import openai
import numpy as np
from PyPDF2 import PdfReader
from streamlit.logger import get_logger
@st.cache_data
def visualize_timeseries(df, level, country, category, brand, SKU):
    df_t = df[df["geo"] == country]
    if category:
        df_t = df_t[df_t["category"] == category]
    if brand:
        df_t = df_t[df_t["brand"] == brand]
    if SKU:
        df_t = df_t[df_t["SKU"] == SKU]

    if df_t.empty:
        st.warning("No data available for the selected combination.")
    else:
        group_cols = level + ["month","scenario"]
        aggregation = {"quantity": "sum"}
        df_t = df_t.groupby(group_cols, as_index=True).agg(aggregation).reset_index()
    df_t = df_t.dropna()
    chart_data = df_t.set_index("month")
    title = "_".join([country] + [val for val in [category, brand, SKU] if val])
    color_discrete_map = {
        "actual": "blue",
        "predicted": "red"
    }

    quantity_chart = px.line(
        chart_data,
        x=chart_data.index,
        y="quantity",
        title=title,
        color="scenario",
        color_discrete_map=color_discrete_map
    )
    quantity_chart.update_layout(height=500, xaxis_title="Month", yaxis_title="Quantity")
    st.plotly_chart(quantity_chart, use_container_width=True)
    st.markdown("---")

    return df_t


@st.cache_data
def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(

    model=model,

    messages=messages,

    temperature=0,

    )

    return response.choices[0].message["content"]

@st.cache_data
def yoy_growth(df):
    df["year"] = pd.to_datetime(df["month"]).dt.year
    df_yoy = df.groupby(["year"]).sum()["quantity"].reset_index()
    grouped_yoy = df_yoy[2:-1]
    grouped_yoy['yoy_growth'] = grouped_yoy['quantity'].pct_change(periods=1) * 100
    return grouped_yoy[["year","yoy_growth"]]


@st.cache_data
def calculate_trend_slope_dataframe(dataframe, polynomial_degree=1):
    if dataframe.empty:
        st.warning("No data available for the selected combination.")
    else:
        dataframe=dataframe.reset_index(drop=True)
        df_copy = dataframe.copy() 
        df_copy['cumulative_sum'] = df_copy['quantity'].cumsum()
        first_nonzero_index = df_copy['cumulative_sum'].ne(0).idxmax()
        df_copy = df_copy.iloc[first_nonzero_index:]
        df_copy.drop(columns=['cumulative_sum'], inplace=True)
        df_copy_his =df_copy[df_copy["scenario"]=="actual"]
        df_copy_for = df_copy[df_copy["scenario"]=="predicted"]
        time_points_his = [i for i in range(len(df_copy_his["quantity"]))]
        quantity_values_his = df_copy_his["quantity"]
        coefficients_his = np.polyfit(time_points_his, quantity_values_his, polynomial_degree)
        slope_his = coefficients_his[0]
        df_copy_his["slope_his"]=slope_his
        if slope_his>1:
            df_copy_his["trend"]="Increasing"
        elif slope_his <-1:
            df_copy_his["trend"]="Decreasing"
        else:
            df_copy_his["trend"]="No Trend"
        time_points_for = [i for i in range(len(df_copy_for["quantity"]))]
        quantity_values_for = df_copy_for["quantity"]
        coefficients_for = np.polyfit(time_points_for, quantity_values_for, polynomial_degree)
        slope_for = coefficients_for[0]
        df_copy_for["slope_for"]=slope_for
        if slope_for>1:
            df_copy_for["trend"]="Increasing"
        elif slope_for <-1:
            df_copy_for["trend"]="Decreasing"
        else:
            df_copy_for["trend"]="No Trend"
        df_final = pd.concat([df_copy_his,df_copy_for])

        return df_final
    
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


import os
@st.cache_data
def read_text_file(filename):
    data = []
    full_path = os.path.join(os.getcwd(), filename) 
    with open(full_path, "r") as inp:
        for line in inp:
            stripped_line = line.strip()
            if stripped_line:
                data.append(stripped_line)
    return data

model ="gpt-3.5-turbo-0301"

@st.cache_data(show_spinner=False)
def is_open_ai_key_valid(openai_api_key) -> bool:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar!")
        return False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            api_key=openai_api_key,
        )
    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        logger.error(f"{e.__class__.__name__}: {e}")
        return False
    return True

