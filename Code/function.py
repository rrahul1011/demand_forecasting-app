import pandas as pd
import streamlit as st
import io
from PIL import Image
import base64



model ="gpt-3.5-turbo-0301"


@st.cache_data  
def recommend_products(user_id,df,top_n=5):
    if user_id not in df["user_id"].unique().tolist():
        # If the user is new, recommend the top-rated products
        top_rated_products = list(pd.Series(df.sort_values(by='rating', ascending=False)['product_id'].head(top_n)))
        return top_rated_products
    else:
        pivot_table = df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
        user_ratings = pivot_table.loc[user_id]
        similarity = pivot_table.corrwith(user_ratings, axis=0)
        similar_products = similarity.sort_values(ascending=False).index[1:]  
        recommended_products = [product for product in similar_products if product not in df[df['user_id'] == user_id]['product_id']]
        return recommended_products[:top_n]
    
def open_ai_key():
    openai_api_key = st.sidebar.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    submit_button = st.sidebar.button("Submit")
    if submit_button:
        openai_api_key = openai_api_key
    return openai_api_key

### Page Configuration
def configure_streamlit_page():
    st.set_page_config(
        page_title="Sigmoid GenAI",
        page_icon="./Data/cropped-Sigmoid_logo_3x.png",
        layout="wide",
    )

    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left:0.2 rem;
                padding-right: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

## Logo in the side bar
def add_logo():
    # Path to your image
    image_path = "/Users/rahulkushwaha/gENAI/demand_forecasting-app/Data/cropped-Sigmoid_logo_3x.png"
    image_width = 160
    image_height = 80
    background_position_x =30
    background_position_y = 30

    # Read and resize the image
    file = open(image_path, "rb")
    contents = file.read()
    img_str = base64.b64encode(contents).decode("utf-8")
    buffer = io.BytesIO()
    file.close()
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize((image_width, image_height))
    resized_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

   
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{img_b64}');
                background-repeat: no-repeat;
                padding-top: 50px;
                background-position: {background_position_x}px {background_position_y}px;
            }}
            [data-testid="stSidebarNav"]::before {{
                content: "";
                border-bottom: 3px solid crimson;
                width: 100%;
                position: absolute;
                top: 130px; /* Adjust the distance below the image as needed */
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def sidebar_fix_width():
    st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 40px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
    )
