import numpy as np 
import pandas as pd
import streamlit as st 
import openai
import os
from function import model,recommend_products,open_ai_key,configure_streamlit_page,add_logo,sidebar_fix_width
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

from prompt_per_msg import customer_style, template_string, template_string_new, best_selling_product, welcome_offer,instruction_existing
configure_streamlit_page()
sidebar_fix_width()
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
st.sidebar.image("./Data/cropped-Sigmoid_logo_3x.png", use_column_width=True)
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
st.markdown('<style>div.row-widget.stButton > button:first-child {background-color: blue; color: white;}</style>', unsafe_allow_html=True)

openai_api_key=open_ai_key()
if openai_api_key:
    tab1,tab2 = st.tabs(["CodeAI","Personalized Welcome Message"])
    with tab1:



    # Initialize an empty dictionary to store column descriptions
        column_descriptions = {}

        def main():
            st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">CodeAI:</p>', unsafe_allow_html=True)
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">👨‍💻 How to Use:</p>', unsafe_allow_html=True)
            st.markdown("""
            - 📂 Upload a CSV or Excel file containing your dataset.
            - 📝 Provide descriptions for each column of the dataset in the 'Column Descriptions' section.
            - 📂 Optionally, upload a CSV or Excel file containing column descriptions.
            - ❓ Ask questions about the dataset in the 'Ask a question about the dataset' section.
            - 🔍 Click the 'Get Answer' button to generate an answer based on your question.
            """)

            # Display limitations with emojis
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
            st.markdown("""
            - The quality of AI responses depends on the quality and relevance of your questions.
            - Ensure that you have a good understanding of the dataset columns to ask relevant questions.
            """)   
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)

            # Upload the dataset file
            uploaded_file = st.file_uploader("Upload a CSV or Excel file (Dataset)", type=["csv", "xlsx"])
            st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Head of the Dataset:</p>', unsafe_allow_html=True)
            
            df_user = pd.DataFrame()
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_user = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                        df_user = pd.read_excel(uploaded_file)
                    
                    # Display the first few rows of the dataset
                    st.write(df_user.head())
                    

                    st.info("Please add column descriptions of your dataset")
                    for col in df_user.columns:
                        col_description = st.text_input(f"Description for column '{col}':")
                        if col_description:
                            column_descriptions[col] = col_description
                        
                    if st.button("Submit Descriptions"):
                        st.success("Descriptions submitted successfully!")
                except Exception as e:
                    st.error(f"An error occurred while reading the dataset file: {e}")
                    return

            # Optionally, upload column descriptions file
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            uploaded_desc_file = st.file_uploader("Upload a CSV or Excel file (Column Descriptions)", type=["csv", "xlsx"])
            if uploaded_desc_file is not None:
                try:
                    if uploaded_desc_file.name.endswith('.csv'):
                        desc_df = pd.read_csv(uploaded_desc_file)
                    elif uploaded_desc_file.name.endswith(('.xls', '.xlsx')):
                        desc_df = pd.read_excel(uploaded_desc_file)

                    # Assuming the column descriptions are in two columns: 'Column Name' and 'Description'
                    for index, row in desc_df.iterrows():
                        col_name = row['Column Name']
                        col_description = row['Description']
                        if col_name and col_description:
                            column_descriptions[col_name] = col_description

                    st.success("Column descriptions loaded successfully!")
                except Exception as e:
                    st.error(f"An error occurred while reading the column descriptions file: {e}")

        


            
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Ask a question about the dataset:</p>', unsafe_allow_html=True)
            user_question = st.text_input(" ")
            user_question+= "call the function below in the same script with 'df_user"

            code_string ="""Generate the python code based on the user question\
                that is delimated by triple backticks\
                    based on the instruction that is {instruction}.\
                        user question: ```{user_question}```\
                            """
            code_templete= ChatPromptTemplate.from_template(code_string)

            instruction =f"""1. You are functioning as an AI data analyst.
            2. Task: Respond to questions based on the provided dataset by giving code
            3. Dataset columns enclosed in square brackets {df_user.columns.tolist()}.
            4. Columns Description in dict format - {column_descriptions}.
            5. Provide code based on the user's question.
            6. Do not create any dummy or test dataset; call the function with DataFrame name: 'df_user'.
            7. Print result using 'st.write' for text or 'st.pyplot' for plots use plotly with white background for the plot
            8. Return the output in function form only.
            9. Provide all the code together.
            10. Only return the code; no explanations or extra text.
            11. Include code to suppress warnings.
            12. Do not include [assistant].
            13. Do not read any dataset; call the function with df_user.
            14. Return final output with st.write or st.pyplot.
            15. Only give the executable code.
            16. Code must start with 'def' and end with the function call.
            17. Do not enclose the code in triple backticks.
            18. Only give the executable line; no non-executable characters.
            19. Call the function below the response in the same script.
            20. Always call the function in the same script with 'df_user'"""

            user_message = code_templete.format_messages(instruction=instruction,user_question=user_question)
                    
            chat2 = ChatOpenAI(temperature=0.0, model=model,openai_api_key=openai_api_key)
            if st.button("Get Answer"):
                if user_question:
                    user_message = code_templete.format_messages(instruction=instruction,user_question=user_question)
                    code = chat2(user_message)
                    st.code(code.content)
                    exec(code.content)
                else:
                    st.warning("Not a valid question. Please enter a question to analyze.")
            
            # st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Code Execution Dashboard:</p>', unsafe_allow_html=True)
        
            # st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            # code_input = st.text_area("Enter your code here", height=200)
            # st.warning(("⚠️ If there is any non-executable line in generated code; please remove it"))
            
            # if st.button("Execute code"): 
            #     try:
            #         # Use exec() to execute the code
            #         exec(code_input)
            #     except Exception as e:
            #         st.error(f"An error occurred: {e}")

        # Check if the script is run as the main program
        if __name__ == "__main__":
            main()

    with tab2:
        llm_model = "gpt-3.5-turbo-0301"
        chat = ChatOpenAI(temperature=1, model=llm_model, openai_api_key=openai_api_key)
        df_final = pd.read_csv("./Data/df_final_with_name2.csv")
        existing_user = df_final["user_id"].unique()
        # Function to generate personalized messages for new users
        def personlized_message_new_user(template, style, welcome_offer, best_selling_pro, user_data,instruction_existing):
            prompt_template = ChatPromptTemplate.from_template(template)
            customer_messages = prompt_template.format_messages(
                style=style,
                welcome_offer=welcome_offer,
                best_selling_product=best_selling_pro,
                user_data=user_data,
                instruction_existing=instruction_existing

            )
            customer_response = chat(customer_messages)
            return customer_response.content

        # Function to generate personalized messages for existing users
        def personlized_message_existing_user(template, style, Existing_user_data, Rec_product, Offers_and_promotion,instruction_existing):
            prompt_template = ChatPromptTemplate.from_template(template)
            customer_messages = prompt_template.format_messages(
                style=style,
                Existing_user_data=Existing_user_data,
                Offers_and_promotion=Offers_and_promotion,
                Rec_product=Rec_product,
                instruction_existing=instruction_existing
            )
            customer_response = chat(customer_messages)
            return customer_response.content

        # Define custom colors
        primary_color = "#3498db"  # Blue
        secondary_color = "#2ecc71"  # Green
        background_color = "#f0f3f6"  # Light Gray
        text_color = "#333333"  # Dark Gray

        # Apply custom styles
        st.markdown(
            f"""
            <style>
            .reportview-container {{
                background: {background_color};
                color: {text_color};
            }}
            .sidebar .sidebar-content {{
                background: {primary_color};
                color: white;
            }}
            .widget-label {{
                color: {text_color};
            }}
            .stButton.button-primary {{
                background: {secondary_color};
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Recommendation part
        st.markdown("### Personalized Welcome Message")
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        with st.form("login_form"):
            user_id = st.text_input("User ID")
            user_name = st.text_input("Your Name")
            submitted = st.form_submit_button("Login")

        if submitted:
            with st.spinner('Generating...'):
                if user_id in existing_user:
                    if user_id:
                        recommended_products = recommend_products(user_id, df_final)

                        if recommended_products:
                            offer = df_final[df_final["product_id"].isin(recommended_products)]["offers"].unique().tolist()
                            offers = [offer[0], offer[-1]]
                            item_cart=["Tanqueray Sterling Vodka", "7 Crown Appl", "Ursus Punch Vodka"]
                            Existing_user_data = {"Name": user_name, "Existing Items in the cart":item_cart }
                            Rec_product = recommended_products
                            Offers_and_promotion = offers
                            existing_user = personlized_message_existing_user(template_string, customer_style, Existing_user_data, Rec_product, Offers_and_promotion,instruction_existing)
                            with st.chat_message("user"):
                                st.write(existing_user)
                        else:
                            st.warning("Please enter the right details.")
                else:
                    new_message = personlized_message_new_user(template_string_new, customer_style, welcome_offer, best_selling_product, user_name,instruction_existing)
                    with st.chat_message("user"):
                        st.write(new_message)
else:
    st.warning("Please Enter Your API key!")
