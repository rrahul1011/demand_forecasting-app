# Import necessary libraries
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from function import add_logo, configure_streamlit_page, sidebar_fix_width, head_df, model, open_ai_key

# Set up Streamlit page configuration
configure_streamlit_page()
add_logo()
sidebar_fix_width()
model = model

# Centered Page Heading with red color and improved styling
st.markdown('<h4 style="color: red; text-align: center;">CODE AI: Transforming Data into Insights</h4>', unsafe_allow_html=True)

# Display About CODE AI, Usage Instructions, and Additional Information
col1, col2, col3 = st.columns(3)
with col1:
    with st.expander("About CODE AI ❓", expanded=False):
        st.markdown(
            "Welcome to CODE AI, where powerful analytics meets simplicity. "
            "Empowering both developers and non-coders, our user-friendly interface turns raw data into actionable insights. "
            "Explore the potential of your data effortlessly. Let CODE AI redefine how you interact with your information. "
            "Elevate your understanding, empower your decisions – with CODE AI, analytics made simple."
        )
with col2:
    with st.expander(" ☞ Usage Instructions", expanded=False):
        st.markdown("""
            - 📂 Upload a CSV or Excel file containing your dataset.
            - 📝 Provide descriptions for each column of the dataset in the 'Column Descriptions' section.
            - 📂 Optionally, upload a CSV or Excel file containing column descriptions.
            - ❓ Ask questions about the dataset in the 'Ask a question about the dataset' section.
            - 🔍 Click the 'Get Answer' button to generate an answer based on your question.
        """)

with col3:
    with st.expander(" 📜 Additional Information", expanded=False):
        st.markdown("""
        - The quality of AI responses depends on the quality and relevance of your questions.
        - Ensure that you have a good understanding of the dataset columns to ask relevant questions.
        """)

# Horizontal line separator
st.markdown("<hr style='border: .5px solid red; width: 100%;'>", unsafe_allow_html=True)

# File upload section for the dataset
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload a CSV or Excel file (Dataset)", type=["csv", "xlsx"])
    column_descriptions = {}
    df_user = pd.DataFrame()

    # Read the dataset and display a preview
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_user = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df_user = pd.read_excel(uploaded_file)
            with st.expander("Data Set Preview", expanded=False):
                st.dataframe(df_user)
        except Exception as e:
            st.error(f"An error occurred while reading the dataset file: {e}")

with col2:
    # File upload section for column descriptions
    uploaded_desc_file = st.file_uploader("Upload a CSV or Excel file (Column Descriptions)", type=["csv", "xlsx"])
    if uploaded_desc_file is not None:
        try:
            if uploaded_desc_file.name.endswith('.csv'):
                desc_df = pd.read_csv(uploaded_desc_file)
            elif uploaded_desc_file.name.endswith(('.xls', '.xlsx')):
                desc_df = pd.read_excel(uploaded_desc_file)
            for index, row in desc_df.iterrows():
                col_name = row['Column Name']
                col_description = row['Description']
                if col_name and col_description:
                    column_descriptions[col_name] = col_description

        except Exception as e:
            st.error(f"An error occurred while reading the column descriptions file: {e}")
        col_des_df = pd.DataFrame.from_dict(column_descriptions, orient="index")
        col_des_df = col_des_df.rename(columns={0: "Description"})
        with st.expander("Column Description Preview", expanded=False):
            st.markdown('<p style="color:red; font-size:15px; font-weight:bold;">Column Description:</p>', unsafe_allow_html=True)
            st.dataframe(col_des_df)

# Checkbox to enable dataset analysis summary
if df_user is not None:
    dataframe_summary = st.checkbox("Need quick analysis of dataset? ")

# Display dataset analysis summary if selected
if dataframe_summary:
    st.markdown('#### Analysis Report')
    head_df(df_user)

# Get OpenAI API key
openai_api_key = open_ai_key()

# Horizontal line separator
st.markdown("<hr style='border: .5px solid red; width: 100%;'>", unsafe_allow_html=True)

# User input section to ask questions about the dataset
st.markdown('<p style="color:red; font-size:15px; font-weight:bold;">Ask a question about the dataset:</p>', unsafe_allow_html=True)
user_question = st.text_input("Ask a Question")

# Validate OpenAI API key
if openai_api_key:
    # If user question is provided, modify it for code execution
    if user_question is not None:
        user_question += "call the function below in the same script with 'df_user"

    # Set up code generation template
    code_string = """Generate the python code based on the user question\
        that is delimited by triple backticks\
            based on the instruction that is {instruction}.\
                user question: ```{user_question}```\
                    """
    code_template = ChatPromptTemplate.from_template(code_string)

    # Provide instructions for code generation
    instruction = f"""1. You are functioning as an AI data analyst.
    2. Task: Respond to questions based on the provided dataset by giving code
    3. Dataset columns enclosed in square brackets {df_user.columns.tolist()}.
    4. Columns Description in dict format - {column_descriptions}.
    5. Provide code based on the user's question.
    6. Do not create any dummy or test dataset; call the function with DataFrame name: 'df_user'.
    7. Print result using 'st.write' for text or 'st.pyplot' for plots use plotly with a white background for the plot
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
    19. Call the function below the response in the same script with 'df_user'
    20. Always call the function in the same script with 'df_user'"""

    user_message = code_template.format_messages(instruction=instruction, user_question=user_question)

    # Initialize ChatOpenAI instance
    chat2 = ChatOpenAI(temperature=0.0, model=model, openai_api_key=openai_api_key)

    # Display user input for generating code
    if user_question:
        user_message = code_template.format_messages(instruction=instruction, user_question=user_question)
        code = chat2(user_message)
        # Display generated code and execute on button click
        if st.button("Get Answer"):
            with st.chat_message("user"):
                st.code(code.content)
                exec(code.content)
    else:
        st.warning("Not a valid question. Please enter a question to analyze.")
else:
    st.warning("To assist you further, kindly provide your API key.")
