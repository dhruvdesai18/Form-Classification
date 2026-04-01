import json
import re
import streamlit as st
import dotenv
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

#assuming calculators.py is in the same directory
from calculators import calculate_emi, calculate_foir, calculate_ltv, evaluate_eligibility
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

#load llms and embeddings
#changed  : use Google Generative AI for LLM and embeddings(using gemini 1.5 flash)
#model = "gemini-1.5-flash-latest" specifies the desired model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", api_key=google_api_key, temperature = 0.1)

#changed: initialize GoogleGenerativeAIEmbeddings with the API key
#model = "embedding-001" is typically used for Google Generative AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)

#Load FAISS vector store
#the embeddings object passed here must match the one used to create the vector store
#since we are now using Google Generative AI embeddings, this is correct
db = FAISS.load_local("vector_store/", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})


#streamlit setup
st.title("🏠 HOME LOAN CHATBOT - INTENT DRIVER")
if "eligibility_data" not in st.session_state:
    st.session_state.eligibility_data = {}

user_input = st.text_input("Ask me about home loans:", key = "user_input")


if st.button("Submit"):
    if user_input.strip():
        #initialze session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        #append user message to session history
        st.session_state.messages.append({"role": "user", "content": user_input})

        #step 1: detect intent and extract data
        detection_promt = f"""
        You are a Home Loan Assistant.
        Given the QUESTION below, respond in JSON format
        {{
            "intent": "CheckEligibility" or "GeneralQuery",
            "data": {{
                "income": number or None,
                "loan_amount": number or None,
                "tenure": number or None,
                "property_value": number or None
                }},
            "missing": [list of any missing fields]
        }}
        QUESTION: "{user_input}"
        Respond in JSON only
        """

        raw_response = llm.invoke(detection_promt).content
        print(raw_response)

        # Clean possible markdown JSON formatting
        cleaned = re.sub(r"```json|```", "", raw_response).strip()

        try:
            parsed = json.loads(cleaned)
            print(parsed)
        except json.JSONDecodeError as e:
            error_msg = f"Couldn't parse the response: {e}. Raw response: {cleaned}"
            st.session_state.messages.append({"role": "bot", "content": error_msg})
            st.stop()

        intent = parsed.get("intent", "")
        data = parsed.get("data", {})
        missing = parsed.get("missing", [])



        if intent == "CheckEligibility":
            for key, val in data.items():
                if val is not None:
                    st.session_state.eligibility_data[key] = val

            still_missing = [f for f in ["income", "loan_amount", "tenure", "property_value"]
                                if f not in st.session_state.eligibility_data or not st.session_state.eligibility_data[f]]

            if still_missing:
                response = f"To check eligibility, I need the following information: {', '.join(still_missing)}."
            else:
                ed = st.session_state.eligibility_data
                try:
                    income = float(ed.get("income"))
                    loan_amount = float(ed.get("loan_amount"))
                    property_value = float(ed.get("property_value"))
                    tenure = int(ed.get("tenure"))
                except (ValueError, TypeError):
                    response = "Error : Could not convert input data to numbers."
                
                
                # else:
                emi = calculate_emi(loan_amount, tenure)
                foir = calculate_foir(emi, income)
                ltv = calculate_ltv(loan_amount, property_value)

                context_docs = retriever.get_relevant_documents(user_input)
                context_text = "\n\n".join(doc.page_content for doc in context_docs)
                print(context_text)

                summary_prompt = f"""
                You are a smart home loan assistant.
                Given the policy CONTEXT and the customer calculated values below, create a short eligibility assessment:
                -Show a markdown table of EMI, FOIR %, LTV %
                -Give 3 short bullet remarks referencing CONTEXT.
                -No long paragraphs.

                CONTEXT: {context_text}

                CALCULATED:
                EMI: {emi:,.2f}
                FOIR: {foir:,.2f}%
                LTV: {ltv:,.2f}%
                """

                response = llm.invoke(summary_prompt).content
                print(response)

        else: 
                        #GeneralQuery
                docs = retriever.get_relevant_documents(user_input)
                context_text = "\n\n".join(doc.page_content for doc in docs)

                qa_prompt = f"""
                        You are a Home Loan Expert. Answer SHORTLY. Highlight the important parts of the explaination.
                        Use this CONTEXT only. If user asks anything that is not related to home loans, ask them to stick to home loan topic only.

                        CONTEXT: {context_text}

                        QUESTION: {user_input}
                        """

                response = llm.invoke(qa_prompt).content

                    #save bot response in session history
        st.session_state.messages.append({"role": "bot", "content": response})

            #at the top or before rendering text_input
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

            #after generating bot response
        if st.button("Clear Input"):
            st.session_state.user_input = "" #only safe to reset on a button trigger

            #display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Bot:** {msg['content']}")

with st.sidebar:
    st.write("Manually fill eligibility data if needed:")
    ed = st.session_state.eligibility_data
    #use key to persist values across reruns consistently
    ed["income"] = st.number_input("Monthly Income (₹)", value=ed.get("income", 0.0), key="income")
    ed["loan_amount"] = st.number_input("Desired Loan Amount", value=ed.get("loan_amount", 0.0), key="loan_amount")
    ed["tenure"] = st.number_input("Tenure (years)", value=ed.get("tenure", 0), key="tenure")
    ed["property_value"] = st.number_input("Property Value", value=ed.get("property_value", 0.0), key="property_value")
                        




## python -m streamlit run app.py