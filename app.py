import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import contextlib
import io
import re
from openai import OpenAI

# Load dataset
df = pd.read_csv("data/sc_voting.csv")
df["Year"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.year

# Set up page
st.set_page_config(page_title="UNSC Voting Explorer", layout="wide")
st.title("🗳️ UN Security Council Voting Explorer")
st.caption("💬 Ask anything about voting on draft resolutions in the Security Council since 1994.")

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Column descriptions to inform GPT prompt
column_descriptions = """
The DataFrame `df` contains the following columns:

- ID: A unique numeric identifier for each row in the dataset.
- Date: The date on which the Security Council held the vote (in DD/MM/YYYY format).
- Resolution: The number assigned to the resolution, if successfully adopted (e.g., "924 (1994)").
- Draft: The UN document symbol of the draft resolution (e.g., "S/1994/646").
- Outcome results: The result of the vote on the draft resolution (e.g., "Adopted unanimously", "Adopted by majority", "Not adopted").
- Agenda item: The agenda item of the Security Council under which the vote took place (e.g., "The situation in the Republic of Yemen").
- Agenda category: Indicates whether the agenda item is country-/region-specific, or thematic.
- Agenda region: The geographical region related to the agenda item (e.g., "Middle East", "Africa", "Asia").
- Vote: The vote cast by the Member State on the draft resolution. Values include "Yes", "No", or "Abstain".
- Member State: The name of the country casting the vote (e.g., "Argentina", "China", "United States").
- Year: Extracted year from the Date column, stored as an integer (e.g., 1994). Use this column to filter by year.
"""

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about Security Council voting patterns."}
    ]

# Function to explain code steps
def explain_code_steps(code, user_question):
    prompt = f"""
You are an assistant helping non-technical users understand Python code used in a data analysis app.

Please explain the logic of the code below in no more than 4 short, plain-English bullet points. Do not use programming jargon.

Focus only on what the code does — not how or why.

Question:
"{user_question}"

Code:
{code}
"""

    try:
        explanation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return explanation_response.choices[0].message.content.strip()
    except Exception as e:
        return "⚠️ Could not generate explanation."


# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "code" in msg and "explanation" in msg:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 💻 Code Used")
                st.code(msg["code"], language="python")
            with col2:
                st.markdown("#### 🧠 Steps Taken")
                st.markdown(msg["explanation"])

# Chat input
if user_input := st.chat_input("Your question..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build GPT prompt for code generation
    prompt = f"""
You are a Python data analyst working with a Pandas DataFrame called `df` in a Streamlit app.

Only return clean, executable code — no markdown, no triple backticks, no comments, no imports, and do not redefine the DataFrame.

Always use clear, human-readable sentences in st.write, e.g.:
"France has voted No 22 times since 1994."

The columns available in the DataFrame are:
{column_descriptions}

Use the following examples as **inspiration for tone and logic** — do NOT copy them literally or execute them:

\"\"\"
Question: How many draft resolutions were not adopted?
Code:
st.write("There were", df[df["Outcome results"] == "Not adopted"]["Draft"].nunique(), "draft resolutions that were not adopted.")

---

Question: How many No votes did France cast since 1994?
Code:
st.write("France has voted No", len(df[(df["Member State"] == "France") & (df["Vote"] == "No") & (df["Year"] >= 1994)]), "times since 1994.")

---

Question: Compare how often each P5 member voted No.
Code:
p5 = ["China", "France", "Russian Federation", "United Kingdom", "United States"]
df_p5_no = df[(df["Vote"] == "No") & (df["Member State"].isin(p5))]
no_counts = df_p5_no["Member State"].value_counts()
st.bar_chart(no_counts)

---

Question: Show a stacked bar chart by year of voting outcome (e.g. adopted, not adopted).
Code:
df_unique_res = df.drop_duplicates(subset="Draft")
outcome_by_year = df_unique_res.groupby(["Year", "Outcome results"]).size().unstack(fill_value=0)
st.bar_chart(outcome_by_year)
\"\"\"

The user’s question is:
\"{user_input}\"

Write only executable code using Streamlit to answer the question. Use the column names provided. Ensure the result is clearly written using natural sentences.
"""

    with st.spinner("Generating response..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            code = response.choices[0].message.content.strip()

            # Extract clean Python code block from assistant response
            match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                # Use the full response as fallback, if it's short and looks like code
                code = code.strip()

            # Execute code
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                try:
                    exec(code)
                    result = buffer.getvalue().strip()
                except Exception as e:
                    result = f"⚠️ Execution error:\n\n{e}"
                    code = f"# Error occurred during execution\n{code}"

            explanation = explain_code_steps(code, user_input)

            # Display response
            with st.chat_message("assistant"):
                st.markdown(result)

                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("💻 Show code used"):
                        st.code(code, language="python")
                with col2:
                    with st.expander("🧠 Show steps taken"):
                        st.markdown(explanation)
                    
            st.session_state.messages.append({
                "role": "assistant",
                "content": result,
                "code": code,
                "explanation": explanation
            })

        except Exception as e:
            st.chat_message("assistant").markdown(f"❌ Error generating response:\n\n{e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error generating response:\n\n{e}"
            })
