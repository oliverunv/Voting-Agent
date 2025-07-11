import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import contextlib
import io
import re
from openai import OpenAI

# Load dataset
df = pd.read_csv("data/sc_voting.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Year"] = df["Year"].round().astype("Int64")

# Set up page
st.set_page_config(page_title="UNSC Voting Explorer", layout="wide")
st.title("🗳️ UN Security Council Voting Explorer")

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Column descriptions to inform GPT prompt
column_descriptions = """
The DataFrame `df` contains the following columns:

- ID: An identifier for each draft resolution put to a vote. Since the data is unpivoted by Member State, each ID appears once per vote cast. Use `ID.nunique()` to count draft resolutions, and `len(df)` or `.shape[0]` to count total individual votes.
- Year: stored as an integer (e.g., 1994). Use this column to filter by year. When plotting, treat it as a categorical/time label — do not format it with commas (e.g., write "2002", not "2,002").
- Date: The date on which the Security Council held the vote (in DD/MM/YYYY format).
- Resolution: The number assigned to the resolution, if successfully adopted (e.g., "924 (1994)").
- Draft: The UN document symbol of the draft resolution (e.g., "S/1994/646").
- Outcome results: The result of the vote on the draft resolution (contains the followin categories: "Adopted unanimously", "Adopted by consensus", "Adopted by acclamation", "Adopted by majority", "Adopted without a vote", "Not adopted - failed to receive required number of votes", "Not adopted - no vote", "Not adopted - veto").
- Agenda item: The agenda item of the Security Council under which the vote took place (e.g., "The situation in the Republic of Yemen").
- Agenda category: Indicates whether the agenda item is country-/region-specific, or thematic.
- Agenda region: The geographical region related to the agenda item (e.g., "Middle East", "Africa", "Asia").
- Vote: The vote cast by the Member State on the draft resolution. Values include "Yes", "No", or "Abstain".
- Member State: The name of the country casting the vote (e.g., "Argentina", "China", "United States").
"""

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about voting on draft resolutions in the Security Council since 1992."}
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

    # Build system prompt and history-based message list
    system_prompt = f"""
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

Question: How many No votes did France cast since 1992?
Code:
st.write("France has voted No", len(df[(df["Member State"] == "France") & (df["Vote"] == "No") & (df["Year"] >= 1992)]), "times since 1992.")

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

    # Add system + prior messages
    chat_history = [{"role": "system", "content": system_prompt}]
    chat_history += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.spinner("Generating response..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=chat_history,
                temperature=0.3
            )
            code = response.choices[0].message.content.strip()

            # Extract clean code if it’s inside a code block
            match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()

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

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 0.5rem;
        font-size: 0.8rem;
        color: gray;
        z-index: 999;
        background-color: transparent;
        backdrop-filter: blur(2px);
    }
    </style>
    <div class="footer">
        Built by Oliver Unverdorben · Powered by OpenAI · © 2025
    </div>
    """,
    unsafe_allow_html=True
)