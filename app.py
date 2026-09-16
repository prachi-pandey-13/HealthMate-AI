import os
import streamlit as st
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from rag import (
    load_knowledge_base,
    create_chunks,
    create_embeddings,
    create_index,
    retrieve_information
)


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HealthMate AI",
    page_icon="💙",
    layout="centered"
)


# --------------------------------------------------
# APP TITLE
# --------------------------------------------------

st.title("💙 HealthMate AI")

st.subheader(
    "AI-Powered Preventive Health & Wellness Assistant"
)

st.info(
    "🌱 Ask general questions about sleep, nutrition, "
    "physical activity, mental well-being, and preventive health."
)

st.write(
    "HealthMate AI provides general educational guidance about "
    "health, wellness, nutrition, sleep, physical activity, "
    "mental well-being, and preventive health."
)

st.divider()


# --------------------------------------------------
# INITIALIZE RAG
# --------------------------------------------------

@st.cache_resource
def initialize_rag():

    text = load_knowledge_base()

    chunks = create_chunks(text)

    embedding_model, embeddings = create_embeddings(chunks)

    index = create_index(embeddings)

    return chunks, embedding_model, index


# Load the RAG system
try:

    chunks, embedding_model, index = initialize_rag()

except Exception as e:

    st.error("Unable to initialize the HealthMate AI knowledge base.")

    st.exception(e)

    st.stop()
# --------------------------------------------------
# RESPONSIBLE AI SAFETY CHECK
# --------------------------------------------------

def safety_check(question):

    question_lower = question.lower()

    emergency_keywords = [
        "can't breathe",
        "cannot breathe",
        "difficulty breathing",
        "chest pain",
        "heart attack",
        "stroke",
        "unconscious",
        "severe bleeding",
        "heavy bleeding",
        "suicide",
        "kill myself",
        "self harm",
        "overdose",
        "poisoning"
    ]

    for keyword in emergency_keywords:

        if keyword in question_lower:

            return (
                "⚠️ **This may require urgent medical attention.**\n\n"
                "HealthMate AI provides general educational information "
                "and cannot assess emergencies. Please contact your "
                "local emergency service or seek immediate medical care."
            )

    return None

# --------------------------------------------------
# IBM GRANITE AI FUNCTION
# --------------------------------------------------

def get_ai_response(question, category, context):

    credentials = Credentials(
        url=WATSONX_URL,
        api_key=API_KEY
    )

    model = ModelInference(
        model_id="ibm/granite-4-h-small",
        credentials=credentials,
        project_id=PROJECT_ID
    )

    prompt = f"""
You are HealthMate AI, a responsible preventive health
and wellness assistant.

Your purpose is to provide simple, general, educational
information that can help users make better-informed
wellness decisions.

Selected health category:
{category}

User question:
{question}


Trusted Knowledge Base:
{context}


IMPORTANT INSTRUCTIONS:

1. Use the supplied knowledge base as the primary source
   of factual information.

2. Do not invent health facts that are not supported by
   the supplied knowledge.

3. If the knowledge base does not contain enough information
   to answer the question, clearly say that the available
   knowledge base does not contain enough information.

4. Give general educational wellness information only.

5. Do NOT diagnose diseases or medical conditions.

6. Do NOT prescribe medicines.

7. Do NOT recommend starting, stopping, or changing medication.

8. Do NOT claim to replace a doctor or healthcare professional.

9. Do not make assumptions about the user's personal health.

10. Avoid requesting unnecessary sensitive or personal
    health information.

11. If the user describes a potentially serious or emergency
    situation, clearly advise them to seek appropriate
    professional medical care or emergency services.

12. Use simple and understandable language.

13. Do not present uncertain information as a fact.

14. Encourage professional medical advice when symptoms are
    persistent, severe, worsening, or concerning.


Response format:

### 💡 Answer

Give a short and clear answer to the user's question.


### 🌱 Practical Tips

Give 3-5 practical and general suggestions when appropriate.


### ⚠️ Important

Mention important safety considerations and explain when
the user should consider speaking with a healthcare
professional.


Remember:

HealthMate AI provides educational wellness information.
It does not diagnose diseases, prescribe medicines,
or replace professional medical advice.

Answer the user's question now.
"""

    response = model.generate_text(
        prompt=prompt,
        params={
            "max_new_tokens": 400,
            "temperature": 0.3
        }
    )

    return response


# --------------------------------------------------
# HEALTH CATEGORIES
# --------------------------------------------------

st.write("### 🌱 Explore Health & Wellness")

category = st.selectbox(
    "Choose a category",
    [
        "General Wellness",
        "Sleep",
        "Nutrition",
        "Physical Activity",
        "Mental Well-being",
        "Preventive Health"
    ]
)

st.write(f"**Selected category:** {category}")


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------
st.write("### 💬 Ask HealthMate AI")

st.caption(
    "Try an example or enter your own wellness question."
)

example_questions = {
    "Sleep": "How can I improve my sleep routine?",
    "Nutrition": "What are some healthy eating habits?",
    "Physical Activity": "How much physical activity should an adult get?",
    "Mental Well-being": "What habits can support mental well-being?",
    "Preventive Health": "What are some general preventive health habits?"
}

question = st.text_area(
    "Your question",
    placeholder=example_questions.get(
        category,
        "Example: How can I improve my sleep routine?"
    )
)

# --------------------------------------------------
# ASK HEALTHMATE AI BUTTON
# --------------------------------------------------

if st.button(
    "🤖 Ask HealthMate AI",
    use_container_width=True
):

    # Check question
    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    # Check Watsonx configuration
    elif not API_KEY or not PROJECT_ID or not WATSONX_URL:

        st.error(
            "Watsonx configuration is missing. "
            "Please check your .env file."
        )

    else:

        # --------------------------------------------------
        # RESPONSIBLE AI SAFETY CHECK
        # --------------------------------------------------

        safety_message = safety_check(question)

        if safety_message:

            st.warning(safety_message)

        else:

            with st.spinner(
                "🤖 HealthMate AI is searching its knowledge base..."
            ):

                try:

                    # --------------------------------------------------
                    # RAG RETRIEVAL
                    # --------------------------------------------------

                    relevant_chunks = retrieve_information(
                        question,
                        chunks,
                        embedding_model,
                        index,
                        top_k=3
                    )

                    context = "\n\n".join(
                        relevant_chunks
                    )

                    # --------------------------------------------------
                    # IBM GRANITE RESPONSE
                    # --------------------------------------------------

                    with st.spinner(
                        "🤖 HealthMate AI is generating your answer..."
                    ):

                        answer = get_ai_response(
                            question,
                            category,
                            context
                        )

                    # --------------------------------------------------
                    # DISPLAY RESPONSE
                    # --------------------------------------------------

                    st.success(
                        "HealthMate AI Response"
                    )

                    st.markdown(answer)

                    # --------------------------------------------------
                    # SHOW KNOWLEDGE USED
                    # --------------------------------------------------

                    with st.expander(
                        "📚 Knowledge used by HealthMate AI"
                    ):

                        st.write(
                            "These knowledge-base passages were "
                            "retrieved and provided to the AI."
                        )

                        for i, chunk in enumerate(
                            relevant_chunks,
                            1
                        ):

                            st.write(
                                f"**Knowledge Passage {i}**"
                            )

                            st.write(chunk)

                            st.divider()

                except Exception as e:

                    st.error(
                        "Unable to generate a response from "
                        "IBM watsonx.ai."
                    )

                    st.exception(e)



# --------------------------------------------------
# ABOUT THE PROJECT
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ About HealthMate AI"):

    st.write(
        "HealthMate AI is an AI-powered preventive health and "
        "wellness assistant developed as part of the IBM SkillsBuild "
        "AI for Sustainability Virtual Internship."
    )

    st.write(
        "The project aligns with **UN Sustainable Development "
        "Goal 3 — Good Health and Well-being**."
    )

    st.write(
        "The system combines a health knowledge base, RAG "
        "(Retrieval-Augmented Generation), and IBM Granite "
        "through IBM watsonx.ai."
    )
# --------------------------------------------------
# RESPONSIBLE AI
# --------------------------------------------------

st.divider()

st.subheader("🛡️ Responsible AI")

st.write(
    "HealthMate AI is designed with responsible AI principles "
    "to provide safer and more transparent wellness guidance."
)

with st.expander("🔒 Privacy"):

    st.write(
        "• Avoid entering unnecessary personal or sensitive health information."
    )

    st.write(
        "• HealthMate AI is designed to provide general wellness "
        "information rather than collect detailed medical histories."
    )


with st.expander("🤖 AI Transparency"):

    st.write(
        "• Responses are generated using IBM Granite through "
        "IBM watsonx.ai."
    )

    st.write(
        "• HealthMate AI uses a knowledge base and RAG to provide "
        "context for AI-generated responses."
    )

    st.write(
        "• AI-generated information may not always be complete or accurate."
    )


with st.expander("⚠️ Limitations"):

    st.write(
        "• HealthMate AI does not diagnose diseases."
    )

    st.write(
        "• It does not prescribe medicines or recommend medication changes."
    )

    st.write(
        "• It does not replace a qualified healthcare professional."
    )


with st.expander("👥 Fairness & Ethics"):

    st.write(
        "• HealthMate AI avoids making assumptions about a user's "
        "personal health."
    )

    st.write(
        "• Responses are intended to provide general guidance "
        "without discrimination or judgment."
    )


with st.expander("🧑‍⚕️ Professional Care"):

    st.write(
        "• Users should seek professional medical advice when "
        "symptoms are persistent, severe, worsening, or concerning."
    )

    st.write(
        "• Potential emergency situations are directed toward "
        "urgent professional medical care."
    )


st.caption(
    "⚠️ HealthMate AI provides general educational wellness "
    "information. It does not diagnose diseases, prescribe "
    "medicines, or replace professional medical advice."
)