# Install dependencies (run once)
# pip install streamlit langchain langchain_openai langchain_community hdbcli ddgs tqdm

import streamlit as st
from hdbcli import dbapi
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever
from ddgs import DDGS
from openai import AzureOpenAI

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AI Blog Generator")

# Sidebar options
st.sidebar.header("Blog Options")
word_limit = st.sidebar.number_input("Word Limit", min_value=100, max_value=2000, value=1000)
tone = st.sidebar.selectbox("Tone", ["Formal", "Casual"])
industry = st.sidebar.text_input("Industry (optional)", value="") 


# File uploader
uploaded_file = st.file_uploader("Upload a reference document (TXT only)", type=["txt"])

query = st.text_input("Enter your blog query:")

generate_btn = st.button("Generate Blog")

# ---------------------------
# Database & GPT Setup
# ---------------------------
# HANA connection
# --- Database Connection (⚠️ update credentials before running) ---
connection = dbapi.connect(
            address=st.secrets["database"]["address"],
            port=st.secrets["database"]["port"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            encrypt=True,
            autocommit=True,
            sslValidateCertificate=False,
        )

        # --- Azure Setup ---
client = AzureOpenAI(
            azure_endpoint=st.secrets["azure"]["openai_endpoint"],
            api_key=st.secrets["azure"]["api_key"],
            api_version=st.secrets["azure"]["api_version"],
        )

embeddings = AzureOpenAIEmbeddings(
            azure_deployment=st.secrets["azure"]["embeddings_deployment"],
            openai_api_version=st.secrets["azure"]["embeddings_api_version"],
            api_key=st.secrets["azure"]["api_key"],
            azure_endpoint=st.secrets["azure"]["openai_endpoint"],
        )

db = HanaDB(
            embedding=embeddings,
            connection=connection,
            table_name="MARKETING_APP_CONTENT_GENERATION"
        )

# ---------------------------
# Blog generation logic
# ---------------------------
def generate_blog(query, word_limit, tone, uploaded_file=None):
    docs = []

    # 1. If HANA DB has documents
    hana_docs = db.similarity_search(query, k=20)
    docs.extend(hana_docs)

    # 2. If uploaded file exists, use as document
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        docs.append(Document(page_content=content))

    # 3. If no docs from HANA or uploaded file → use DuckDuckGo
    if not docs:
        st.info("No HANA docs found. Using DuckDuckGo to fetch online references...")
        ddgs = DDGS()
        results = ddgs.text(query, max_results=5)
        ddg_texts = [res["body"] for res in results]
        combined_text = "\n".join(ddg_texts)
        docs.append(Document(page_content=combined_text))

    # Combine all docs content
    all_docs = "\n".join([doc.page_content for doc in docs])

    # Prepare instructions for GPT
    instructions = f"""

You are a professional blog writer. Follow the strict structure and formatting guidelines below.

Word limit: {word_limit} words  
Tone: {tone}  
Topic: {query}  
Industry: {industry or "Not specified"}  

If industry is specified, adapt examples, context, and recommendations to that industry.  
If industry is not specified, write for a general business/technology audience.  

# Strict Blog Structure and Formatting Guidelines

1. Title
   - Short, focused, and benefit-driven.
   - Must include important keywords relevant to the topic.

2. Introduction (2 short paragraphs)
   - Start with a compelling business pain point or a thought-provoking question.
   - Clearly explain the core problem addressed by the topic, using keywords naturally.
   - Second paragraph should set the stage for what the reader will learn, guiding them naturally into the content.
   - Each paragraph must be 3–5 lines maximum.

3. Body
   - Use clear, descriptive subheadings for each section.
   - After each subheading, add a short introductory paragraph (3–5 lines) that sets the context.
   - Ensure logical flow and strong connections between all sections.
   - Present steps or points in logical order, directly relevant to the section.
   - Explain technical concepts simply, while adding depth where valuable.
   - Integrate industry best practices, practical tips, or relevant SAP standards.
   - Formatting rules:
     * Each paragraph must be 3–5 lines.
     * Each sentence must be no more than 25 words.
     * Use bullet points or numbered lists where they improve clarity.

4. Conclusion
   - Heading must be concise, engaging, and summarize the benefit or next step. Avoid generic terms like "Wrap-up" or "Summary."
   - Provide a brief, clear summary of the key takeaways.

# Prohibited Content
- Do not use filler phrases such as "In this blog," "This blog will cover," or "We will explore."
- Do not generate generic or fabricated content. All insights must be real and context-based.
- Do not include instructions or conversational filler in the final output.

"""

    prompt = f"{instructions}\n\nReferences:\n{all_docs}"

    # GPT request
    message_text = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        messages=message_text,
        model="Codetest",
        max_tokens=1600,
        temperature=0.7
    )

    blog = response.choices[0].message.content
    return blog

# ---------------------------
# Generate blog button
# ---------------------------
if generate_btn and query:
    with st.spinner("Generating blog..."):
        blog_content = generate_blog(query, word_limit, tone, uploaded_file)
    st.subheader("Generated Blog")
    st.write(blog_content)
