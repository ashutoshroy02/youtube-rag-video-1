import streamlit as st
from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

st.title("RAG for youtube")

# Initialize Pinecone
pc = Pinecone(api_key='pcsk_jm186_Tu4FnqfZuDvFkL87vYvmN7UREMfFEn6APKxfuNq8p4XaSj9cMb9JFqDnTpXQp55')
index_name = "commerce-gpt-v1"
index = pc.Index(index_name)

# Initialize Google embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw"
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize Groq client
groq_client = Groq(api_key="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")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("What is your question?")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Get relevant documents from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    docs = retriever.invoke(prompt)

    # Display retrieved documents and their scores (for debugging)
    st.sidebar.write("### Retrieved Documents:")
    if docs:
        for i, doc in enumerate(docs):
            st.sidebar.write(f"Document {i+1}:")
            st.sidebar.write(f"Content: {doc.page_content}")
            if hasattr(doc.metadata, 'score'):
                st.sidebar.write(f"Similarity Score: {doc.metadata.score}")
            st.sidebar.write("---")
    else:
        st.sidebar.write("No relevant documents found in the vector store.")

    # Combine all retrieved documents
    docs_text = "".join(d.page_content for d in docs) if docs else ""

    # Create chat completion with Groq
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""You are an assistant for question-answering tasks.
Use the following context to answer the question. If the context is empty or irrelevant,
explicitly mention that you're using general knowledge instead.

Context from vector store: {docs_text}

Please start your response with either:
[Using Vector Store Knowledge] - if your answer is based on the provided context
[Using General Knowledge] - if you're using your general knowledge because no relevant context was found

Keep your answer concise within three sentences."""
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    result = chat_completion.choices[0].message.content

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))
