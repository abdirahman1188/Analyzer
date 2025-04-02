import streamlit as st
import os
import json
from dotenv import load_dotenv
from document_query_engine import DocumentQueryEngine
from vector_store import VectorStore

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Page configuration
st.set_page_config(
    page_title="Somalia AI Readiness Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to initialize vector store and document query engine
@st.cache_resource
def initialize_query_engine():
    # Load vector store
    vector_store = VectorStore(api_key)
    vector_store.load("results/vector_store")
    
    # Create document query engine
    query_engine = DocumentQueryEngine(vector_store, api_key)
    
    # Get list of unique documents
    docs = set()
    for doc in vector_store.documents:
        if "filename" in doc and doc["filename"]:
            base_filename = os.path.basename(doc["filename"])
            docs.add(base_filename)
    
    return query_engine, sorted(list(docs))

# Initialize resources
query_engine, document_list = initialize_query_engine()

# App header
st.title("Somalia AI Readiness Analyzer")
st.markdown("""
This tool lets you ask questions about Somalia's AI readiness based on a collection of research papers and documents.
You can query specific documents or the entire collection to gain insights into Somalia's AI readiness landscape.
""")

# Sidebar
with st.sidebar:
    st.header("Options")
    
    # Document filter
    doc_filter = st.selectbox(
        "Filter by document (optional)",
        ["All Documents"] + document_list
    )
    
    # Query templates
    st.subheader("Query Templates")
    template_options = [
        "What are the key challenges for AI adoption in Somalia?",
        "What infrastructure is needed for AI readiness in Somalia?",
        "How does Somalia compare to other African nations in AI readiness?",
        "What policy recommendations would improve Somalia's AI readiness?",
        "Generate a full paper on Somalia's AI readiness",
    ]
    
    def set_template(template):
        st.session_state.query_input = template
    
    for template in template_options:
        st.button(template, on_click=set_template, args=(template,), key=f"template_{template}")
    
    # Report generation
    st.subheader("Report Generation")
    if st.button("Generate Full AI Readiness Report"):
        st.session_state.generate_report = True

# Main chat interface
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

user_input = st.text_area(
    "Enter your query about Somalia's AI readiness:",
    value=st.session_state.query_input,
    height=100,
    key="query_area"
)

# Submit button
col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("Submit Query")

# Process query
if submit_button and user_input:
    with st.spinner("Analyzing documents..."):
        # Set document filter
        document_filter = None if doc_filter == "All Documents" else doc_filter
        
        # Get response
        response = query_engine.query(
            user_input,
            document_filter=document_filter
        )
        
        # Add to chat history
        st.session_state.chat_history.append({
            "query": user_input,
            "response": response,
            "document_filter": document_filter
        })
        
        # Clear input
        st.session_state.query_input = ""

# Handle report generation
if st.session_state.get("generate_report", False):
    with st.spinner("Generating comprehensive AI readiness report..."):
        report = query_engine.generate_report(
            "Somalia AI Readiness Analysis"
        )
        
        # Add to chat history
        st.session_state.chat_history.append({
            "query": "Generate a comprehensive report on Somalia's AI readiness",
            "response": report,
            "document_filter": None,
            "is_report": True
        })
    
    # Reset flag
    st.session_state.generate_report = False

# Display chat history
if st.session_state.chat_history:
    st.divider()
    st.subheader("Results")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Query: {chat['query'][:100]}{'...' if len(chat['query']) > 100 else ''}", expanded=(i == 0)):
            if chat.get('document_filter'):
                st.caption(f"Document: {chat['document_filter']}")
            
            st.markdown(chat['response'])
            
            # Download button for reports
            if chat.get('is_report', False):
                st.download_button(
                    "Download Report as Markdown",
                    chat['response'],
                    "somalia_ai_readiness_report.md"
                )

# Instructions at the bottom
with st.expander("How to use this tool"):
    st.markdown("""
    ## How to Use This Tool
    
    1. **Ask a Question**: Type your query in the text area and click "Submit Query"
    2. **Filter by Document**: Use the sidebar dropdown to focus on a specific document
    3. **Use Templates**: Click on template queries in the sidebar for common questions
    4. **Generate Reports**: Click "Generate Full AI Readiness Report" for a comprehensive analysis
    5. **Download Reports**: Download generated reports using the download button
    
    This tool analyzes a collection of documents on AI readiness in Somalia and uses AI to extract relevant information.
    """)