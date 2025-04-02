import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Somalia AI Readiness Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "api_key" not in st.session_state:
    # Try to load from environment first (but don't raise error if missing)
    load_dotenv()
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "document_list" not in st.session_state:
    st.session_state.document_list = []
if "engine_initialized" not in st.session_state:
    st.session_state.engine_initialized = False

# App header
st.title("Somalia AI Readiness Analyzer")
st.markdown("""
This tool lets you ask questions about Somalia's AI readiness based on a collection of research papers and documents.
You can query specific documents or the entire collection to gain insights into Somalia's AI readiness landscape.
""")

# API Key input section
with st.expander("API Key Settings", expanded=not st.session_state.api_key):
    api_key_input = st.text_input(
        "Enter your Google Gemini API Key:",
        value=st.session_state.api_key,
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Save API Key"):
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.session_state.engine_initialized = False  # Force reinitialization
                st.success("API key saved successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Please enter a valid API key")
    
    st.info("Your API key is required to use this application. It will be stored in your browser's session storage only.")
    st.warning("IMPORTANT: Your API key will be used to make requests to Google's API, and any usage will count against your quota.")

# Function to initialize vector store and document query engine
def initialize_query_engine(api_key):
    try:
        logger.info("Initializing query engine...")
        # Import here to avoid errors if API key isn't provided
        from document_query_engine import DocumentQueryEngine
        from vector_store import VectorStore
        
        # Load vector store
        vector_store_path = "results/vector_store"
        logger.info(f"Loading vector store from {vector_store_path}")
        
        # Check if directory exists
        if not os.path.exists(vector_store_path):
            logger.error(f"Vector store directory not found: {vector_store_path}")
            return None, None, []
            
        # Check if directory has files
        if len(os.listdir(vector_store_path)) <= 1:  # Only .gitkeep
            logger.error(f"Vector store directory is empty: {vector_store_path}")
            return None, None, []
            
        # Load the vector store
        vector_store = VectorStore(api_key)
        vector_store.load(vector_store_path)
        
        # Create document query engine
        query_engine = DocumentQueryEngine(vector_store, api_key)
        
        # Get list of unique documents
        docs = set()
        for doc in vector_store.documents:
            if "filename" in doc and doc["filename"]:
                base_filename = os.path.basename(doc["filename"])
                docs.add(base_filename)
        
        logger.info(f"Found {len(docs)} unique documents")
        return query_engine, vector_store, sorted(list(docs))
    except Exception as e:
        logger.error(f"Error initializing query engine: {str(e)}")
        st.error(f"Error initializing query engine: {str(e)}")
        return None, None, []

# Initialize the query engine if API key is provided
if st.session_state.api_key and not st.session_state.engine_initialized:
    with st.spinner("Initializing AI engine..."):
        query_engine, vector_store, document_list = initialize_query_engine(st.session_state.api_key)
        if query_engine:
            st.session_state.query_engine = query_engine
            st.session_state.vector_store = vector_store
            st.session_state.document_list = document_list
            st.session_state.engine_initialized = True
            st.success("AI engine initialized successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to initialize AI engine. Please check if vector store exists and API key is valid.")

# Only show the main UI if the engine is initialized
if st.session_state.api_key and st.session_state.engine_initialized:
    # Sidebar
    with st.sidebar:
        st.header("Options")
        
        # Document filter
        doc_filter = st.selectbox(
            "Filter by document (optional)",
            ["All Documents"] + st.session_state.document_list
        )
        
        # Knowledge toggle
        use_model_knowledge = st.toggle("Use model's knowledge", value=True, 
                                      help="When enabled, the model can use its general knowledge to supplement document information")
        
        # Query templates
        st.subheader("Query Templates")
        template_options = [
            "What are the key challenges for AI adoption in Somalia?",
            "What infrastructure is needed for AI readiness in Somalia?",
            "How does Somalia compare to other African nations in AI readiness?",
            "What policy recommendations would improve Somalia's AI readiness?",
            "Generate a full paper on Somalia's AI readiness",
            "Create an AI readiness assessment questionnaire for Somalia",
        ]
        
        def set_template(template):
            st.session_state.query_input = template
        
        for template in template_options:
            st.button(template, on_click=set_template, args=(template,), key=f"template_{template}")
        
        # Report generation options
        st.subheader("Report Generation")
        
        report_type = st.selectbox(
            "Report Type",
            ["Standard Report", "Comprehensive Report", "Executive Summary", "Questionnaire"]
        )
        
        report_length = st.selectbox(
            "Report Length",
            ["Standard", "Short", "Long (10+ pages)"]
        )
        
        if st.button("Generate Report"):
            # Map UI selections to parameter values
            type_map = {
                "Standard Report": "standard",
                "Comprehensive Report": "comprehensive", 
                "Executive Summary": "executive",
                "Questionnaire": "questionnaire"
            }
            
            length_map = {
                "Standard": "standard",
                "Short": "short",
                "Long (10+ pages)": "long"
            }
            
            st.session_state.generate_report = {
                "type": type_map[report_type],
                "length": length_map[report_length]
            }

    # Main chat interface
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
            
            try:
                # Get response
                response = st.session_state.query_engine.query(
                    user_input,
                    document_filter=document_filter,
                    use_model_knowledge=use_model_knowledge
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": user_input,
                    "response": response,
                    "document_filter": document_filter,
                    "used_model_knowledge": use_model_knowledge,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Clear input
                st.session_state.query_input = ""
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"Error processing query: {str(e)}")

    # Handle report generation
    if st.session_state.get("generate_report", False):
        report_settings = st.session_state.generate_report
        
        with st.spinner(f"Generating {report_settings['type']} report (length: {report_settings['length']})..."):
            try:
                # Special handling for questionnaire
                if report_settings['type'] == 'questionnaire':
                    report = st.session_state.query_engine.generate_report(
                        "Somalia AI Readiness Assessment Questionnaire",
                        max_length=report_settings["length"],
                        report_type=report_settings["type"]
                    )
                    filename = "somalia_ai_readiness_questionnaire.md"
                else:
                    # Regular report generation
                    report = st.session_state.query_engine.generate_report(
                        "Somalia AI Readiness Analysis",
                        max_length=report_settings["length"],
                        report_type=report_settings["type"]
                    )
                    filename = "somalia_ai_readiness_report.md"
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": f"Generate a {report_settings['length']} {report_settings['type']} on Somalia's AI readiness",
                    "response": report,
                    "document_filter": None,
                    "is_report": True,
                    "report_type": report_settings['type'],
                    "report_length": report_settings['length'],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": filename
                })
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                st.error(f"Error generating report: {str(e)}")
        
        # Reset flag
        st.session_state.generate_report = False

    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("Results")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query: {chat['query'][:100]}{'...' if len(chat['query']) > 100 else ''}", expanded=(i == 0)):
                # Show metadata about the query
                metadata_cols = st.columns(3)
                
                with metadata_cols[0]:
                    if chat.get('document_filter'):
                        st.caption(f"📄 Document: {chat['document_filter']}")
                    else:
                        st.caption("📚 All Documents")
                        
                with metadata_cols[1]:
                    if chat.get('is_report'):
                        st.caption(f"📊 {chat.get('report_type', 'Standard')} Report ({chat.get('report_length', 'Standard')})")
                    elif chat.get('used_model_knowledge', False):
                        st.caption("🧠 Using model knowledge")
                    else:
                        st.caption("📝 Document data only")
                        
                with metadata_cols[2]:
                    timestamp = chat.get('timestamp', 'Now')
                    st.caption(f"⏱️ {timestamp}")
                
                # Display the response
                st.markdown(chat['response'])
                
                # Download button for reports
                if chat.get('is_report', False):
                    filename = chat.get('filename', "somalia_ai_readiness_report.md")
                    st.download_button(
                        "Download as Markdown",
                        chat['response'],
                        filename
                    )

    # Instructions at the bottom
    with st.expander("How to use this tool"):
        st.markdown("""
        ## How to Use This Tool
        
        1. **Enter API Key**: Provide your Google Gemini API key in the API Key Settings section
        2. **Ask a Question**: Type your query in the text area and click "Submit Query"
        3. **Filter by Document**: Use the sidebar dropdown to focus on a specific document
        4. **Knowledge Toggle**: Choose whether to use just document data or also the model's general knowledge
        5. **Use Templates**: Click on template queries in the sidebar for common questions
        6. **Generate Reports**: 
           - Choose a report type (Standard, Comprehensive, Executive Summary, or Questionnaire)
           - Select length (Standard, Short, or Long)
           - Click "Generate Report"
        7. **Download Results**: Download generated reports using the download button
        
        This tool analyzes a collection of documents on AI readiness in Somalia and uses AI to extract relevant information.
        """)

else:
    # Show this if API key is not provided or engine failed to initialize
    if not st.session_state.api_key:
        st.warning("Please enter your Google Gemini API key to use this application.")
        st.markdown("""
        ### How to get a Google Gemini API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Create a new API key
        4. Copy and paste the key into the API Key field above
        """)
    else:
        st.error("Failed to initialize AI engine. Please check if vector store exists and your API key is valid.")
        
        # Show more debug information
        st.markdown("### Debug Information")
        vector_store_path = "results/vector_store"
        if not os.path.exists(vector_store_path):
            st.error(f"Vector store directory not found: {vector_store_path}")
        elif len(os.listdir(vector_store_path)) <= 1:
            st.error(f"Vector store directory is empty or contains only hidden files: {vector_store_path}")
            st.info(f"Files found: {', '.join(os.listdir(vector_store_path))}")
        else:
            st.info(f"Vector store directory exists with {len(os.listdir(vector_store_path))} files")
            st.info(f"Files found: {', '.join(os.listdir(vector_store_path)[:5])}{'...' if len(os.listdir(vector_store_path)) > 5 else ''}")