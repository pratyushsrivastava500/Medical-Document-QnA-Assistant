"""
AI-Powered Medical Document Q&A Assistant
Main Streamlit Application
"""
import os
import streamlit as st
from pathlib import Path
import time
from datetime import datetime
import base64

# Import custom modules
from config.config import Config
from utils.document_processor import DocumentProcessor
from utils.embeddings import EmbeddingGenerator
from utils.vector_store import VectorStore
from services.llm_client import LLMClient
from services.conversation_memory import ConversationMemory
from services.rag_pipeline import RAGPipeline
from services.google_drive import GoogleDriveClient
from services.report_generator import ReportGenerator
from utils.pdf_exporter import PDFExporter


# Function to load and encode background image
@st.cache_data
def get_base64_image(image_path):
    """Convert image to base64 string for CSS"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error loading background image: {e}")
        return ""


# Page configuration
st.set_page_config(
    page_title="Medical Document Q&A Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load background image
bg_image = get_base64_image("images/clipboard-stethoscope.jpg")

# Custom CSS with medical background
st.markdown(f"""
<style>
    /* Main app background with medical theme */
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.65), rgba(255, 255, 255, 0.65)),
                    url('data:image/jpeg;base64,{bg_image}') no-repeat center center fixed;
        background-size: cover;
    }}
    
    /* Sidebar background with slight transparency */
    [data-testid="stSidebar"] {{
        background: linear-gradient(rgba(240, 248, 255, 0.75), rgba(230, 240, 255, 0.75)),
                    url('data:image/jpeg;base64,{bg_image}') no-repeat center center;
        background-size: cover;
    }}
    
    /* Remove ALL white backgrounds */
    .main .block-container {{
        background-color: transparent !important;
    }}
    
    /* All content blocks transparent */
    div[data-testid="stVerticalBlock"] > div {{
        background-color: transparent !important;
    }}
    
    div[data-testid="stHorizontalBlock"] {{
        background-color: transparent !important;
    }}
    
    /* Expander sections */
    .streamlit-expanderHeader, .streamlit-expanderContent {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(8px);
    }}
    
    /* Columns */
    div[data-testid="column"] {{
        background-color: transparent !important;
    }}
    
    /* Forms */
    div[data-testid="stForm"] {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(8px);
        border-radius: 10px;
        padding: 1rem;
    }}
    
    /* Make text areas and inputs transparent */
    .stTextArea textarea, .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(31, 119, 180, 0.3);
    }}
    
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {{
        background-color: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(8px);
        border: 2px dashed rgba(31, 119, 180, 0.4) !important;
    }}
    
    /* Sidebar expander */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {{
        background-color: rgba(255, 255, 255, 0.70) !important;
    }}
    
    section[data-testid="stSidebar"] .streamlit-expanderContent {{
        background-color: rgba(255, 255, 255, 0.65) !important;
    }}
    
    /* Remove white from all divs in main area */
    .main > div {{
        background-color: transparent !important;
    }}
    
    /* Stats/Metric boxes */
    [data-testid="stMetricValue"] {{
        background-color: transparent !important;
    }}
    
    div[data-testid="metric-container"] {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(8px);
        border-radius: 8px;
        padding: 1rem;
    }}
    
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-top: 0rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9);
    }}
    
    .citation-box {{
        background-color: rgba(240, 242, 246, 0.75) !important;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }}
    
    .source-badge {{
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }}
    
    .stChatMessage {{
        background-color: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(10px);
        border-radius: 8px;
    }}
    
    /* Headers with better visibility */
    h1, h2, h3, h4, h5, h6 {{
        color: #1f77b4 !important;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.9);
    }}
    
    /* Remove white from markdown */
    .stMarkdown {{
        background-color: transparent !important;
    }}
    
    /* Radio buttons background */
    div[role="radiogroup"] {{
        background-color: transparent !important;
        padding: 0.8rem;
        border-radius: 8px;
        backdrop-filter: none !important;
    }}
    
    /* Radio button labels */
    label[data-baseweb="radio"] {{
        background-color: transparent !important;
    }}
    
    /* Radio button container */
    .stRadio {{
        background-color: transparent !important;
    }}
    
    .stRadio > div {{
        background-color: transparent !important;
    }}
    
    /* Checkbox background */
    .stCheckbox {{
        background-color: transparent !important;
    }}
    
    /* Selectbox */
    .stSelectbox > div > div {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(8px);
    }}
    
    /* Multiselect */
    .stMultiSelect > div {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(8px);
    }}
    
    /* Buttons */
    .stButton button {{
        background-color: rgba(31, 119, 180, 0.9) !important;
        color: white !important;
        border: none;
        backdrop-filter: blur(5px);
        font-weight: 600;
    }}
    
    .stButton button:hover {{
        background-color: rgba(31, 119, 180, 1) !important;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3);
    }}
    
    /* Download button */
    .stDownloadButton button {{
        background-color: rgba(76, 175, 80, 0.9) !important;
        color: white !important;
    }}
    
    .stDownloadButton button:hover {{
        background-color: rgba(76, 175, 80, 1) !important;
    }}
    
    /* Remove white background from top section */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    
    /* Remove white from bottom section */
    footer {{
        background-color: transparent !important;
    }}
    
    /* Footer caption text */
    .stCaption {{
        background-color: transparent !important;
    }}
    
    /* Any div containing footer text */
    footer > div {{
        background-color: transparent !important;
    }}
    
    /* Remove white from chat input container at bottom */
    .stChatInputContainer {{
        background-color: transparent !important;
    }}
    
    div[data-testid="stChatInput"] {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }}
    
    /* Chat input text area */
    div[data-testid="stChatInput"] textarea {{
        background-color: rgba(255, 255, 255, 0.70) !important;
        backdrop-filter: blur(8px);
    }}
    
    /* The entire bottom section wrapper */
    div.stChatFloatingInputContainer {{
        background-color: transparent !important;
    }}
    
    /* Target any remaining white section */
    section[tabindex="-1"] {{
        background-color: transparent !important;
    }}
    
    /* Bottom fixed container */
    .stBottom {{
        background-color: transparent !important;
    }}
    
    /* All sections */
    section {{
        background-color: transparent !important;
    }}
    
    /* Remove white from all stApp children */
    .stApp > div {{
        background-color: transparent !important;
    }}
    
    /* Target the main content wrapper */
    section.main > div {{
        background-color: transparent !important;
    }}
    
    /* Remove white from element containers */
    .element-container {{
        background-color: transparent !important;
    }}
    
    /* Specifically target the top white section */
    div[data-testid="stAppViewContainer"] {{
        background-color: transparent !important;
    }}
    
    div[data-testid="stAppViewContainer"] > section {{
        background-color: transparent !important;
    }}
    
    /* Remove white from bottom toolbar */
    div[data-testid="stBottomBlockContainer"] {{
        background-color: transparent !important;
    }}
    
    div[data-testid="stStatusWidget"] {{
        background-color: transparent !important;
    }}
    
    /* Dataframe/Table */
    .stDataFrame {{
        background-color: rgba(255, 255, 255, 0.70) !important;
        backdrop-filter: blur(10px);
        border-radius: 8px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.60) !important;
        backdrop-filter: blur(8px);
    }}
    
    /* Tooltips */
    .stTooltipIcon {{
        color: #1f77b4 !important;
    }}
    
    /* Info/Warning/Error boxes */
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.80) !important;
        backdrop-filter: blur(10px);
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_components():
    """Initialize all components (cached for efficiency)"""
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize components
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore()
        llm_client = LLMClient()
        
        return embedding_generator, vector_store, llm_client
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, None, None


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'gdrive_client' not in st.session_state:
        st.session_state.gdrive_client = GoogleDriveClient()
    
    if 'gdrive_files' not in st.session_state:
        st.session_state.gdrive_files = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "Q&A"  # Default mode
    
    if 'report_config' not in st.session_state:
        st.session_state.report_config = {
            "sections": [],
            "instructions": ""
        }


def display_header():
    """Display application header"""
    st.markdown('<div class="main-header">🏥 Medical Document Q&A Assistant</div>', 
                unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #666;'>
    Upload medical documents or connect to Google Drive, then ask questions to get grounded answers with citations.
    </p>
    """, unsafe_allow_html=True)
    st.divider()


def sidebar_controls(vector_store, embedding_generator):
    """Display sidebar controls"""
    with st.sidebar:
        # Mode selector at the top
        st.header("⚙️ Settings")
        mode = st.radio(
            "Select Mode:",
            ["💬 Q&A Mode", "📊 Report Generation"],
            horizontal=False,
            help="Choose between asking questions or generating reports"
        )
        
        st.divider()
        
        st.header("📁 Document Management")
        
        # File upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload medical documents",
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word, Excel, Text, Images"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files", type="primary"):
                process_uploaded_files(uploaded_files, vector_store, embedding_generator)
        
        st.divider()
        
        # Google Drive section
        st.subheader("📁 Google Drive Integration")
        
        st.markdown("Enter publicly accessible Google Drive file links (one per line):")
        gdrive_files_input = st.text_area(
            "Google Drive File Links",
            placeholder="https://drive.google.com/file/d/FILE_ID_1/view\nhttps://drive.google.com/file/d/FILE_ID_2/view",
            help="Paste shareable links to individual files (one per line). Files must be set to 'Anyone with the link can view'",
            height=100,
            key="gdrive_files_input"
        )
        
        if gdrive_files_input:
            if st.button("📥 Download & Process Google Drive Files", type="primary"):
                file_links = [link.strip() for link in gdrive_files_input.split('\n') if link.strip()]
                
                if file_links:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, file_link in enumerate(file_links):
                        status_text.text(f"Processing link {idx + 1}/{len(file_links)}...")
                        
                        try:
                            # Download file and get metadata using the new method
                            status_text.text(f"Downloading file {idx + 1}...")
                            file_info = st.session_state.gdrive_client.download_file_from_link(file_link)
                            
                            if file_info and os.path.exists(file_info['path']):
                                file_name = file_info['name']
                                file_path = file_info['path']
                                
                                if file_name in st.session_state.processed_files:
                                    status_text.text(f"⚠️ {file_name} already processed, skipping...")
                                    time.sleep(1)
                                    continue
                                
                                # Process document
                                status_text.text(f"Processing {file_name}...")
                                documents = st.session_state.document_processor.process_document(
                                    file_path,
                                    file_name,
                                    source_type="gdrive"
                                )
                                
                                # Add Google Drive link to metadata
                                for doc in documents:
                                    doc['metadata']['gdrive_link'] = file_info['link']
                                
                                if documents:
                                    # Generate embeddings
                                    texts = [doc['content'] for doc in documents]
                                    embeddings = embedding_generator.generate_embeddings(texts)
                                    
                                    # Add to vector store
                                    vector_store.add_documents(documents, embeddings)
                                    
                                    st.session_state.processed_files.add(file_name)
                                    st.session_state.gdrive_files.append(file_info)
                                    status_text.text(f"✅ Processed {file_name} ({len(documents)} chunks)")
                                else:
                                    status_text.text(f"⚠️ No content extracted from {file_name}")
                            else:
                                status_text.text(f"❌ Failed to download file from link {idx + 1}")
                        
                        except Exception as e:
                            status_text.text(f"❌ Error: {str(e)[:100]}")
                        
                        progress_bar.progress((idx + 1) / len(file_links))
                        time.sleep(0.5)
                    
                    status_text.text(f"✅ Processed {len(file_links)} Google Drive files!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
        
        st.divider()
        
        # Document statistics
        st.subheader("📊 Statistics")
        doc_count = vector_store.count()
        sources = vector_store.get_all_sources()
        
        st.metric("Total Chunks", doc_count)
        st.metric("Unique Documents", len(sources))
        
        if sources:
            with st.expander("View Loaded Documents"):
                for source in sources:
                    st.text(f"• {source}")
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation_memory.clear_history()
            st.session_state.messages = []
            st.success("Conversation cleared!")
            st.rerun()
        
        # Clear all documents
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                vector_store.clear_all()
                st.session_state.processed_files.clear()
                st.session_state.gdrive_files.clear()
                st.success("All documents cleared!")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm")
        
        return mode


def process_uploaded_files(uploaded_files, vector_store, embedding_generator):
    """Process uploaded files and add to vector store"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        
        if file_name in st.session_state.processed_files:
            status_text.text(f"Skipping {file_name} (already processed)")
            continue
        
        status_text.text(f"Processing {file_name}...")
        
        try:
            # Process document
            documents = st.session_state.document_processor.process_file_object(
                uploaded_file, file_name, source_type="uploaded"
            )
            
            if documents:
                # Generate embeddings
                texts = [doc['content'] for doc in documents]
                embeddings = embedding_generator.generate_embeddings(texts)
                
                # Add to vector store
                vector_store.add_documents(documents, embeddings)
                
                st.session_state.processed_files.add(file_name)
                status_text.text(f"✅ Processed {file_name} ({len(documents)} chunks)")
            else:
                status_text.text(f"⚠️ No content extracted from {file_name}")
        
        except Exception as e:
            status_text.text(f"❌ Error processing {file_name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
        time.sleep(0.5)
    
    status_text.text("✅ All files processed!")
    time.sleep(2)
    st.rerun()


def display_chat_interface(rag_pipeline):
    """Display chat interface"""
    
    # Get vector store from rag pipeline
    vector_store = rag_pipeline.vector_store
    
    # Check if documents are available
    all_sources = vector_store.get_all_sources()
    if not all_sources:
        st.warning("⚠️ No documents uploaded yet. Please upload documents first using the sidebar.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if available and message is not an error
            if message["role"] == "assistant" and "citations" in message:
                # Check if message contains error text
                is_error = any(error_text in message["content"] for error_text in [
                    "Error:", "403", "Forbidden", "Unable to generate response",
                    "Client Error", "Server Error", "Connection Error"
                ])
                
                if message["citations"] and not is_error:
                    display_citations(message["citations"])
    
    # Show example questions if no messages yet
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #1f77b4; margin-bottom: 0px;'>🤔 What can I help you with?</h3>
            <p style='font-size: 1.1rem; color: #666; margin-top: 0px; margin-bottom: 0px;'>
                Ask questions about your medical documents and get accurate, grounded answers with citations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📝 Example Questions You Can Ask:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Document Analysis:**
            - What are the main findings in the patient's report?
            - Can you summarize all the test results?
            - What medications and dosages are mentioned?
            """)
        
        with col2:
            st.markdown("""
            **💡 Medical Insights:**
            - What is the patient's diagnosis and recommended treatments?
            - Explain this medical term or abbreviation
            - Compare lab results from different dates
            """)
        
        st.info("📊 **Want a comprehensive report?** Switch to **Report Generation** mode in the sidebar to create professional medical reports with patient info, test results, diagnosis, and treatment plans!")
        
        st.success("✨ **All answers are grounded in your uploaded documents with source citations for verification!**")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Q&A always searches all documents
                response = rag_pipeline.query(prompt, selected_sources=None)
                
                st.markdown(response["answer"])
                
                # Display citations only if response is not an error
                is_error = any(error_text in response["answer"] for error_text in [
                    "Error:", "403", "Forbidden", "Unable to generate response",
                    "Client Error", "Server Error", "Connection Error"
                ])
                
                if response["citations"] and not is_error:
                    display_citations(response["citations"])
                
                # Store assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "citations": response["citations"] if not is_error else []
                })
    
    # Add spacing to prevent overlap with fixed footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Footer at the bottom
    st.markdown("""
    <div style='position: fixed; bottom: 0; left: 0; right: 0; text-align: center; 
                padding: 10px; background: transparent; 
                backdrop-filter: blur(10px); border-top: 1px solid rgba(0,0,0,0.1);
                z-index: 999;'>
        <p style='margin: 0; color: #666; font-size: 0.8rem;'>
        Powered by Euriai GPT-4.1-Nano | Built with Streamlit | All answers are grounded in provided documents
        </p>
    </div>
    <style>
    /* Add bottom padding to main content to prevent overlap with fixed footer */
    .main .block-container {{
        padding-bottom: 80px !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def display_report_interface(report_generator, pdf_exporter=None):
    """Display report generation interface"""
    st.subheader("📊 Generate Medical Report")
    
    st.markdown("""
    Generate structured medical reports from your uploaded documents. 
    When multiple documents are uploaded, a separate report will be generated for each document.
    """)
    
    # Get vector store from report generator
    vector_store = report_generator.vector_store
    
    # Check if documents are available
    all_sources = vector_store.get_all_sources()
    if not all_sources:
        st.warning("⚠️ No documents uploaded yet. Please upload documents first using the sidebar.")
        return
    
    # Show number of reports that will be generated
    num_reports = len(all_sources)
    if num_reports == 1:
        st.info(f"📄 **{num_reports}** report will be generated from: **{all_sources[0]}**")
    else:
        st.info(f"📄 **{num_reports}** separate reports will be generated (one per document)")
        with st.expander("📋 Documents to Process", expanded=False):
            for idx, source in enumerate(sorted(all_sources), 1):
                st.markdown(f"{idx}. **{source}**")
    
    st.divider()
    
    # Section selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Select Report Sections:**")
        
        section_mapping = {
            "Introduction": "introduction",
            "Clinical Findings": "clinical_findings",
            "Diagnosis": "diagnosis",
            "Treatment Plan": "treatment_plan",
            "Summary": "summary"
        }
        
        selected_sections = []
        for display_name, section_key in section_mapping.items():
            if st.checkbox(display_name, key=f"section_{section_key}"):
                selected_sections.append(section_key)
    
    with col2:
        st.markdown("**Report Instructions:**")
        
        user_instructions = st.text_area(
            "Additional Instructions",
            placeholder="e.g., 'Focus on patient vitals', 'Highlight lab results', etc.",
            height=150,
            help="Provide specific instructions for report generation"
        )
    
    # Generate button
    if st.button("🔄 Generate Reports", type="primary", use_container_width=True):
        if not selected_sections:
            st.warning("Please select at least one section to include in the report.")
            return
        
        # Generate separate report for each document
        for idx, source_file in enumerate(sorted(all_sources), 1):
            st.divider()
            st.markdown(f"### 📄 Report {idx} of {num_reports}: {source_file}")
            
            with st.spinner(f"Generating report for {source_file}..."):
                # Generate report for this specific document
                report = report_generator.generate_report(
                    sections=selected_sections,
                    user_instructions=user_instructions if user_instructions else None,
                    selected_sources=[source_file]  # Generate for single document
                )
                
                if report.get("success"):
                    st.success(f"✅ Report generated successfully for {source_file}")
                    
                    # Display report - Simple text only, no visualizations
                    st.markdown("# Medical Analysis Report\n")
                    st.markdown("---\n")
                    st.markdown(f"**Document:** {source_file}\n")
                    st.markdown("---\n")
                    
                    # Get section titles
                    section_titles = {
                        "introduction": "Introduction",
                        "clinical_findings": "Clinical Findings",
                        "diagnosis": "Diagnosis",
                        "treatment_plan": "Treatment Plan",
                        "summary": "Summary"
                    }
                    
                    # Display each section - simple text display only
                    sections = report.get("generated_sections", {})
                    for section_name, section_data in sections.items():
                        # Skip patient_tables and graphs_and_charts sections
                        if section_name in ["patient_tables", "graphs_and_charts"]:
                            continue
                            
                        title = section_titles.get(section_name, section_name.replace("_", " ").title())
                        st.markdown(f"## {title}\n")
                        
                        # Display simple text content
                        content = section_data.get("content", "")
                        if content:
                            st.markdown(content)
                        else:
                            st.markdown("*No information available for this section.*")
                        
                        st.markdown("\n---\n")
                    
                    st.markdown("*This report is based on the analysis of provided medical documentation. ")
                    st.markdown("All information should be verified against original source materials.*\n")
                    
                    st.divider()
                    
                    # PDF Download Button
                    st.markdown("### 📄 Download Report")
                    
                    if pdf_exporter:
                        import tempfile
                        import os
                        from datetime import datetime as dt
                        
                        # Create temp PDF file
                        temp_dir = tempfile.gettempdir()
                        clean_filename = source_file.replace('.pdf', '').replace('.docx', '').replace('.xlsx', '')
                        pdf_filename = f"report_{clean_filename}_{dt.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        pdf_path = os.path.join(temp_dir, pdf_filename)
                        
                        try:
                            # Export to PDF
                            with st.spinner("Generating PDF..."):
                                result = pdf_exporter.export_report_to_pdf(
                                    report, 
                                    pdf_path,
                                    chart_images=None  # No charts
                                )
                            
                            # Show download button
                            if result.get("success"):
                                # Read PDF for download
                                with open(pdf_path, 'rb') as f:
                                    pdf_data = f.read()
                                
                                st.download_button(
                                    label="📥 Download Report as PDF",
                                    data=pdf_data,
                                    file_name=pdf_filename,
                                    mime="application/pdf",
                                    type="primary",
                                    use_container_width=True,
                                    key=f"pdf_download_{idx}"
                                )
                                st.success("✅ Report is ready! Click the button above to download.")
                            else:
                                st.error(f"⚠️ PDF export failed: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"⚠️ Error generating PDF: {str(e)}")
                            import traceback
                            with st.expander("Show Error Details"):
                                st.code(traceback.format_exc())
                    else:
                        st.warning("⚠️ PDF export not available. Install required packages.")
                    
                    # Show metadata in expander
                    with st.expander(f"📋 Report Metadata - {source_file}"):
                        st.json(report["metadata"])
                else:
                    st.error(f"❌ Report generation failed for {source_file}: {report.get('error', 'Unknown error')}")
                    if "available_sections" in report:
                        st.info(f"Available sections: {', '.join(report['available_sections'])}")
                    if "available_sources" in report:
                        st.info(f"Available documents: {', '.join(report['available_sources'])}")


def display_citations(citations):
    """Display citations in a formatted way"""
    if not citations:
        return
    
    # Get unique sources for summary
    unique_sources = list(set([c.get('source', 'Unknown') for c in citations]))
    
    with st.expander(f"📚 Sources ({len(citations)} chunks from {len(unique_sources)} document(s))", expanded=False):
        st.markdown("**Retrieved Evidence:**")
        st.caption("The following document excerpts were used to generate the answer above.")
        st.divider()
        
        for idx, citation in enumerate(citations, 1):
            source = citation.get('source', 'Unknown')
            source_type = citation.get('source_type', 'uploaded')
            score = citation.get('relevance_score', 0)
            chunk_id = citation.get('chunk_id', 0)
            chunk_content = citation.get('chunk_content', '')
            gdrive_link = citation.get('metadata', {}).get('gdrive_link') if isinstance(citation.get('metadata'), dict) else None
            
            # Citation header with source info
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Source {idx}: {source}**")
                # Show Google Drive link prominently if available
                if gdrive_link:
                    st.markdown(f"🔗 [Open in Google Drive]({gdrive_link})")
                elif source_type == 'gdrive':
                    # Try to find link from session state
                    for file_info in st.session_state.get('gdrive_files', []):
                        if file_info.get('name') == source and file_info.get('link'):
                            st.markdown(f"🔗 [Open in Google Drive]({file_info['link']})")
                            break
            with col2:
                st.caption(f"Chunk #{chunk_id}")
            
            st.caption(f"📊 Relevance: {score:.1%} • 📁 Type: {source_type.upper()}")
            
            # Display the actual chunk content in a clean format
            if chunk_content:
                st.markdown("**Content:**")
                # Show full content in an info box with proper formatting
                display_text = chunk_content.strip()
                st.info(display_text, icon="📄")
            
            # Add separator between citations except for the last one
            if idx < len(citations):
                st.divider()


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Initialize components
    embedding_generator, vector_store, llm_client = initialize_components()
    
    if not all([embedding_generator, vector_store, llm_client]):
        st.error("Failed to initialize components. Please check your configuration.")
        st.stop()
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        llm_client=llm_client,
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        conversation_memory=st.session_state.conversation_memory
    )
    
    # Initialize Report Generator
    report_generator = ReportGenerator(
        llm_client=llm_client,
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        document_processor=st.session_state.document_processor
    )
    
    # Initialize PDF Exporter
    try:
        from utils.pdf_exporter import PDFExporter
        pdf_exporter = PDFExporter()
    except Exception as e:
        st.error(f"⚠️ PDF export not available: {str(e)}")
        pdf_exporter = None  # PDF export not available
    
    # Sidebar controls
    mode = sidebar_controls(vector_store, embedding_generator)
    
    # Display appropriate interface based on mode
    if mode == "💬 Q&A Mode":
        display_chat_interface(rag_pipeline)
    else:  # Report Generation
        display_report_interface(report_generator, pdf_exporter)


if __name__ == "__main__":
    main()
