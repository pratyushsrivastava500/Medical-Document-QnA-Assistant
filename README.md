# 🏥 Medical Document Analysis Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An intelligent AI-powered medical document analysis system using **Retrieval-Augmented Generation (RAG)** for accurate Q&A and professional medical report generation.

![Medical AI](https://img.shields.io/badge/AI-Medical%20Assistant-brightgreen)
![RAG Pipeline](https://img.shields.io/badge/RAG-Powered-orange)
![No Hallucination](https://img.shields.io/badge/Grounded-Responses-red)

---

## 🌟 Key Features

### 💬 Interactive Q&A Mode
- **Multi-format Document Support**: PDF, Word, Excel, Images (OCR)
- **Google Drive Integration**: Direct file links without OAuth
- **Grounded Responses**: Every answer backed by source citations
- **Conversation Context**: Multi-turn dialogue with memory
- **Relevance Scoring**: Confidence indicators for answers
- **No Hallucinations**: Explicit responses when data unavailable

### 📊 Professional Report Generation
Create comprehensive medical reports with customizable sections:

| Section | Description |
|---------|-------------|
| 📝 Introduction | Document overview and background |
| 🔬 Clinical Findings | Observations and test results |
| 🩺 Diagnosis | Medical conditions identified |
| 💊 Treatment Plan | Medications and interventions |
| 📋 Summary | Comprehensive synthesis |

**Report Features:**
- ✅ Per-document generation
- ✅ Duplicate heading removal
- ✅ Professional PDF export
- ✅ Clean formatting
- ✅ Custom instructions support

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medical-document-assistant.git
cd medical-document-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:
```env
# Euriai API Configuration
EURIAI_API_KEY=your_api_key_here
EURIAI_MODEL=gpt-4.1-nano

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Tesseract OCR Path
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

5. **Run the application**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📖 Usage

### Document Upload

**Method 1: Local Files**
1. Use the sidebar file uploader
2. Select PDF, DOCX, XLSX, or image files
3. Wait for automatic processing

**Method 2: Google Drive**
1. Get a shareable link from Google Drive
2. Paste the link in the sidebar input
3. File is downloaded and cached locally

### Q&A Interaction

1. Switch to **"💬 Q&A Mode"** in the sidebar
2. Wait for documents to process
3. Ask questions in the chat interface
4. View answers with source citations
5. Ask follow-up questions with context

**Example Questions:**
```
• What is the patient's primary diagnosis?
• List all medications and dosages mentioned
• Summarize the treatment plan
• What were the lab test results?
• When was the last checkup?
```

### Report Generation

1. Switch to **"📄 Report Generation"** in sidebar
2. Select desired sections (checkboxes)
3. Optionally add custom instructions
4. Click **"🔄 Generate Reports"**
5. Review generated content
6. Click **"📥 Download Report as PDF"**

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                       │
│  (app.py - Chat Interface & Report Display)         │
└───────────────┬─────────────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
┌───────▼──────┐   ┌───▼──────────────┐
│  RAG Pipeline│   │ Report Generator │
│  (Q&A Mode)  │   │  (Report Mode)   │
└───────┬──────┘   └───┬──────────────┘
        │              │
        └──────┬───────┘
               │
    ┌──────────▼───────────┐
    │   Document Processor │
    │  (Multi-format)      │
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │   Vector Store       │
    │   (FAISS Index)      │
    └──────────────────────┘
```

### Tech Stack

- **Frontend**: Streamlit 1.29.0
- **LLM**: Euriai API (GPT-4.1-Nano)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (CPU-optimized)
- **Document Processing**: pypdf, python-docx, openpyxl, pytesseract
- **PDF Export**: ReportLab 4.0.7

### Project Structure

```
medical-document-assistant/
├── app.py                      # Main Streamlit application
├── config/
│   ├── config.py              # Configuration settings
│   └── __init__.py
├── services/
│   ├── llm_client.py          # Euriai API client
│   ├── rag_pipeline.py        # RAG implementation
│   ├── report_generator.py    # Report generation
│   ├── conversation_memory.py # Chat history
│   ├── google_drive.py        # GDrive integration
│   └── __init__.py
├── utils/
│   ├── document_processor.py  # Document parsing
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # FAISS operations
│   ├── pdf_exporter.py        # PDF report export
│   └── __init__.py
├── data/
│   ├── uploaded/              # User files
│   ├── gdrive_cache/         # GDrive cache
│   └── vector_db/            # FAISS index
├── prompts/
│   ├── system_prompt.txt     # LLM instructions
│   └── user_prompt_template.txt
├── requirements.txt          # Dependencies
├── .env                     # Environment config
└── README.md               # Documentation
```

---

## 🔧 Configuration

### API Settings

Edit `.env` file:
```env
EURIAI_API_KEY=your_key_here
EURIAI_MODEL=gpt-4.1-nano
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Document Processing

Edit `config/config.py`:
```python
CHUNK_SIZE = 1000        # Text chunk size
CHUNK_OVERLAP = 200      # Overlap between chunks
TOP_K = 5               # Retrieved chunks per query
```

### Vector Store

- **Engine**: FAISS
- **Index Type**: Flat (exact search)
- **Metric**: Cosine similarity
- **Storage**: `data/vector_db/medical_documents_faiss.index`

---

## 🎯 Core Features

### RAG Pipeline

1. **Document Ingestion**
   - Multi-format parsing (PDF, DOCX, XLSX, images)
   - Intelligent text chunking with overlap
   - Vector embedding generation

2. **Retrieval**
   - Semantic similarity search
   - Top-K relevant chunk selection
   - Relevance score calculation

3. **Generation**
   - Context-aware response generation
   - Source citation tracking
   - Grounded responses (no hallucination)

### Text Cleaning

Automatically removes:
- PDF/HTML headers and footers
- Hyphenation at line breaks
- Page numbers and metadata
- Broken words and fragments
- Irregular spacing and line breaks

### Report Features

- **Section-based Generation**: Independent processing
- **Context Integration**: Uses retrieved documents
- **Professional Formatting**: Clean markdown output
- **Duplicate Removal**: Single headings per section
- **PDF Export**: ReportLab with proper styling

### Error Handling

- ✅ API error detection (403, 500, etc.)
- ✅ No citations on error responses
- ✅ Graceful fallbacks for missing files
- ✅ Clear user feedback

---

## 📦 Dependencies

### Core Libraries
```txt
streamlit==1.29.0           # Web framework
python-dotenv==1.0.0       # Environment config
euriai                     # LLM API client
langchain==0.1.0          # Text processing
sentence-transformers==2.3.1  # Embeddings
faiss-cpu==1.7.4          # Vector store
```

### Document Processing
```txt
pypdf==3.17.0             # PDF parsing
python-docx==1.1.0        # Word documents
openpyxl==3.1.2          # Excel files
pillow==10.1.0           # Image processing
pytesseract==0.3.10      # OCR
```

### Export & Utils
```txt
reportlab==4.0.7         # PDF generation
markdown==3.5.1          # Markdown parsing
pandas==2.1.4            # Data handling
numpy==1.26.2            # Numerical operations
```

See `requirements.txt` for complete list.

---

## 🐛 Troubleshooting

### Common Issues

**OCR Not Working**
```bash
# Install Tesseract OCR
# Windows: Download from https://github.com/tesseract-ocr/tesseract
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract

# Set path in .env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**API Errors (403 Forbidden)**
- Verify `EURIAI_API_KEY` is correct
- Check API quota and rate limits
- Ensure API key has proper permissions
- Citations will be hidden on API errors

**Documents Not Processing**
- Check file format is supported (PDF, DOCX, XLSX, images)
- Verify file is not corrupted
- Check console logs for detailed errors
- Ensure sufficient disk space

**PDF Export Issues**
```bash
# Reinstall reportlab
pip uninstall reportlab
pip install reportlab==4.0.7

# Check write permissions
# Windows: Check temp folder permissions
# Linux/Mac: Check /tmp permissions
```

---

## 🔒 Security & Privacy

- ✅ Documents stored locally only
- ✅ No external data sharing (except LLM API)
- ✅ Google Drive files cached securely
- ✅ Session-based conversation memory
- ✅ Environment variables for credentials
- ✅ No user authentication required (local use)

---

## 📊 Performance

- **Vector Search**: Sub-second retrieval with FAISS
- **Document Processing**: ~2-5 seconds per document
- **Report Generation**: ~10-30 seconds per section
- **Memory Usage**: ~500MB-1GB depending on documents
- **Supported File Size**: Up to 50MB per document

---

## 🚧 Known Limitations

- OCR requires separate Tesseract installation
- Large files (>50MB) may process slowly
- Google Drive links must be publicly accessible
- Vector index not persistent across restarts
- Single-user local deployment only

---

## 🔄 Roadmap

### Planned Features

- [ ] Persistent vector store
- [ ] Multi-language document support
- [ ] Advanced report visualizations
- [ ] Batch document processing
- [ ] Word/HTML export formats
- [ ] User authentication
- [ ] Multi-tenancy support
- [ ] Analytics dashboard
- [ ] API endpoints for integration
- [ ] Mobile-responsive UI

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/medical-document-assistant.git

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/medical-document-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/medical-document-assistant/discussions)
- **Email**: support@example.com

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Powered by [Euriai](https://euron.one) LLM API
- Embeddings by [Sentence Transformers](https://www.sbert.net)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)

---

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

**Made with ❤️ for healthcare professionals**

*Disclaimer: This tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.*

- Internet connection for Groq API
- (Optional) Tesseract OCR for image processing

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-document-assistant.git
cd medical-document-assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

**Get your Groq API key**: [https://console.groq.com/keys](https://console.groq.com/keys)

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Q&A Mode

1. **Upload Documents**
   - Click sidebar → Upload Documents
   - Support: PDF, DOCX, XLSX, PNG, JPG
   - Or paste Google Drive links

2. **Ask Questions**
   - Type your question in the chat
   - Get grounded answers with citations
   - Ask follow-up questions naturally

3. **View Citations**
   - Each answer includes source documents
   - Relevance scores for transparency
   - Full chunk content for verification

### Report Generation Mode

1. **Select Document** - Choose from uploaded files
2. **Choose Sections** - Select desired report sections
3. **Generate Report** - Click to create professional report
4. **Download** - Export as Markdown or PDF

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│         Streamlit Web Interface                      │
│  • File Upload  • Google Drive  • Chat  • Reports  │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              RAG Pipeline Service                    │
│  • Query Processing  • Context Building             │
│  • Response Generation  • Citation Extraction       │
└──┬──────────┬──────────────┬──────────────┬────────┘
   │          │              │              │
   ▼          ▼              ▼              ▼
┌─────┐  ┌────────┐  ┌────────────┐  ┌───────────┐
│ LLM │  │Vector  │  │Conversation│  │ Document  │
│Groq │  │Store   │  │  Memory    │  │ Processor │
│70B  │  │FAISS   │  │  History   │  │Multi-fmt  │
└─────┘  └────────┘  └────────────┘  └───────────┘
```

---

## 🛡️ No-Hallucination Safety

### Four-Layer Protection System

1. **Pre-LLM Relevance Filtering**
   - Filters queries with low relevance scores
   - Prevents irrelevant context from reaching LLM

2. **System Prompt (Global Rules)**
   - Emphatic instructions to use only provided information
   - Forbids external knowledge and assumptions

3. **User Prompt (Per-Query)**
   - Repeated enforcement of grounding rules
   - Clear failure handling for missing information

4. **Post-Generation Validation**
   - Detects "not available" responses
   - Returns empty citations when appropriate

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq (llama-3.3-70b-versatile) | Response generation |
| **Embeddings** | Sentence Transformers | Semantic search |
| **Vector DB** | FAISS | Document retrieval |
| **UI** | Streamlit | Web interface |
| **PDF Export** | ReportLab | Professional reports |
| **Doc Processing** | pypdf, python-docx, openpyxl | Multi-format parsing |

---

## 📁 Project Structure

```
medical-document-assistant/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
├── README.md                       # This file
│
├── config/
│   └── config.py                   # Configuration management
│
├── prompts/
│   ├── system_prompt.txt           # Q&A system prompt
│   ├── user_prompt_template.txt    # Q&A user template
│   ├── report_section_system_prompt.txt
│   ├── report_section_user_prompt.txt
│   ├── report_summary_system_prompt.txt
│   └── graphs_charts_user_prompt.txt
│
├── utils/
│   ├── document_processor.py       # Multi-format parsing
│   ├── pdf_exporter.py             # PDF generation
│   ├── embeddings.py               # Embedding generation
│   └── vector_store.py             # FAISS operations
│
├── services/
│   ├── llm_client.py               # Groq LLM client
│   ├── rag_pipeline.py             # RAG implementation
│   ├── report_generator.py         # Report generation
│   ├── google_drive.py             # Google Drive integration
│   └── conversation_memory.py      # Chat history
│
└── data/
    ├── uploaded/                   # Uploaded documents
    ├── vector_db/                  # FAISS index
    └── gdrive_cache/               # Cached Drive files
```

---

## 🎯 Key Features Explained

### Retrieval-Augmented Generation (RAG)

1. **Document Ingestion**
   - Parse documents (PDF, Word, Excel, images)
   - Split into chunks (1000 chars, 200 overlap)
   - Generate embeddings
   - Store in FAISS vector database

2. **Question Answering**
   - Convert question to embedding
   - Search for similar document chunks
   - Build context from top-k results
   - Generate grounded response
   - Extract and display citations

3. **No Hallucination**
   - Only uses provided documents
   - Explicitly states when information unavailable
   - Shows source chunks for verification

---

## 🔒 Security & Privacy

- ✅ API keys stored in `.env` (gitignored)
- ✅ Local document processing
- ✅ No data retention beyond session
- ✅ Public Google Drive links only (no OAuth)
- ✅ All processing happens locally except LLM calls

---

## 🛠️ Customization

### Modify AI Behavior

Edit prompt files in `prompts/` folder:
- `system_prompt.txt` - Global AI behavior
- `user_prompt_template.txt` - Query template
- `report_section_system_prompt.txt` - Report writing style

### Adjust RAG Parameters

Edit `.env` file:
```env
CHUNK_SIZE=1000          # Size of document chunks
CHUNK_OVERLAP=200        # Overlap between chunks
TOP_K_RESULTS=5          # Number of chunks to retrieve
```

### Change LLM Model

Available Groq models:
- `llama-3.3-70b-versatile` (default, best accuracy)
- `llama-3.1-70b-versatile` (alternative)
- `llama-3.1-8b-instant` (faster, lower accuracy)

---

## 📚 Documentation

For detailed documentation, see the main README.md file included in the project.

Topics covered:
- Detailed usage instructions
- Troubleshooting guide
- API configuration
- Testing procedures
- FAQ section

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📊 Project Stats

- **Code**: 2,500+ lines of Python
- **Modules**: 11 core modules
- **Features**: 15+ major features
- **Dependencies**: 28 packages
- **Safety Layers**: 4 hallucination prevention mechanisms
- **Supported Formats**: 5 (PDF, Word, Excel, PNG, JPG)

---

## 🐛 Known Issues & Limitations

- OCR requires Tesseract installation for image processing
- Google Drive files must be publicly accessible
- Large documents may require increased memory
- Rate limits apply to Groq API (generous free tier)

---

## 🗺️ Roadmap

- [ ] Add support for more document formats (TXT, RTF, HTML)
- [ ] Implement user authentication system
- [ ] Add batch document processing
- [ ] Support for multiple languages
- [ ] Advanced visualization for graphs/charts
- [ ] Export to more formats (DOCX, HTML)
- [ ] Cloud deployment guides (AWS, Azure, GCP)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **Sentence Transformers** for embeddings
- **FAISS** for efficient vector search
- **Streamlit** for the amazing UI framework
- **ReportLab** for PDF generation

---

## 📧 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Documentation: See main README.md

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ for Healthcare Organizations**

Made with Python 🐍 | Powered by Groq 🚀 | Secured by RAG 🛡️

