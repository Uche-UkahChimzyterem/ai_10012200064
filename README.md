# 🤖 AcityPal – AI RAG Chatbot for Ghana Elections & Budget

GitHub Repository | Python | Streamlit | OpenAI

## 👤 Developer Details
**Developer:** Uche-Ukah Chimzyterem Janet  
**Roll Number:** 10012200064  
**GitHub Repository:** https://github.com/Uche-UkahChimzyterem/ai_10012200064.git  
**Institution:** Academic City University College

## ✨ Overview
AcityPal is a sophisticated Retrieval-Augmented Generation (RAG) chatbot system designed to answer questions about Ghana's election results and the 2025 Budget Statement & Economic Policy. The system combines custom embedding pipelines, hybrid retrieval strategies, and advanced prompt engineering to deliver accurate, context-aware responses while minimizing hallucination risks.

The platform features a modern Ghana-themed Streamlit interface with real-time query processing, similarity scoring, and comprehensive evaluation metrics.

## 📦 Repository Details
**Project Name:** AcityPal AI Chatbot  
**Stack:** Full-stack Python (Streamlit + Custom RAG Pipeline)  
**Data Sources:** Ghana Election Results (CSV), 2025 Budget Statement (PDF)  
**LLM Provider:** OpenAI GPT-4o-mini

## 🚀 Core Capabilities

### 🇬🇭 Ghana-Specific Features
- **Election Data Analysis:** Query presidential election results by region, constituency, and candidate
- **Budget Policy Insights:** Access detailed information from the 2025 Budget Statement & Economic Policy
- **Regional Filtering:** Filter queries by Ghanaian regions (Greater Accra, Ashanti, Western, etc.)
- **Multi-Source Integration:** Seamlessly combines structured CSV data with unstructured PDF documents
- **Localized Context:** Tailored for Ghana's political and economic landscape

### 🧠 AI & RAG Essentials
- **Custom Embedding Pipeline:** Hashing-based text embeddings optimized for retrieval
- **Hybrid Retrieval System:** Combines vector similarity with keyword-based search
- **Similarity Scoring:** Real-time relevance scoring for retrieved chunks
- **Hallucination Control:** Strict prompt engineering with abstention rules
- **Context Window Management:** Intelligent truncation to optimize LLM performance
- **Query Intent Routing:** Automatic classification (ELECTION / BUDGET / COMPARE)

### 📊 Advanced Features
- **Chunking Strategy Comparison:** Evaluates different text chunking approaches
- **Failure Case Analysis:** Identifies and addresses retrieval failures
- **Adversarial Evaluation:** Tests system robustness against challenging queries
- **RAG vs No-Retrieval Comparison:** Benchmarking framework
- **Stage-by-Stage Logging:** Complete pipeline transparency
- **Interactive Visualizations:** Vote bar charts and structured response cards

### 💡 Planned Enhancements
- Multi-turn conversation memory with context preservation
- Real-time data streaming for live election updates
- Voice query input and text-to-speech responses
- Advanced analytics dashboard for query patterns
- Export functionality for reports and insights

## 🏗️ Application Structure
```
AI-Exam-B/
├── src/                       # Core RAG pipeline modules
│   ├── data_prep.py           # Data cleaning + chunking strategies
│   ├── embedding.py           # Custom hashing-based embeddings
│   ├── retrieval.py           # Custom vector store + hybrid scoring
│   ├── prompting.py           # Prompt templates + context management
│   ├── pipeline.py            # RAG orchestration + logging
│   ├── evaluation.py          # Benchmark + adversarial tests
│   ├── llm.py                 # LLM integration (OpenAI)
│   ├── memory.py              # Conversation memory management
│   ├── config.py              # Configuration management
│   ├── architecture.py        # System architecture definitions
│   ├── build_index.py         # Vector index construction
│   └── utils.py               # Utility functions
├── docs/                      # Documentation
│   ├── architecture.md        # Architecture overview
│   ├── manual_experiment_logs.md  # Experiment logs
│   ├── rubric_report.md       # Exam rubric compliance
│   └── video_walkthrough_script.md  # Demo script
├── logs/                      # Pipeline logs
│   ├── feedback.jsonl         # User feedback logs
│   └── pipeline_logs.jsonl    # Execution logs
├── outputs/                   # Generated data
│   ├── cleaned_election_data.csv   # Processed election data
│   ├── chunks_strategy_a.jsonl     # Chunking strategy A output
│   ├── chunks_strategy_b.jsonl     # Chunking strategy B output
│   ├── chunking_comparison.json    # Chunking comparison results
│   └── vector_index.pkl             # Serialized vector index
├── images/                    # Static assets
│   └── acitylogo.png          # Application logo
├── .streamlit/                # Streamlit configuration
│   ├── config.toml            # Streamlit settings
│   └── auto-scroll.js         # Custom JavaScript
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── Ghana_Election_Result.csv  # Raw election data
└── 2025-Budget-Statement-and-Economic-Policy_v4.pdf  # Budget document
```

## 🧰 Technology Stack

### Backend
- **Framework:** Streamlit (Python)
- **LLM Provider:** OpenAI GPT-4o-mini
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** scikit-learn
- **PDF Processing:** PyPDF
- **Environment Management:** python-dotenv

### RAG Pipeline
- **Embedding Strategy:** Custom hashing-based text embeddings
- **Vector Store:** Custom implementation with similarity search
- **Retrieval Method:** Hybrid (vector + keyword-based)
- **Prompt Engineering:** Custom templates with context window management
- **Evaluation:** Adversarial testing + failure case analysis

### Frontend
- **UI Framework:** Streamlit
- **Styling:** Custom CSS with Ghana flag color scheme
- **Interactive Components:** Charts, expanders, chat interface
- **State Management:** Streamlit session state
- **Dark Mode:** Toggle support

### Data Sources
- **Election Data:** CSV format with presidential results by constituency
- **Budget Document:** PDF format with 2025 Budget Statement & Economic Policy

## ⚙️ Installation & Setup

### Requirements
- Python 3.8+
- pip
- OpenAI API Key
- Git

### Clone the Repository
```bash
git clone https://github.com/Uche-UkahChimzyterem/ai_10012200064.git
cd AI-Exam-B
```

### Configure the Project

**Install dependencies:**
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

**Environment Variables**
Create a `.env` file:
```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
```

⚠️ **Do not commit `.env` files. Use `.env.example` as template.**

### Build the Knowledge Base

**Step 1: Prepare and clean data**
```bash
python -m src.data_prep
```

**Step 2: Build vector index**
```bash
python -m src.build_index
```

**Step 3: Run evaluation (optional)**
```bash
python -m src.evaluation
```

### Run the Application

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

## 🔌 System Features

### Query Processing
- **Intent Classification:** Automatically routes queries to appropriate data sources
- **Region Filtering:** Filter results by Ghanaian regions
- **Hybrid Search:** Combines semantic and keyword-based retrieval
- **Similarity Scoring:** Real-time relevance metrics for each retrieved chunk

### Chat Interface
- **Conversation History:** Track and manage multiple chat sessions
- **Pin & Export:** Save important conversations and export them
- **Dark/Light Mode:** Toggle between themes
- **Auto-scroll:** Smooth chat experience with automatic scrolling
- **RAG Details:** Expandable cards showing retrieval details and source chunks

### Evaluation & Testing
- **Chunking Comparison:** Compare different text chunking strategies
- **Adversarial Queries:** Test system with challenging questions
- **RAG vs LLM Comparison:** Benchmark retrieval-augmented vs pure LLM responses
- **Failure Case Analysis:** Identify and fix retrieval weaknesses

## 🧪 Testing

### Manual Testing
Use the Streamlit interface to test various query types:
- Election results by region
- Budget policy questions
- Comparative queries
- Adversarial questions

### Evaluation Script
Run the automated evaluation suite:
```bash
python -m src.evaluation
```

This generates:
- Chunking comparison metrics
- RAG vs no-retrieval performance
- Failure case analysis
- Adversarial query results

## 🚀 Deployment Notes

**For Local Deployment:**
- Ensure Python 3.8+ is installed
- Set up virtual environment
- Configure OpenAI API key
- Run with Streamlit

**For Cloud Deployment (Streamlit Cloud, Heroku, etc.):**
- Set `OPENAI_API_KEY` in environment variables
- Ensure all dependencies are in `requirements.txt`
- Configure Streamlit settings in `.streamlit/config.toml`
- Enable HTTPS for production
- Set appropriate resource limits

**Performance Optimization:**
- Use cached vector index for faster startup
- Adjust `top_k` parameter based on query complexity
- Enable hybrid search for better accuracy
- Monitor token usage for cost management

## 📊 Data Models

### Core Entities
- **Document:** Processed text chunks from PDF and CSV sources
- **Chunk:** Text segments with metadata (source, region, page, etc.)
- **Embedding:** Vector representations of text chunks
- **Query:** User input with intent classification
- **Retrieval Result:** Ranked list of relevant chunks with similarity scores
- **Conversation:** Chat session with message history
- **Evaluation Metric:** Performance metrics for system assessment

### Data Sources
- **Election Data:** Presidential results by constituency and region
- **Budget Document:** 2025 Budget Statement & Economic Policy sections

## 🤝 Contributions

This is an academic project for examination purposes. For suggestions or improvements:
1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Add tests where applicable
5. Open a pull request

## 📚 Documentation

- **Architecture Overview:** See `docs/architecture.md`
- **Experiment Logs:** See `docs/manual_experiment_logs.md`
- **Rubric Report:** See `docs/rubric_report.md`
- **Video Walkthrough:** See `docs/video_walkthrough_script.md`

## 🔐 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#CE1126"
backgroundColor = "#f0f4f8"
secondaryBackgroundColor = "#ffffff"
textColor = "#111827"
font = "sans serif"
```

### Retrieval Settings
- **top_k:** Number of chunks to retrieve (default: 5)
- **use_hybrid:** Enable hybrid search (default: True)
- **region_filter:** Filter by Ghanaian region (default: "All Regions")

### Prompt Variants
- **hybrid:** Balanced approach with context and instructions
- **strict:** Conservative prompts with strong hallucination control
- **creative:** More flexible prompts for exploratory queries

## 📈 Performance Metrics

The system includes comprehensive evaluation:
- **Retrieval Accuracy:** Precision, Recall, F1-score
- **Response Quality:** Hallucination rate, relevance scoring
- **Latency:** Query processing time
- **Token Efficiency:** Average tokens per query

## 🎯 Use Cases

### Election Queries
- "Who won the presidential election in Greater Accra?"
- "What were the vote counts for NPP in Ashanti Region?"
- "Compare election results between Northern and Upper East regions"

### Budget Queries
- "What is the allocation for infrastructure in the 2025 budget?"
- "How much revenue is projected from taxation?"
- "What are the key priorities in the fiscal policy?"

### Comparative Queries
- "Compare the vote shares of NDC and NPP nationwide"
- "How does the 2025 budget compare to previous years on education spending?"

## 🏆 Exam Compliance

This project aligns with academic examination requirements:
- ✅ Data cleaning and preprocessing
- ✅ Chunking strategy comparison
- ✅ Custom embedding pipeline
- ✅ Custom vector store implementation
- ✅ Hybrid retrieval with similarity scoring
- ✅ Prompt engineering with hallucination control
- ✅ Full pipeline with logging
- ✅ Adversarial evaluation
- ✅ RAG vs no-retrieval comparison
- ✅ Comprehensive documentation

---

**AcityPal – Empowering Ghana's Democratic & Economic Discourse with AI** 🇬🇭

**Developed by:** Uche-Ukah Chimzyterem (Roll Number: 10012200064)  
**Institution:** Academic City University College
