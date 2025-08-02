# Feel Good Spas – Customer Service Analytics Platform

An AI-powered analytics solution that transforms raw customer service conversations (`feel-good-spas-vcons.json`) into actionable business insights for **Feel Good Spas** management.

This project demonstrates how raw, unstructured conversational data can be processed, enriched, visualized, and queried using AI to support better decision-making for non-technical managers.

---

## 🎯 **Assessment Deliverables**

### 1️⃣ Insight & Strategy Plan
**Deliverable:** Business insights document with strategic recommendations  
**Implementation:** 
- **Live Dashboard:** [https://feel-good-spas-analytics.streamlit.app](https://feel-good-spas-analytics.streamlit.app)
- **PDF Report Generation:** Click "Generate Insights Report (PDF)" button in dashboard
- **Key Insights:** Customer sentiment trends, agent performance metrics, call drivers, ROI projections

---

### 2️⃣ Transformed & Enriched Dataset
**Deliverable:** Clean, analysis-ready dataset with derived features  
**Files:**
- **Processing Script:** [`scripts/process_data.py`](./scripts/process_data.py)
- **Raw Data:** [`data/feel-good-spas-vcons.json`](./data/feel-good-spas-vcons.json)
- **Enriched Output:** [`data/processed_calls_enriched.csv`](./data/processed_calls_enriched.csv)

**Features Added:**
- Sentiment scores (AI-generated)
- Issue categorization (Booking, Rescheduling, Cancellation, etc.)
- Resolution status tracking
- Call duration analysis
- Agent performance metrics

---

### 3️⃣ Management Dashboard
**Deliverable:** User-centric visualization of critical business insights  
**Implementation:** [`app/dashboard.py`](./app/dashboard.py)

**Features:**
- **KPI Dashboard:** Resolution rate, sentiment scores, call volume
- **Interactive Charts:** Sentiment trends, agent performance, issue categories
- **Advanced Filtering:** By date, location, agent, sentiment
- **PDF Report Generation:** Executive insights with strategic recommendations
- **Real-time Analytics:** Live data visualization

👉 **Live Dashboard:** [https://feel-good-spas-analytics.streamlit.app](https://feel-good-spas-analytics.streamlit.app)

---

### 4️⃣ Conversational AI System
**Deliverable:** Usable web application for natural language analytics  
**Implementation:** [`app/chat.py`](./app/chat.py)

**Features:**
- **25+ Query Types:** Sentiment analysis, resolution rates, agent performance
- **Semantic Search:** Find similar conversations using embeddings
- **AI-Powered Insights:** GPT-4o-mini for complex analytical questions
- **Natural Language Interface:** Ask questions in plain English
- **Time-based Filtering:** "This month", "Last week", "Today"

👉 **Live Chat App:** [https://feel-good-spas-chat.streamlit.app](https://feel-good-spas-chat.streamlit.app)

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
# Set OpenAI API key (required for AI features)
export OPENAI_API_KEY="your-api-key-here"
```


### Process Data
```bash
# Generate enriched dataset from raw JSON
python scripts/process_data.py
```

### Run Applications
```bash
# Start Dashboard (Port 8501)
streamlit run app/dashboard.py

# Start Chat App (Port 8502)
streamlit run app/chat.py --server.port 8502
```


---

## 📁 Project Structure

```
clarity_voice_assessement/
├── data/
│   ├── feel-good-spas-vcons.json          # Raw input data
│   └── processed_calls_enriched.csv       # Enriched dataset
├── scripts/
│   └── process_data.py                    # Data processing script
├── app/
│   ├── dashboard.py                       # Management dashboard
│   └── chat.py                           # Conversational AI system
├── utils/
│   └── report_generator.py               # PDF report generation
├── requirements.txt                      # Python dependencies
└── README.md                            # Project documentation
```

---

## 🔧 Technical Features

- **AI Integration:** OpenAI GPT-4o-mini + text embeddings
- **Data Processing:** Pandas for data manipulation and analysis
- **Web Framework:** Streamlit for interactive applications
- **Visualization:** Plotly for interactive charts
- **PDF Generation:** ReportLab for professional reports
- **Error Handling:** Robust error management and validation
- **Caching:** Performance optimization with data caching

---

## 📊 Business Impact

**Current Metrics:**
- 63.2% resolution rate
- 0.14 average sentiment score
- 6.81 minutes average call duration
- 114 total calls analyzed

**Target Improvements:**
- 75% resolution rate target
- 0.35 sentiment score target
- 25% reduction in call volume
- 15-20% increase in customer retention

---

## 🚀 Deployment

**Streamlit Cloud:**
- Dashboard: [https://feel-good-spas-analytics.streamlit.app](https://feel-good-spas-analytics.streamlit.app)
- Chat App: [https://feel-good-spas-chat.streamlit.app](https://feel-good-spas-chat.streamlit.app)

