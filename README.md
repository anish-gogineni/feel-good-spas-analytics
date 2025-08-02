# Feel Good Spas – Customer Insights Platform

An AI-powered analytics solution that transforms raw customer service conversations (`feel-good-spas-vcons.json`) into actionable business insights for **Feel Good Spas** management.

This project demonstrates how raw, unstructured conversational data can be processed, enriched, visualized, and queried using AI to support better decision-making for non-technical managers.

---

## 🎯 **Project Deliverables**

### 1️⃣ Insight & Strategy Plan
Managers can view **key business insights** (sentiment trends, agent performance, issue patterns, ROI projections) and **download a detailed PDF report** directly from the dashboard.  
👉 **Live Dashboard:** [https://feel-good-spas-analytics.streamlit.app](https://feel-good-spas-analytics.streamlit.app)

---

### 2️⃣ Transformed & Enriched Dataset
📜 **Script:** [`scripts/process_data.py`](./scripts/process_data.py)  
📄 **Output:** `data/processed_calls_enriched.csv`  

- Converts raw `feel-good-spas-vcons.json` → Clean, enriched CSV  
- Adds sentiment scores, issue categorization, and resolution status  

---

### 3️⃣ Management Dashboard
📊 **Interactive Streamlit Dashboard**  
- Displays KPIs, sentiment trends, agent performance, and issue patterns  
- Includes **PDF report download** with actionable insights  

👉 **Live App:** [https://feel-good-spas-analytics.streamlit.app](https://feel-good-spas-analytics.streamlit.app)

---

### 4️⃣ Conversational AI Assistant
🤖 **AI-powered chat app** for natural language analytics  
- Supports 25+ query types (sentiment, resolution rate, agent performance, etc.)  
- Uses semantic search with embeddings + GPT-powered insights  

👉 **Live Chat App:** [https://feel-good-spas-chat.streamlit.app](https://feel-good-spas-chat.streamlit.app)

---

## 🚀 Quick Start (Local)

```bash
# Clone the repo
git clone <repo-url>
cd feel-good-spas-analytics

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/dashboard.py

# Run chat app
streamlit run app/chat.py --server.port 8502
