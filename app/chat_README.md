# Feel Good Spas - AI Chat Assistant

## 🤖 Overview

An intelligent chat interface that combines semantic search with AI-powered analysis to answer questions about customer service call data. The app uses OpenAI's text-embedding-3-small model for semantic search and GPT-4o-mini for generating comprehensive answers.

## 🚀 Quick Start

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Run the chat app
streamlit run app/chat.py

# Access at http://localhost:8501
```

## ✨ Key Features

### 🧠 Intelligent Query Routing
- **KPI Queries**: Automatically detects numerical questions and runs Pandas calculations
- **Open-ended Questions**: Uses semantic search + GPT analysis for complex questions

### 🔍 Semantic Search
- **Embeddings**: Uses OpenAI's text-embedding-3-small model
- **Caching**: Embeddings are cached locally for fast performance
- **Similarity**: Cosine similarity matching with configurable thresholds
- **Context**: Combines transcripts and summaries for richer search

### 💬 Chat Interface
- **Streamlit Chat**: Native chat UI with `st.chat_input()` and `st.chat_message()`
- **Message History**: Persistent conversation state
- **Source Citations**: Shows relevant call records that informed each answer
- **Visual Charts**: Sentiment analysis charts for search results

## 🎯 Query Types

### KPI Queries (Automatically Detected)
These queries trigger direct Pandas calculations:

- **Call Duration**: "What's the average call duration?"
- **Call Volume**: "How many total calls do we have?"
- **Sentiment Analysis**: "What's the average sentiment score?"
- **Resolution Rates**: "What percentage of calls are resolved?"
- **Issue Categories**: "What are the most common issues?"
- **Agent Performance**: "Which agents have the best sentiment scores?"

### Open-ended Queries (Semantic Search + AI)
These queries use semantic search to find relevant calls, then GPT analysis:

- **Customer Complaints**: "What do customers complain about most?"
- **Issue Patterns**: "What are the main booking problems?"
- **Customer Sentiment**: "What makes customers frustrated?"
- **Resolution Analysis**: "Why are some calls unresolved?"
- **Service Quality**: "How can we improve customer service?"

## 🔧 Technical Architecture

### Data Processing Flow
```
CSV Data → Pandas DataFrame → Text Cleaning → OpenAI Embeddings → Vector Storage
```

### Query Processing Flow
```
User Query → KPI Detection → Route to:
├── Pandas Calculations (KPIs)
└── Semantic Search → GPT Analysis (Open-ended)
```

### Embedding Strategy
- **Model**: text-embedding-3-small (cost-effective, high-quality)
- **Input**: Combined transcript + summary for rich context
- **Caching**: Pickle storage for fast reload
- **Truncation**: 8000 character limit to avoid token issues

## 📊 Response Types

### KPI Responses
- Direct statistical answers with formatting
- Bullet points for multi-item results (top categories, agent rankings)
- Percentage calculations with counts

### Semantic Responses
- GPT-4o-mini generated analysis
- Source call citations with similarity scores
- Sentiment charts for visual context
- Expandable source details

## 🎨 User Interface

### Main Chat Area
- **Welcome Message**: Explains capabilities and gives examples
- **Message History**: Scrollable conversation with user/assistant messages
- **Charts**: Embedded Plotly visualizations
- **Source Expansion**: Click to view detailed call information

### Sidebar Features
- **Data Overview**: Key metrics (total calls, avg sentiment, resolution rate)
- **Example Queries**: Clickable buttons for common questions
- **Embedding Status**: Progress and caching information

## 🔍 Search Quality

### Semantic Similarity
- **Threshold**: 0.1 minimum similarity (filters irrelevant results)
- **Top-K**: Returns 5 most relevant calls by default
- **Ranking**: Sorted by cosine similarity score

### Context Quality
- **Rich Input**: Transcript + summary provides comprehensive context
- **Length Management**: Smart truncation preserves important content
- **Metadata**: Includes sentiment, category, resolution status

## 💡 Example Interactions

### KPI Query Example
```
User: "What's the average call duration?"
Assistant: "The average call duration is 8.45 minutes (507 seconds)."
```

### Semantic Query Example
```
User: "What do customers complain about most?"
Assistant: "Based on the call data, customers most frequently complain about:

1. **Booking System Issues** - Many customers report problems with online booking, including:
   - Appointments disappearing from their accounts
   - System glitches preventing changes
   - Confirmation emails not being sent

2. **Cancellation Policies** - Customers express frustration with:
   - Strict 72-hour cancellation windows
   - No exceptions for emergencies
   - Loss of payment for unavoidable cancellations

[Shows 5 relevant call examples with sentiment chart]
```

## 🛠️ Customization Options

### Embedding Parameters
- **Model**: Change `EMBEDDING_MODEL` constant for different models
- **Cache Location**: Modify `EMBEDDINGS_FILE` path
- **Similarity Threshold**: Adjust minimum similarity in `semantic_search()`

### Chat Model Settings
- **Model**: Change `CHAT_MODEL` for different GPT models
- **Temperature**: Modify for more/less creative responses
- **Max Tokens**: Adjust response length limits

### UI Customization
- **Page Config**: Modify title, icon, layout in `st.set_page_config()`
- **Example Queries**: Update sidebar examples
- **Welcome Message**: Customize initial assistant message

## 🚨 Troubleshooting

### Common Issues
1. **API Key Error**: Ensure `OPENAI_API_KEY` environment variable is set
2. **Embedding Computation**: First run takes time to compute embeddings for all calls
3. **Memory Usage**: Large datasets may require more RAM for embeddings
4. **Token Limits**: Very long transcripts are automatically truncated

### Performance Optimization
- **Embedding Cache**: Embeddings are cached after first computation
- **Streamlit Cache**: Data loading is cached with `@st.cache_data`
- **Batch Processing**: Could implement batch embedding for very large datasets

### Cost Management
- **Embedding Model**: Uses cost-effective text-embedding-3-small
- **Chat Model**: Uses efficient GPT-4o-mini
- **Caching**: Prevents repeated embedding API calls
- **Truncation**: Limits token usage for long content

## 📈 Analytics & Monitoring

### Usage Patterns
- Monitor which query types are most common
- Track semantic search result quality
- Analyze user satisfaction with responses

### Cost Tracking
- Embedding API calls (one-time per dataset)
- Chat completion tokens (per query)
- Total API usage and costs

## 🔄 Data Updates

### Adding New Calls
1. Update the CSV file with new call records
2. Delete the embeddings cache file (`data/call_embeddings.pkl`)
3. Restart the app to recompute embeddings

### Schema Changes
- Modify data loading functions if CSV schema changes
- Update embedding text combination strategy if needed
- Adjust KPI calculations for new metrics

---

**Ready to explore your customer service data with AI! 🚀** 