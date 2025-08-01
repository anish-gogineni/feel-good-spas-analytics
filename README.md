# Feel Good Spas Call Analysis Pipeline

A comprehensive data processing pipeline for analyzing customer service calls from Feel Good Spas using AI-powered sentiment analysis, categorization, and resolution tracking.

## 📋 Overview

This project processes vCon (Virtual Conversation) formatted call data and enriches it with AI-generated insights to help understand customer service performance, sentiment trends, and issue resolution patterns.

## 🏗️ Architecture

```
data/feel-good-spas-vcons.json
         ↓
   [Data Processing]
         ↓
data/processed_calls.csv
         ↓
   [AI Enrichment]
         ↓
data/processed_calls_enriched.csv
```

## 📊 Data Schema

### Input: vCon JSON Format
- Nested conversation records with metadata, dialogue, parties, and analysis

### Output: Clean CSV Format
| Column | Type | Description |
|--------|------|-------------|
| `call_id` | String | Unique call identifier |
| `subject` | String | Call topic/issue description |
| `call_created_at` | DateTime | Call creation timestamp |
| `call_duration` | Integer | Call duration in seconds |
| `location` | String | Spa location identifier |
| `call_type` | String | Call direction (inbound/outbound) |
| `agent_name` | String | Customer service agent name |
| `customer_name` | String | Customer name |
| `transcript` | Text | Complete conversation transcript |
| `summary` | Text | AI-generated call summary |
| `sentiment_score` | Float | Customer sentiment (-1 to 1) |
| `issue_category` | String | Issue classification (1-3 words) |
| `resolution_status` | String | resolved/unresolved/escalated |

## 🚀 Quick Start

### 1. Basic Data Processing

```bash
# Process vCon data into clean CSV
python scripts/process_data.py
```

This creates `data/processed_calls.csv` with extracted fields and placeholder columns for AI analysis.

### 2. AI Enrichment (Optional)

```bash
# Set up OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Run full pipeline with AI enrichment
python scripts/process_data.py
```

This creates both files:
- `data/processed_calls.csv` (basic processing)
- `data/processed_calls_enriched.csv` (with AI analysis)

### 3. Demo AI Analysis

```bash
# See AI enrichment demo without API costs
python scripts/demo_ai_enrichment.py
```

## 🔧 Features

### Data Processing
- ✅ **Robust extraction** from nested vCon JSON structure
- ✅ **Error handling** for missing or malformed fields
- ✅ **Data validation** and quality reporting
- ✅ **Modular design** with helper functions
- ✅ **Comprehensive logging** with progress tracking

### AI Enrichment
- ✅ **Sentiment analysis** using GPT-4o-mini
- ✅ **Issue categorization** with business-relevant labels
- ✅ **Resolution tracking** for performance metrics
- ✅ **Batch processing** with rate limiting (10 records/batch)
- ✅ **Error resilience** - continues on API failures
- ✅ **Progress tracking** with tqdm progress bars

## 📈 Sample Output

```
🤖 AI Enrichment Demo
==================================================

📞 Call 1: Customer Service Call - Booking Issue
Agent: Olivia Martinez
Customer: Ethan Williams
Duration: 403 seconds

🔍 AI Analysis:
  Sentiment Score: 0.7 (Very Positive)
  Issue Category: booking issue
  Resolution Status: resolved
```

## 🎯 Business Value

### Customer Service Insights
- **Sentiment Trends**: Track customer satisfaction over time
- **Issue Patterns**: Identify common problems and their frequency
- **Resolution Efficiency**: Measure first-call resolution rates
- **Agent Performance**: Compare resolution rates across agents

### Operational Analytics
- **Call Duration Analysis**: Understand handling time patterns
- **Location Performance**: Compare service quality across spas
- **Peak Hours**: Identify busy periods and staffing needs
- **Escalation Triggers**: Understand what leads to unresolved calls

## 📊 Data Quality Metrics

From processing 114 call records:
- **Data Completeness**: 96.5% (only 4 missing agent names)
- **Schema Compliance**: 100% (all required columns present)
- **Processing Success**: 100% (all records processed)
- **AI Enrichment**: Configurable based on API availability

## 🛠️ Technical Details

### Dependencies
- `pandas` - Data manipulation and CSV handling
- `openai` - AI analysis via GPT-4o-mini
- `tqdm` - Progress bar visualization
- `pathlib` - Modern file path handling

### Error Handling
- **Missing fields**: Default to empty strings or None
- **API failures**: Log warning, continue processing
- **Malformed JSON**: Skip record, log error
- **Rate limits**: Built-in delays between batches

### Performance Optimizations
- **Batch processing**: 10 records per batch to respect API limits
- **Content truncation**: Limit to 3000 chars to control token usage
- **Temperature setting**: 0.1 for consistent, focused analysis
- **Minimal token usage**: Structured JSON responses

## 🔒 Security & Privacy

- **API Key Management**: Environment variable only, never hardcoded
- **Data Processing**: All processing done locally
- **No Data Persistence**: OpenAI doesn't store API request data
- **Content Filtering**: Transcript content limited to prevent token overflow

## 🎭 Demo Mode

Run `python scripts/demo_ai_enrichment.py` to see AI analysis simulation without API costs. This demonstrates:
- Sentiment analysis logic
- Issue categorization patterns  
- Resolution status detection
- Output formatting

## 📝 Next Steps

### Enhanced Analytics
- Time-series sentiment analysis
- Agent performance dashboards
- Customer journey mapping
- Predictive escalation modeling

### Advanced Features
- Real-time call processing
- Integration with CRM systems
- Custom issue taxonomy
- Multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built for Feel Good Spas** - Transforming customer service data into actionable insights. 