# Feel Good Spas - Customer Service Analytics Dashboard

## 📊 Dashboard Overview

An interactive Streamlit dashboard for analyzing customer service call data with AI-generated insights. This dashboard provides comprehensive business intelligence tools for understanding customer sentiment, identifying trends, and tracking performance metrics.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Running the Dashboard
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/dashboard.py

# Access at http://localhost:8501
```

## 📈 Features

### 🎯 Key Performance Indicators (KPIs)
- **Total Calls**: Count of processed calls
- **Average Sentiment**: AI-generated sentiment score (-1 to 1)
- **Average Call Duration**: Mean call time in minutes
- **Resolution Rate**: Percentage of resolved calls

### 📊 Interactive Charts

#### 1. Sentiment Trend Over Time
- **Type**: Line chart
- **Purpose**: Track customer satisfaction trends
- **Features**: Daily sentiment averages with neutral baseline
- **Insights**: Identify patterns in customer mood

#### 2. Top 5 Issue Categories
- **Type**: Bar chart
- **Purpose**: Understand most common customer issues
- **Features**: Color-coded by frequency
- **Insights**: Prioritize process improvements

#### 3. Agent Performance Analysis
- **Type**: Bar chart
- **Purpose**: Compare agent effectiveness by sentiment
- **Features**: Color-coded performance (green=positive, red=negative)
- **Insights**: Identify top performers and training needs

#### 4. Calls by Location
- **Type**: Bar chart
- **Purpose**: Analyze call volume by spa location
- **Features**: Viridis color scale
- **Insights**: Resource allocation and capacity planning

#### 5. Resolution Status Distribution
- **Type**: Pie chart
- **Purpose**: Visual breakdown of call outcomes
- **Features**: Color-coded status (green=resolved, red=unresolved, orange=escalated)
- **Insights**: Service quality assessment

### 🔍 Interactive Filters

#### Date Range Filter
- **Location**: Sidebar
- **Function**: Filter all data by date range
- **Default**: Full data range
- **Use Case**: Focus on specific time periods

#### Location Filter
- **Location**: Sidebar
- **Options**: All locations + individual spa locations
- **Function**: Filter data by specific spa location
- **Use Case**: Location-specific analysis

#### Agent Filter
- **Location**: Sidebar
- **Options**: All agents + individual agents
- **Function**: Filter data by specific customer service agent
- **Use Case**: Agent performance evaluation

### 📋 Raw Data View

#### Expandable Data Table
- **Features**: 
  - Sortable columns
  - Filtered data display
  - Call details view
- **Columns**: Call ID, Subject, Agent, Customer, Duration, Sentiment, Category, Status, Date

#### Data Export
- **Format**: CSV download
- **Content**: Filtered dataset
- **Naming**: Timestamped filename
- **Use Case**: Further analysis in Excel/other tools

## 🎨 Design Features

### Layout
- **Wide layout**: Maximizes chart visibility
- **Responsive design**: Adapts to screen size
- **Clean interface**: Professional appearance

### Color Schemes
- **Sentiment**: Green (positive) to Red (negative)
- **Performance**: Color-coded metrics
- **Categories**: Distinct colors for easy identification
- **Status**: Intuitive color mapping (green=good, red=bad)

### User Experience
- **Loading indicators**: Smooth data loading
- **Interactive tooltips**: Detailed information on hover
- **Mobile friendly**: Responsive design
- **Fast performance**: Cached data loading

## 📊 Dashboard Sections

### Header Section
- Title and branding
- Data period information
- Loading indicators

### KPI Section
- 4-column metric display
- Visual indicators and deltas
- Emoji-enhanced labels

### Charts Section
- 2x2 grid layout for main charts
- Full-width resolution analysis
- Consistent styling

### Data Section
- Expandable raw data view
- Export functionality
- Filtering applied

## 🔧 Technical Details

### Data Processing
- **Cache**: Streamlit caching for performance
- **Date handling**: Automatic datetime conversion
- **Data cleaning**: Missing value handling
- **Calculations**: Real-time metric computation

### Chart Libraries
- **Plotly Express**: Interactive charts
- **Plotly Graph Objects**: Custom styling
- **Responsive design**: Container-width charts

### Performance Optimizations
- **Data caching**: `@st.cache_data` decorator
- **Lazy loading**: Charts rendered on demand
- **Efficient filtering**: Pandas operations
- **Memory management**: Optimized data structures

## 🎯 Business Use Cases

### Daily Operations
- Monitor real-time call sentiment
- Track resolution rates
- Identify capacity issues

### Performance Management
- Compare agent effectiveness
- Identify training opportunities
- Track improvement trends

### Strategic Planning
- Analyze location performance
- Understand customer issues
- Resource allocation decisions

### Quality Assurance
- Sentiment trend monitoring
- Issue category analysis
- Resolution tracking

## 🔍 Analytics Insights

### What You Can Learn
1. **Customer Satisfaction Trends**: Daily sentiment patterns
2. **Common Issues**: Most frequent customer problems
3. **Agent Performance**: Who resolves issues most effectively
4. **Location Analytics**: Which spas need attention
5. **Resolution Efficiency**: How well issues are resolved

### Actionable Metrics
- Identify negative sentiment spikes
- Find top-performing agents for best practices
- Discover common issues for process improvement
- Track resolution rate improvements
- Monitor location-specific challenges

## 🛠️ Troubleshooting

### Common Issues
1. **Data file not found**: Ensure `data/processed_calls_enriched.csv` exists
2. **Package errors**: Install requirements with `pip install -r requirements.txt`
3. **Port conflicts**: Streamlit runs on port 8501 by default
4. **Memory issues**: Large datasets may require more RAM

### Dashboard Not Loading
1. Check if data file exists in correct location
2. Verify all packages are installed
3. Check terminal for error messages
4. Restart Streamlit server

## 📱 Mobile Support

The dashboard is optimized for desktop use but includes mobile-responsive features:
- Collapsible sidebar on mobile
- Responsive chart sizing
- Touch-friendly interactions
- Optimized loading on slower connections

---

**Built with ❤️ for Feel Good Spas - Transforming customer service data into actionable insights.** 