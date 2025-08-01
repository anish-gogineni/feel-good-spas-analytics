#!/usr/bin/env python3
"""
Feel Good Spas - Customer Service Call Analytics Dashboard

A Streamlit dashboard for analyzing customer service call data with AI-generated insights.
Provides interactive visualizations and filters for business intelligence.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.report_generator import generate_report_pdf

# Configure page
st.set_page_config(
    page_title="Feel Good Spas - Call Analytics",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """
    Load and preprocess the enriched call data.
    
    Returns:
        pd.DataFrame: Processed call data with datetime conversions
    """
    try:
        df = pd.read_csv('data/processed_calls_enriched.csv')
        
        # Convert datetime column
        df['call_created_at'] = pd.to_datetime(df['call_created_at'])
        df['call_date'] = df['call_created_at'].dt.date
        
        # Convert duration to minutes
        df['call_duration_minutes'] = df['call_duration'] / 60
        
        # Clean and standardize data
        df['agent_name'] = df['agent_name'].fillna('Unknown Agent')
        df['location'] = df['location'].fillna('Unknown Location')
        
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Please ensure 'data/processed_calls_enriched.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

def render_kpis(df):
    """
    Render key performance indicators at the top of the dashboard.
    
    Args:
        df (pd.DataFrame): Filtered call data
    """
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate KPIs
    total_calls = len(df)
    avg_sentiment = df['sentiment_score'].mean()
    avg_duration = df['call_duration_minutes'].mean()
    resolved_rate = (df['resolution_status'] == 'resolved').mean() * 100
    
    with col1:
        st.metric(
            label="📞 Total Calls",
            value=f"{total_calls:,}",
            delta=None
        )
    
    with col2:
        sentiment_label = "😊" if avg_sentiment > 0 else "😐" if avg_sentiment == 0 else "😞"
        st.metric(
            label=f"{sentiment_label} Avg Sentiment",
            value=f"{avg_sentiment:.2f}",
            delta=f"{'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'}"
        )
    
    with col3:
        st.metric(
            label="⏱️ Avg Call Duration",
            value=f"{avg_duration:.1f} min",
            delta=None
        )
    
    with col4:
        st.metric(
            label="✅ Resolution Rate",
            value=f"{resolved_rate:.1f}%",
            delta=f"{'Good' if resolved_rate > 70 else 'Needs Improvement'}"
        )

def create_sentiment_trend_chart(df):
    """
    Create a line chart showing sentiment trend over time.
    
    Args:
        df (pd.DataFrame): Call data
        
    Returns:
        plotly.graph_objects.Figure: Sentiment trend chart
    """
    # Group by date and calculate average sentiment
    daily_sentiment = df.groupby('call_date')['sentiment_score'].agg(['mean', 'count']).reset_index()
    daily_sentiment.columns = ['date', 'avg_sentiment', 'call_count']
    
    fig = px.line(
        daily_sentiment,
        x='date',
        y='avg_sentiment',
        title='📈 Daily Sentiment Trend',
        labels={'avg_sentiment': 'Average Sentiment Score', 'date': 'Date'}
    )
    
    # Add horizontal line at 0 (neutral)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Neutral (0.0)")
    
    # Color the line based on sentiment
    fig.update_traces(line_color='green')
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Date",
        yaxis_title="Average Sentiment Score"
    )
    
    return fig

def create_issue_categories_chart(df):
    """
    Create a bar chart showing top 5 issue categories.
    
    Args:
        df (pd.DataFrame): Call data
        
    Returns:
        plotly.graph_objects.Figure: Issue categories chart
    """
    category_counts = df['issue_category'].value_counts().head(5)
    
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title='📋 Top 5 Issue Categories',
        labels={'x': 'Issue Category', 'y': 'Number of Calls'},
        color=category_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Issue Category",
        yaxis_title="Number of Calls"
    )
    
    return fig

def create_agent_performance_chart(df):
    """
    Create a bar chart showing agent performance by average sentiment.
    
    Args:
        df (pd.DataFrame): Call data
        
    Returns:
        plotly.graph_objects.Figure: Agent performance chart
    """
    # Calculate agent performance metrics
    agent_stats = df.groupby('agent_name').agg({
        'sentiment_score': 'mean',
        'call_id': 'count'
    }).reset_index()
    agent_stats.columns = ['agent_name', 'avg_sentiment', 'call_count']
    
    # Filter agents with at least 2 calls
    agent_stats = agent_stats[agent_stats['call_count'] >= 2]
    agent_stats = agent_stats.sort_values('avg_sentiment', ascending=False)
    
    # Color bars based on sentiment
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in agent_stats['avg_sentiment']]
    
    fig = px.bar(
        agent_stats,
        x='agent_name',
        y='avg_sentiment',
        title='👥 Agent Performance (Average Sentiment)',
        labels={'avg_sentiment': 'Average Sentiment Score', 'agent_name': 'Agent'},
        color='avg_sentiment',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Agent",
        yaxis_title="Average Sentiment Score",
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_location_chart(df):
    """
    Create a bar chart showing calls by location.
    
    Args:
        df (pd.DataFrame): Call data
        
    Returns:
        plotly.graph_objects.Figure: Location chart
    """
    location_counts = df['location'].value_counts()
    
    fig = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title='🏢 Calls by Location',
        labels={'x': 'Location', 'y': 'Number of Calls'},
        color=location_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Location",
        yaxis_title="Number of Calls"
    )
    
    return fig

def create_resolution_status_chart(df):
    """
    Create a pie chart showing resolution status distribution.
    
    Args:
        df (pd.DataFrame): Call data
        
    Returns:
        plotly.graph_objects.Figure: Resolution status chart
    """
    status_counts = df['resolution_status'].value_counts()
    
    colors = {
        'resolved': '#2E8B57',      # Sea Green
        'unresolved': '#DC143C',    # Crimson
        'escalated': '#FF8C00'      # Dark Orange
    }
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='🎯 Resolution Status Distribution',
        color=status_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_layout(height=400)
    
    return fig

def apply_filters(df):
    """
    Apply sidebar filters to the dataframe.
    
    Args:
        df (pd.DataFrame): Original call data
        
    Returns:
        pd.DataFrame: Filtered call data
    """
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    min_date = df['call_date'].min()
    max_date = df['call_date'].max()
    
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range if date_range else min_date
    
    # Location filter
    st.sidebar.subheader("🏢 Location")
    locations = ['All'] + sorted(df['location'].unique().tolist())
    selected_location = st.sidebar.selectbox("Select location:", locations)
    
    # Agent filter
    st.sidebar.subheader("👥 Agent")
    agents = ['All'] + sorted(df['agent_name'].unique().tolist())
    selected_agent = st.sidebar.selectbox("Select agent:", agents)
    
    # Apply filters
    filtered_df = df[
        (df['call_date'] >= start_date) & 
        (df['call_date'] <= end_date)
    ]
    
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['location'] == selected_location]
    
    if selected_agent != 'All':
        filtered_df = filtered_df[filtered_df['agent_name'] == selected_agent]
    
    # Show filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Results:** {len(filtered_df)} of {len(df)} calls")
    
    return filtered_df

def main():
    """
    Main function to render the Streamlit dashboard.
    """
    # Header
    st.title("📞 Feel Good Spas - Customer Service Analytics")
    st.markdown("*AI-Powered Call Analysis Dashboard*")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading call data..."):
        df = load_data()
    
    # Apply filters
    filtered_df = apply_filters(df)
    
    # Show data info
    st.markdown(f"**Data Period:** {df['call_date'].min()} to {df['call_date'].max()}")
    
    # Render KPIs
    st.subheader("📊 Key Performance Indicators")
    render_kpis(filtered_df)
    
    st.markdown("---")
    
    # Charts section
    st.subheader("📈 Analytics Dashboard")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_chart = create_sentiment_trend_chart(filtered_df)
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    with col2:
        issue_chart = create_issue_categories_chart(filtered_df)
        st.plotly_chart(issue_chart, use_container_width=True)
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        agent_chart = create_agent_performance_chart(filtered_df)
        st.plotly_chart(agent_chart, use_container_width=True)
    
    with col4:
        location_chart = create_location_chart(filtered_df)
        st.plotly_chart(location_chart, use_container_width=True)
    
    # Resolution status chart (full width)
    st.subheader("🎯 Resolution Analysis")
    resolution_chart = create_resolution_status_chart(filtered_df)
    st.plotly_chart(resolution_chart, use_container_width=True)
    
    # Data table section
    with st.expander("📋 Raw Data View"):
        st.subheader("Call Records")
        
        # Select columns to display
        display_columns = [
            'call_id', 'subject', 'agent_name', 'customer_name',
            'call_duration_minutes', 'sentiment_score', 'issue_category',
            'resolution_status', 'call_date'
        ]
        
        st.dataframe(
            filtered_df[display_columns].sort_values('call_date', ascending=False),
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"feel_good_spas_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # PDF Insights Report Section
    st.markdown("---")
    st.subheader("📋 Executive Insights Report")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Generate a comprehensive business insights report** containing:
        - Executive summary with key findings
        - Strategic recommendations and action items
        - KPI analysis with targets and status
        - ROI projections and business impact assessment
        - Next steps for management implementation
        """)
    
    with col2:
        st.markdown("") # Spacer
        st.markdown("") # Spacer
        
        # PDF Download Button
        if st.button("📄 Generate Insights Report (PDF)", type="primary", use_container_width=True):
            with st.spinner("Generating professional insights report..."):
                try:
                    # Generate PDF using the filtered data
                    pdf_content = generate_report_pdf(filtered_df)
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download FeelGoodSpas_Insights.pdf",
                        data=pdf_content,
                        file_name="FeelGoodSpas_Insights.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated successfully! Click the download button above.")
                    
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
                    st.info("💡 Tip: Ensure the reportlab package is installed: `pip install reportlab>=4.0.0`")

if __name__ == "__main__":
    main() 