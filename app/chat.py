#!/usr/bin/env python3
"""
Feel Good Spas - Conversational AI Assistant

A natural language interface for managers to query customer service call data.
Uses OpenAI embeddings for semantic search and GPT-4o-mini for intelligent responses.
"""

import streamlit as st
import pandas as pd
import numpy as np
import openai
import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, timedelta

# Configure Streamlit page
st.set_page_config(
    page_title="Feel Good Spas - AI Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY') or 'your-api-key-here'

if not openai.api_key:
    st.error("❌ OpenAI API key not configured!")
    st.stop()

# Constants
EMBEDDINGS_CACHE_FILE = "data/embeddings_cache.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and preprocess the enriched call data.
    
    Returns:
        pd.DataFrame: Cleaned and processed call data
    """
    try:
        df = pd.read_csv('data/processed_calls_enriched.csv')
        
        # Convert datetime and clean data
        df['call_created_at'] = pd.to_datetime(df['call_created_at'])
        df['call_date'] = df['call_created_at'].dt.date
        df['call_duration_minutes'] = df['call_duration'] / 60
        
        # Clean missing values
        df['agent_name'] = df['agent_name'].fillna('Unknown Agent')
        df['customer_name'] = df['customer_name'].fillna('Unknown Customer')
        df['location'] = df['location'].fillna('Unknown Location')
        df['transcript'] = df['transcript'].fillna('')
        df['summary'] = df['summary'].fillna('')
        df['issue_category'] = df['issue_category'].fillna('Other')
        df['resolution_status'] = df['resolution_status'].fillna('unknown')
        
        # Ensure numeric columns
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0)
        df['call_duration'] = pd.to_numeric(df['call_duration'], errors='coerce').fillna(0)
        
        st.sidebar.success(f"✅ Loaded {len(df)} call records")
        return df
        
    except FileNotFoundError:
        st.error("❌ Data file not found! Please ensure 'data/processed_calls_enriched.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

def get_embedding(text: str, max_retries: int = 3) -> Optional[List[float]]:
    """
    Get embedding for text using OpenAI's embedding model.
    
    Args:
        text (str): Text to embed
        max_retries (int): Maximum retry attempts
        
    Returns:
        Optional[List[float]]: Embedding vector or None if failed
    """
    if not text or not text.strip():
        return None
        
    # Clean and truncate text
    text = text.replace("\n", " ").strip()
    if len(text) > 8000:  # Conservative token limit
        text = text[:8000] + "..."
    
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding
            
            if embedding and len(embedding) > 0:
                return embedding
            else:
                st.warning(f"Empty embedding received (attempt {attempt + 1})")
                
        except Exception as e:
            st.warning(f"Embedding error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
    
    return None

@st.cache_data
def compute_embeddings(df: pd.DataFrame) -> Tuple[Dict[int, List[float]], List[int]]:
    """
    Compute embeddings for all transcripts with caching.
    
    Args:
        df (pd.DataFrame): Call data
        
    Returns:
        Tuple[Dict[int, List[float]], List[int]]: Embeddings dict and valid indices
    """
    cache_path = Path(EMBEDDINGS_CACHE_FILE)
    
    # Try to load cached embeddings
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data.get('embeddings', {})) == len(df):
                    st.sidebar.success("✅ Loaded cached embeddings")
                    return cached_data['embeddings'], cached_data['valid_indices']
        except Exception as e:
            st.sidebar.warning(f"Cache loading failed: {e}")
    
    # Compute new embeddings
    st.sidebar.info("🔄 Computing embeddings...")
    embeddings = {}
    valid_indices = []
    
    progress_bar = st.sidebar.progress(0)
    
    for i, row in df.iterrows():
        # Combine transcript and summary
        transcript = str(row['transcript']) if pd.notna(row['transcript']) else ""
        summary = str(row['summary']) if pd.notna(row['summary']) else ""
        
        if not transcript.strip() and not summary.strip():
            progress_bar.progress((i + 1) / len(df))
            continue
            
        # Create rich context for embedding
        context = f"""
        Call Subject: {row['subject']}
        Agent: {row['agent_name']}
        Location: {row['location']}
        Issue Category: {row['issue_category']}
        Summary: {summary}
        Transcript: {transcript}
        """.strip()
        
        embedding = get_embedding(context)
        
        if embedding:
            embeddings[i] = embedding
            valid_indices.append(i)
        
        progress_bar.progress((i + 1) / len(df))
    
    st.sidebar.success(f"✅ Generated {len(embeddings)} embeddings")
    
    # Cache embeddings
    try:
        os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_FILE), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'valid_indices': valid_indices
            }, f)
        st.sidebar.info("💾 Embeddings cached for future use")
    except Exception as e:
        st.sidebar.warning(f"Caching failed: {e}")
    
    return embeddings, valid_indices

def classify_query(query: str) -> str:
    """
    Classify user query as numerical/aggregated or open-ended.
    
    Args:
        query (str): User query
        
    Returns:
        str: 'numerical' or 'open_ended'
    """
    query_lower = query.lower()
    
    # First check for complex queries that should be open-ended
    complex_keywords = [
        'complaints', 'complain', 'topics', 'themes', 'issues about',
        'problems with', 'concerns about', 'feedback about', 'mentions',
        'frequently mentioned', 'common themes', 'what do customers',
        'how do customers feel', 'why do customers', 'what makes customers',
        'quality complaints', 'service quality', 'booking issues',
        'frustration', 'upset', 'angry', 'dissatisfied'
    ]
    
    # If query contains complex keywords, treat as open-ended
    if any(keyword in query_lower for keyword in complex_keywords):
        return 'open_ended'
    
    # IMPORTANT: Category classification queries should use AI for intelligent analysis
    # Instead of exact keyword matching, let AI understand the context and patterns
    if 'how many' in query_lower and 'calls' in query_lower:
        # These patterns should use AI for intelligent category analysis:
        ai_patterns = [
            'classified as', 'categorized as', 'tagged as',
            'related', '-related', 'regarding', 'about',
            'billing', 'booking', 'critical', 'urgent', 'complaint'
        ]
        if any(pattern in query_lower for pattern in ai_patterns):
            return 'open_ended'
    
    # Simple numerical queries
    simple_numerical = [
        'average', 'mean', 'total calls', 'count of', 'how many calls',
        'number of calls', 'percentage of', 'resolution rate',
        'duration', 'time', 'score', 'statistics', 'stats'
    ]
    
    # Specific agent/location counting (simple numerical)
    if ('calls' in query_lower and 
        ('by agent' in query_lower or 'by location' in query_lower or 
         any(name in query_lower for name in ['ethan', 'david', 'williams', 'martinez']))):
        return 'numerical'
    
    # Check for simple numerical indicators
    if any(keyword in query_lower for keyword in simple_numerical):
        return 'numerical'
    
    # Simple numerical patterns
    numerical_patterns = [
        r'\bhow many calls\b',
        r'\baverage call duration\b',
        r'\btotal calls\b',
        r'\bresolution rate\b'
    ]
    
    if any(re.search(pattern, query_lower) for pattern in numerical_patterns):
        return 'numerical'
    
    return 'open_ended'

def semantic_search(query: str, embeddings: Dict[int, List[float]], df: pd.DataFrame, top_k: int = 3) -> List[Dict]:
    """
    Perform semantic search to find relevant calls.
    
    Args:
        query (str): Search query
        embeddings (Dict[int, List[float]]): Precomputed embeddings
        df (pd.DataFrame): Call data
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: Top relevant calls with metadata
    """
    if not embeddings:
        return []
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    # Calculate similarities
    similarities = []
    indices = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[i] for i in indices])
    
    if embedding_matrix.size == 0:
        return []
    
    query_vector = np.array(query_embedding).reshape(1, -1)
    
    try:
        similarity_scores = cosine_similarity(query_vector, embedding_matrix)[0]
    except Exception as e:
        st.error(f"Similarity calculation error: {e}")
        return []
    
    # Get top results
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarity_scores[idx] > 0.1:  # Minimum similarity threshold
            df_idx = indices[idx]
            row = df.iloc[df_idx]
            
            results.append({
                'similarity': similarity_scores[idx],
                'call_id': row['call_id'],
                'subject': row['subject'],
                'agent_name': row['agent_name'],
                'customer_name': row['customer_name'],
                'location': row['location'],
                'transcript': row['transcript'],
                'summary': row['summary'],
                'sentiment_score': row['sentiment_score'],
                'issue_category': row['issue_category'],
                'resolution_status': row['resolution_status'],
                'call_date': row['call_date']
            })
    
    return results

def answer_numerical_query(query: str, df: pd.DataFrame) -> Tuple[str, Optional[go.Figure]]:
    """
    Answer numerical/aggregated queries using pandas operations.
    
    Args:
        query (str): User query
        df (pd.DataFrame): Call data
        
    Returns:
        Tuple[str, Optional[go.Figure]]: Answer text and optional chart
    """
    import re
    query_lower = query.lower()
    
    try:
        # Average call duration
        if 'average' in query_lower and 'duration' in query_lower:
            avg_duration = df['call_duration_minutes'].mean()
            return f"The average call duration is **{avg_duration:.2f} minutes** ({avg_duration*60:.0f} seconds).", None
        
        # Specific category/classification queries (with "classified" keyword)
        elif 'how many' in query_lower and 'calls' in query_lower and ('classified' in query_lower or 'categorized' in query_lower or 'tagged' in query_lower):
            # Extract ANY potential category word from the query
            # Look for potential category word after "as" or before "calls"
            category_match = re.search(r'(?:as|classified as)\s+([a-zA-Z-]+)', query_lower)
            if not category_match:
                category_match = re.search(r'([a-zA-Z-]+)(?:-related)?\s+calls', query_lower)
            
            potential_category = category_match.group(1) if category_match else None
            
            if potential_category:
                # Check if this category exists in the data
                category_matches = df[df['issue_category'].str.contains(potential_category, case=False, na=False)]
                count = len(category_matches)
                
                if count > 0:
                    return f"There are **{count:,} calls** classified as **{potential_category}**-related out of {len(df):,} total calls ({count/len(df)*100:.1f}%).", None
                else:
                    available_categories = ', '.join(sorted(df['issue_category'].dropna().unique()))
                    return f"No calls found for **'{potential_category}'**. The available issue categories are: **{available_categories}**", None
            else:
                available_categories = ', '.join(sorted(df['issue_category'].dropna().unique()))
                return f"Could not identify the category from your query. The available issue categories are: **{available_categories}**", None
        
        # Simple category queries (without "classified" keyword)
        elif 'how many' in query_lower and 'calls' in query_lower:
            # Look for any word before "calls" that might be a category
            category_match = re.search(r'how many\s+([a-zA-Z-]+)\s+calls', query_lower)
            if not category_match:
                category_match = re.search(r'([a-zA-Z-]+)(?:-related)?\s+calls', query_lower)
            
            potential_category = category_match.group(1) if category_match else None
            
            if potential_category and potential_category not in ['many', 'total', 'how']:
                # Check issue_category column for matches
                category_matches = df[df['issue_category'].str.contains(potential_category, case=False, na=False)]
                count = len(category_matches)
                
                if count > 0:
                    return f"There are **{count:,} {potential_category}**-related calls out of {len(df):,} total calls ({count/len(df)*100:.1f}%).", None
                else:
                    available_categories = ', '.join(sorted(df['issue_category'].dropna().unique()))
                    return f"No **{potential_category}**-related calls found. The available issue categories are: **{available_categories}**", None
            else:
                # If no category keyword found, it's probably asking for total
                total_calls = len(df)
                return f"There are **{total_calls:,} total calls** in the dataset.", None
        
        # Total number of calls (simple case)
        elif 'total' in query_lower and 'calls' in query_lower:
            total_calls = len(df)
            return f"There are **{total_calls:,} total calls** in the dataset.", None
        
        # Average sentiment score
        elif 'average' in query_lower and 'sentiment' in query_lower:
            avg_sentiment = df['sentiment_score'].mean()
            sentiment_label = "positive" if avg_sentiment > 0 else "negative" if avg_sentiment < 0 else "neutral"
            return f"The average sentiment score is **{avg_sentiment:.2f}** ({sentiment_label}).", None
        
        # Resolution rate
        elif 'resolution' in query_lower and ('rate' in query_lower or 'percentage' in query_lower):
            resolved = (df['resolution_status'] == 'resolved').sum()
            total = len(df)
            percentage = (resolved / total) * 100
            return f"The resolution rate is **{percentage:.1f}%** ({resolved:,} out of {total:,} calls resolved).", None
        
        # Agent performance
        elif 'agent' in query_lower and ('highest' in query_lower or 'best' in query_lower or 'most' in query_lower):
            agent_stats = df.groupby('agent_name').agg({
                'sentiment_score': 'mean',
                'call_id': 'count'
            }).reset_index()
            agent_stats.columns = ['agent_name', 'avg_sentiment', 'call_count']
            agent_stats = agent_stats[agent_stats['call_count'] >= 2].sort_values('avg_sentiment', ascending=False)
            
            if len(agent_stats) > 0:
                top_agent = agent_stats.iloc[0]
                result = f"**{top_agent['agent_name']}** has the highest average sentiment score of **{top_agent['avg_sentiment']:.2f}** ({top_agent['call_count']} calls)."
                
                # Create chart
                fig = px.bar(
                    agent_stats.head(5), 
                    x='agent_name', 
                    y='avg_sentiment',
                    title='Top 5 Agents by Average Sentiment Score',
                    labels={'avg_sentiment': 'Average Sentiment Score', 'agent_name': 'Agent'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                
                return result, fig
            else:
                return "No agent performance data available.", None
        
        # Issue categories
        elif 'issue' in query_lower and ('category' in query_lower or 'categories' in query_lower):
            top_issues = df['issue_category'].value_counts().head(5)
            result = "**Top Issue Categories:**\n"
            for category, count in top_issues.items():
                percentage = (count / len(df)) * 100
                result += f"• {category}: {count:,} calls ({percentage:.1f}%)\n"
            
            # Create chart
            fig = px.bar(
                x=top_issues.index,
                y=top_issues.values,
                title='Top Issue Categories',
                labels={'x': 'Issue Category', 'y': 'Number of Calls'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            
            return result, fig
        
        # Calls by specific agent
        elif 'calls' in query_lower and any(name in query_lower for name in df['agent_name'].unique()):
            # Find agent name in query
            agent_name = None
            for name in df['agent_name'].unique():
                if name.lower() in query_lower:
                    agent_name = name
                    break
            
            if agent_name:
                agent_calls = df[df['agent_name'] == agent_name]
                count = len(agent_calls)
                avg_sentiment = agent_calls['sentiment_score'].mean()
                return f"**{agent_name}** handled **{count:,} calls** with an average sentiment score of **{avg_sentiment:.2f}**.", None
        
        # Basic location statistics
        elif ('location' in query_lower or any(loc in query_lower for loc in df['location'].unique())) and 'calls by location' in query_lower:
            location_stats = df.groupby('location').agg({
                'call_id': 'count',
                'sentiment_score': 'mean'
            }).reset_index()
            location_stats.columns = ['location', 'call_count', 'avg_sentiment']
            location_stats = location_stats.sort_values('call_count', ascending=False)
            
            result = "**Calls by Location:**\n"
            for _, row in location_stats.head(5).iterrows():
                result += f"• {row['location']}: {row['call_count']:,} calls (avg sentiment: {row['avg_sentiment']:.2f})\n"
            
            # Create chart
            fig = px.bar(
                location_stats,
                x='location',
                y='call_count',
                title='Calls by Location',
                labels={'call_count': 'Number of Calls', 'location': 'Location'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            
            return result, fig
        
        else:
            # If this was classified as numerical but doesn't match any pattern,
            # it might be a complex query that should be open-ended
            return None, None  # Signal to retry as open-ended
    
    except Exception as e:
        return f"Error processing numerical query: {e}", None

def answer_open_ended_query(query: str, relevant_calls: List[Dict]) -> str:
    """
    Answer open-ended questions using GPT-4o-mini with relevant call context.
    
    Args:
        query (str): User query
        relevant_calls (List[Dict]): Relevant calls from semantic search
        
    Returns:
        str: AI-generated answer
    """
    if not relevant_calls:
        return "I couldn't find relevant calls to answer your question. Please try rephrasing or being more specific."
    
    # Prepare context for GPT
    context = "Based on the following customer service calls, please provide a comprehensive answer:\n\n"
    
    for i, call in enumerate(relevant_calls, 1):
        context += f"**Call {i}** (Similarity: {call['similarity']:.2f}):\n"
        context += f"- Subject: {call['subject']}\n"
        context += f"- Agent: {call['agent_name']}\n"
        context += f"- Location: {call['location']}\n"
        context += f"- Sentiment: {call['sentiment_score']:.2f}\n"
        context += f"- Issue Category: {call['issue_category']}\n"
        context += f"- Resolution: {call['resolution_status']}\n"
        context += f"- Summary: {call['summary']}\n"
        context += f"- Transcript: {call['transcript'][:500]}...\n\n"
    
    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert customer service analyst for Feel Good Spas. Provide clear, actionable insights based on call data. Focus on patterns, trends, and specific examples from the calls."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\n{context}"
                }
            ],
            max_tokens=800,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI response: {e}"

def answer_query(query: str, df: pd.DataFrame, embeddings: Dict[int, List[float]]) -> Tuple[str, Optional[go.Figure], List[Dict]]:
    """
    Main function to answer user queries.
    
    Args:
        query (str): User query
        df (pd.DataFrame): Call data
        embeddings (Dict[int, List[float]]): Precomputed embeddings
        
    Returns:
        Tuple[str, Optional[go.Figure], List[Dict]]: Answer, chart, and source calls
    """
    query_type = classify_query(query)
    
    if query_type == 'numerical':
        answer, chart = answer_numerical_query(query, df)
        # If numerical query returned None, fall back to open-ended
        if answer is None:
            relevant_calls = semantic_search(query, embeddings, df)
            answer = answer_open_ended_query(query, relevant_calls)
            return answer, None, relevant_calls
        return answer, chart, []
    else:
        relevant_calls = semantic_search(query, embeddings, df)
        answer = answer_open_ended_query(query, relevant_calls)
        return answer, None, relevant_calls

def main():
    """
    Main Streamlit application.
    """
    st.title("💬 Feel Good Spas - AI Assistant")
    st.markdown("*Ask me anything about your customer service calls!*")
    
    # Load data and compute embeddings
    with st.spinner("Loading data..."):
        df = load_data()
    
    with st.spinner("Preparing AI models..."):
        embeddings, valid_indices = compute_embeddings(df)
    
    if not embeddings:
        st.error("❌ No embeddings available. Please check your data and try again.")
        st.stop()
    
    # Sidebar stats
    st.sidebar.header("📊 Data Overview")
    st.sidebar.metric("Total Calls", f"{len(df):,}")
    st.sidebar.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")
    st.sidebar.metric("Available Embeddings", f"{len(embeddings):,}")
    
    # Example queries
    st.sidebar.subheader("💡 Example Questions")
    
    st.sidebar.markdown("**📈 Numerical Queries:**")
    numerical_examples = [
        "What is the average call duration?",
        "Which agent has the highest sentiment score?",
        "How many calls were resolved?",
        "What are the top issue categories?",
        "How many calls did Ethan Williams handle?"
    ]
    
    st.sidebar.markdown("**🔍 Open-ended Queries:**")
    openended_examples = [
        "What do customers complain about most?",
        "Summarize the main booking issues",
        "How do customers feel about Chicago location?",
        "What are the recent unresolved issues?"
    ]
    
    all_examples = numerical_examples + openended_examples
    
    for example in all_examples:
        if st.sidebar.button(example, key=f"ex_{hash(example)}"):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI assistant for Feel Good Spas customer service analysis. I can help you with:\n\n• **📈 Numerical queries** - Statistics, averages, counts, top performers\n• **🔍 Open-ended analysis** - Insights, patterns, and trends from call content\n\nWhat would you like to know about your customer service calls?"
            }
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display chart if available
            if "chart" in message and message["chart"]:
                st.plotly_chart(message["chart"], use_container_width=True, key=f"chart_history_{hash(str(message))}")
            
            # Display source calls if available
            if "sources" in message and message["sources"]:
                with st.expander("📋 Source Calls"):
                    for i, call in enumerate(message["sources"], 1):
                        st.markdown(f"**Call {i}** (Similarity: {call['similarity']:.2f})")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Agent:** {call['agent_name']}")
                            st.markdown(f"**Customer:** {call['customer_name']}")
                        with col2:
                            st.markdown(f"**Location:** {call['location']}")
                            st.markdown(f"**Sentiment:** {call['sentiment_score']:.2f}")
                        with col3:
                            st.markdown(f"**Category:** {call['issue_category']}")
                            st.markdown(f"**Status:** {call['resolution_status']}")
                        st.markdown(f"**Summary:** {call['summary']}")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me about your customer service data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                answer, chart, sources = answer_query(prompt, df, embeddings)
            
            st.markdown(answer)
            
            # Display chart if available
            if chart:
                st.plotly_chart(chart, use_container_width=True, key=f"chart_current_{hash(prompt)}")
            
            # Display source calls if available
            if sources:
                with st.expander("📋 Source Calls"):
                    for i, call in enumerate(sources, 1):
                        st.markdown(f"**Call {i}** (Similarity: {call['similarity']:.2f})")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Agent:** {call['agent_name']}")
                            st.markdown(f"**Customer:** {call['customer_name']}")
                        with col2:
                            st.markdown(f"**Location:** {call['location']}")
                            st.markdown(f"**Sentiment:** {call['sentiment_score']:.2f}")
                        with col3:
                            st.markdown(f"**Category:** {call['issue_category']}")
                            st.markdown(f"**Status:** {call['resolution_status']}")
                        st.markdown(f"**Summary:** {call['summary']}")
                        st.markdown("---")
        
        # Add assistant message to history
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "chart": chart,
            "sources": sources
        }
        st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main() 