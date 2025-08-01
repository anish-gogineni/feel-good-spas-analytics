#!/usr/bin/env python3
"""
Demo script to show AI enrichment functionality with sample data.
This demonstrates what would happen when the OpenAI API is available.
"""

import pandas as pd
import json
from typing import Dict, Any

def mock_ai_analysis(transcript: str, summary: str) -> Dict[str, Any]:
    """
    Mock AI analysis function that simulates OpenAI API responses.
    """
    # Simulate AI analysis based on content keywords
    content = (transcript + " " + summary).lower()
    
    # Sentiment analysis simulation
    negative_words = ['frustrated', 'unacceptable', 'ridiculous', 'disappointed', 'terrible', 'bad']
    positive_words = ['fantastic', 'excellent', 'perfect', 'wonderful', 'great', 'thank you']
    
    neg_count = sum(1 for word in negative_words if word in content)
    pos_count = sum(1 for word in positive_words if word in content)
    
    if neg_count > pos_count:
        sentiment = -0.3 - (neg_count * 0.2)
    elif pos_count > neg_count:
        sentiment = 0.3 + (pos_count * 0.2)
    else:
        sentiment = 0.0
    
    sentiment = max(-1.0, min(1.0, sentiment))
    
    # Issue category simulation
    if 'booking' in content or 'appointment' in content:
        if 'cancel' in content:
            category = "booking cancellation"
        elif 'reschedule' in content:
            category = "booking rescheduling"
        else:
            category = "booking issue"
    elif 'membership' in content:
        category = "membership inquiry"
    elif 'billing' in content or 'charge' in content:
        category = "billing inquiry"
    elif 'accessibility' in content:
        category = "accessibility inquiry"
    else:
        category = "general inquiry"
    
    # Resolution status simulation
    if any(word in content for word in ['resolved', 'fixed', 'thank you', 'perfect']):
        status = "resolved"
    elif any(word in content for word in ['nevermind', 'elsewhere', 'bad review', 'unacceptable']):
        status = "unresolved"
    elif 'manager' in content or 'supervisor' in content:
        status = "escalated"
    else:
        # Default based on sentiment
        status = "resolved" if sentiment > -0.3 else "unresolved"
    
    return {
        "sentiment_score": round(sentiment, 2),
        "issue_category": category,
        "resolution_status": status
    }

def demo_enrichment():
    """
    Demonstrate AI enrichment on a few sample records.
    """
    print("🤖 AI Enrichment Demo")
    print("=" * 50)
    
    # Load a few sample records
    df = pd.read_csv('data/processed_calls.csv').head(5)
    
    for idx, row in df.iterrows():
        print(f"\n📞 Call {idx + 1}: {row['subject']}")
        print(f"Agent: {row['agent_name']}")
        print(f"Customer: {row['customer_name']}")
        print(f"Duration: {row['call_duration']} seconds")
        
        # Get AI analysis
        ai_result = mock_ai_analysis(row['transcript'], row['summary'])
        
        print(f"\n🔍 AI Analysis:")
        print(f"  Sentiment Score: {ai_result['sentiment_score']} ({get_sentiment_label(ai_result['sentiment_score'])})")
        print(f"  Issue Category: {ai_result['issue_category']}")
        print(f"  Resolution Status: {ai_result['resolution_status']}")
        
        print("-" * 50)

def get_sentiment_label(score: float) -> str:
    """Convert sentiment score to human-readable label."""
    if score >= 0.5:
        return "Very Positive"
    elif score >= 0.1:
        return "Positive"
    elif score >= -0.1:
        return "Neutral"
    elif score >= -0.5:
        return "Negative"
    else:
        return "Very Negative"

def show_usage_instructions():
    """Show instructions for using the AI enrichment feature."""
    print("\n🚀 How to Enable AI Enrichment")
    print("=" * 50)
    print("1. Get an OpenAI API key from https://platform.openai.com/api-keys")
    print("2. Set the environment variable:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("3. Run the processing script:")
    print("   python scripts/process_data.py")
    print("4. The enriched data will be saved to data/processed_calls_enriched.csv")
    print("\n💡 Features:")
    print("- Sentiment analysis (-1 to 1 scale)")
    print("- Automatic issue categorization")
    print("- Resolution status detection")
    print("- Batch processing with rate limiting")
    print("- Progress tracking and error handling")

if __name__ == "__main__":
    demo_enrichment()
    show_usage_instructions() 