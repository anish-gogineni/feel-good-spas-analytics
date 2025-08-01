#!/usr/bin/env python3
"""
Create a sample enriched CSV file to demonstrate AI enrichment results.
This shows what the output would look like when using the actual OpenAI API.
"""

import pandas as pd
import random
from typing import Dict, Any

def mock_ai_analysis(transcript: str, summary: str) -> Dict[str, Any]:
    """Enhanced mock AI analysis that simulates realistic OpenAI responses."""
    content = (transcript + " " + summary).lower()
    
    # More sophisticated sentiment analysis
    negative_indicators = [
        'frustrated', 'unacceptable', 'ridiculous', 'disappointed', 
        'terrible', 'bad', 'awful', 'angry', 'upset', 'furious',
        'horrible', 'worst', 'hate', 'disgusted', 'appalled'
    ]
    
    positive_indicators = [
        'fantastic', 'excellent', 'perfect', 'wonderful', 'great',
        'amazing', 'outstanding', 'superb', 'brilliant', 'pleased',
        'satisfied', 'happy', 'delighted', 'appreciate', 'thank you'
    ]
    
    neutral_indicators = [
        'okay', 'fine', 'alright', 'understand', 'got it'
    ]
    
    # Count sentiment indicators
    neg_count = sum(1 for word in negative_indicators if word in content)
    pos_count = sum(1 for word in positive_indicators if word in content)
    neu_count = sum(1 for word in neutral_indicators if word in content)
    
    # Calculate sentiment score
    if neg_count > pos_count:
        sentiment = max(-1.0, -0.2 - (neg_count * 0.15) + random.uniform(-0.1, 0.1))
    elif pos_count > neg_count:
        sentiment = min(1.0, 0.2 + (pos_count * 0.15) + random.uniform(-0.1, 0.1))
    else:
        sentiment = random.uniform(-0.2, 0.2)
    
    # Determine issue category
    if 'reschedule' in content or 'change' in content:
        category = "appointment rescheduling"
    elif 'cancel' in content:
        category = "appointment cancellation"
    elif 'booking' in content or 'appointment' in content:
        category = "booking issue"
    elif 'membership' in content:
        category = "membership inquiry"
    elif 'billing' in content or 'charge' in content or 'payment' in content:
        category = "billing inquiry"
    elif 'accessibility' in content:
        category = "accessibility inquiry"
    elif 'promotion' in content or 'discount' in content:
        category = "promotion inquiry"
    else:
        category = "general inquiry"
    
    # Determine resolution status
    if any(word in content for word in ['resolved', 'fixed', 'rescheduled', 'confirmed']):
        status = "resolved"
    elif any(word in content for word in ['nevermind', 'elsewhere', 'bad review', 'unacceptable', 'disappointed']):
        status = "unresolved"
    elif 'manager' in content or 'supervisor' in content:
        status = "escalated"
    else:
        # Use sentiment to help determine status
        if sentiment > 0.2:
            status = "resolved"
        elif sentiment < -0.4:
            status = "unresolved"
        else:
            status = random.choice(["resolved", "unresolved"])
    
    return {
        "sentiment_score": round(sentiment, 2),
        "issue_category": category,
        "resolution_status": status
    }

def create_sample_enriched_data():
    """Create a sample enriched CSV file."""
    print("🔧 Creating sample enriched data...")
    
    # Load the processed calls
    df = pd.read_csv('data/processed_calls.csv')
    
    print(f"Processing {len(df)} records with mock AI analysis...")
    
    # Apply mock AI analysis to each row
    for idx, row in df.iterrows():
        ai_result = mock_ai_analysis(row['transcript'], row['summary'])
        df.at[idx, 'sentiment_score'] = ai_result['sentiment_score']
        df.at[idx, 'issue_category'] = ai_result['issue_category']
        df.at[idx, 'resolution_status'] = ai_result['resolution_status']
    
    # Save sample enriched data
    output_path = 'data/sample_processed_calls_enriched.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample enriched data saved to {output_path}")
    
    # Show statistics
    print("\n📊 Sample Enrichment Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Sentiment scores filled: {df['sentiment_score'].notna().sum()}")
    print(f"Issue categories filled: {df['issue_category'].notna().sum()}")
    print(f"Resolution status filled: {df['resolution_status'].notna().sum()}")
    
    # Show sentiment distribution
    print(f"\n😊 Sentiment Distribution:")
    sentiment_ranges = {
        'Very Positive (0.5 to 1.0)': len(df[df['sentiment_score'] >= 0.5]),
        'Positive (0.1 to 0.5)': len(df[(df['sentiment_score'] >= 0.1) & (df['sentiment_score'] < 0.5)]),
        'Neutral (-0.1 to 0.1)': len(df[(df['sentiment_score'] >= -0.1) & (df['sentiment_score'] < 0.1)]),
        'Negative (-0.5 to -0.1)': len(df[(df['sentiment_score'] >= -0.5) & (df['sentiment_score'] < -0.1)]),
        'Very Negative (-1.0 to -0.5)': len(df[df['sentiment_score'] < -0.5])
    }
    
    for range_name, count in sentiment_ranges.items():
        percentage = count / len(df) * 100
        print(f"  {range_name}: {count} ({percentage:.1f}%)")
    
    # Show top issue categories
    print(f"\n📋 Top Issue Categories:")
    category_counts = df['issue_category'].value_counts().head(5)
    for category, count in category_counts.items():
        percentage = count / len(df) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Show resolution status distribution
    print(f"\n✅ Resolution Status Distribution:")
    status_counts = df['resolution_status'].value_counts()
    for status, count in status_counts.items():
        percentage = count / len(df) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    
    return df

if __name__ == "__main__":
    enriched_df = create_sample_enriched_data() 