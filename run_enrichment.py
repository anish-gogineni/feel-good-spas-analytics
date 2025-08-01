#!/usr/bin/env python3
"""
Wrapper script to run the complete Feel Good Spas data processing and AI enrichment pipeline.
This script checks for API key configuration and runs the full process.
"""

import os
import sys
import subprocess

def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("\nPlease set your API key first:")
        print("export OPENAI_API_KEY='sk-your-actual-key-here'")
        print("\nThen run this script again:")
        print("python run_enrichment.py")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key doesn't look like a valid OpenAI key (should start with 'sk-')")
        return False
    
    print(f"✅ OpenAI API key configured (ending in ...{api_key[-6:]})")
    return True

def run_pipeline():
    """Run the complete data processing pipeline."""
    try:
        print("\n🚀 Starting Feel Good Spas data processing and AI enrichment...")
        print("=" * 70)
        
        # Run the main processing script
        result = subprocess.run([sys.executable, 'scripts/process_data.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print("\n🎉 Pipeline completed successfully!")
            return True
        else:
            print("❌ Pipeline failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

def show_results():
    """Show summary of the enrichment results."""
    try:
        # Check if enriched file exists
        enriched_file = 'data/processed_calls_enriched.csv'
        if not os.path.exists(enriched_file):
            print(f"❌ Enriched file not found: {enriched_file}")
            return
        
        print("\n📊 Showing enrichment results...")
        print("=" * 50)
        
        # Run a quick analysis of the results
        analysis_code = '''
import pandas as pd
df = pd.read_csv("data/processed_calls_enriched.csv")

print(f"📋 Dataset Summary:")
print(f"Total records: {len(df)}")
print(f"Records with sentiment scores: {df['sentiment_score'].notna().sum()}")
print(f"Records with issue categories: {df['issue_category'].notna().sum()}")
print(f"Records with resolution status: {df['resolution_status'].notna().sum()}")

print(f"\\n😊 Sentiment Distribution:")
sentiment_ranges = {
    'Very Positive (0.5 to 1.0)': len(df[df['sentiment_score'] >= 0.5]),
    'Positive (0.1 to 0.5)': len(df[(df['sentiment_score'] >= 0.1) & (df['sentiment_score'] < 0.5)]),
    'Neutral (-0.1 to 0.1)': len(df[(df['sentiment_score'] >= -0.1) & (df['sentiment_score'] < 0.1)]),
    'Negative (-0.5 to -0.1)': len(df[(df['sentiment_score'] >= -0.5) & (df['sentiment_score'] < -0.1)]),
    'Very Negative (-1.0 to -0.5)': len(df[df['sentiment_score'] < -0.5])
}

for range_name, count in sentiment_ranges.items():
    if count > 0:
        percentage = count / len(df) * 100
        print(f"  {range_name}: {count} ({percentage:.1f}%)")

print(f"\\n📋 Top Issue Categories:")
category_counts = df['issue_category'].value_counts().head(5)
for category, count in category_counts.items():
    percentage = count / len(df) * 100
    print(f"  {category}: {count} ({percentage:.1f}%)")

print(f"\\n✅ Resolution Status:")
status_counts = df['resolution_status'].value_counts()
for status, count in status_counts.items():
    percentage = count / len(df) * 100
    print(f"  {status}: {count} ({percentage:.1f}%)")

print(f"\\n📞 Sample of First 3 Enriched Records:")
for i in range(min(3, len(df))):
    row = df.iloc[i]
    print(f"\\n{i+1}. {row['subject']}")
    print(f"   Agent: {row['agent_name']}")
    print(f"   Customer: {row['customer_name']}")
    print(f"   Sentiment: {row['sentiment_score']}")
    print(f"   Category: {row['issue_category']}")
    print(f"   Status: {row['resolution_status']}")
'''
        
        result = subprocess.run([sys.executable, '-c', analysis_code], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Error analyzing results:", result.stderr)
            
    except Exception as e:
        print(f"Error showing results: {e}")

def main():
    """Main function to run the complete workflow."""
    print("🤖 Feel Good Spas AI Enrichment Pipeline")
    print("=" * 50)
    
    # Check API key
    if not check_api_key():
        return
    
    # Run the pipeline
    if not run_pipeline():
        return
    
    # Show results
    show_results()
    
    print(f"\n✨ All done! Check the enriched data at: data/processed_calls_enriched.csv")

if __name__ == "__main__":
    main() 