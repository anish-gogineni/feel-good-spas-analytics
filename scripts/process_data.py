#!/usr/bin/env python3
"""
Process Feel Good Spas vCon Data

This script processes the raw vCon JSON data from Feel Good Spas customer service calls
and converts it into a clean, structured CSV format for analysis.

The script extracts key fields from the nested vCon structure and creates placeholder
columns for future sentiment analysis and categorization.
"""

import json
import pandas as pd
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import openai
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.warning("OPENAI_API_KEY environment variable not set. AI enrichment will be skipped.")


def extract_call_metadata(call_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic call metadata from vCon record.
    
    Args:
        call_data: Single vCon call record
        
    Returns:
        Dictionary with extracted metadata fields
    """
    try:
        return {
            'call_id': call_data.get('id', ''),
            'subject': call_data.get('subject', ''),
            'call_created_at': call_data.get('created_at', '')
        }
    except Exception as e:
        logger.warning(f"Error extracting metadata for call: {e}")
        return {
            'call_id': '',
            'subject': '',
            'call_created_at': ''
        }


def extract_call_attachments(vcon_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract call duration, location, and call type from attachments.
    
    Args:
        vcon_json: The vcon_json section of the call record
        
    Returns:
        Dictionary with duration, location, and call_type
    """
    duration = None
    location = ''
    call_type = ''
    
    try:
        attachments = vcon_json.get('attachments', [])
        for attachment in attachments:
            if attachment.get('type') == 'tags':
                tags = attachment.get('body', [])
                for tag in tags:
                    if isinstance(tag, str):
                        if tag.startswith('duration:'):
                            duration = int(tag.split(':')[1])
                        elif tag.startswith('location:'):
                            location = tag.split(':', 1)[1]
                        elif tag.startswith('call_type:'):
                            call_type = tag.split(':', 1)[1]
    except Exception as e:
        logger.warning(f"Error extracting attachments: {e}")
    
    return {
        'call_duration': duration,
        'location': location,
        'call_type': call_type
    }


def extract_parties(vcon_json: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract agent and customer names from parties data.
    
    Args:
        vcon_json: The vcon_json section of the call record
        
    Returns:
        Dictionary with agent_name and customer_name
    """
    agent_name = ''
    customer_name = ''
    
    try:
        parties = vcon_json.get('parties', [])
        for party in parties:
            role = party.get('role', '')
            name = party.get('name', '')
            
            if role == 'support_agent':
                agent_name = name
            elif role == 'customer':
                customer_name = name
            elif not role and name:
                # Handle cases where role is not specified
                # Heuristic: if name contains "Customer Service" or company name, it's likely the agent
                if any(keyword in name.lower() for keyword in ['customer service', 'feel good spas', 'support']):
                    agent_name = name
                else:
                    customer_name = name
    except Exception as e:
        logger.warning(f"Error extracting parties: {e}")
    
    return {
        'agent_name': agent_name,
        'customer_name': customer_name
    }


def extract_analysis_content(vcon_json: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract transcript and summary from analysis data.
    
    Args:
        vcon_json: The vcon_json section of the call record
        
    Returns:
        Dictionary with transcript and summary
    """
    transcript = ''
    summary = ''
    
    try:
        analysis = vcon_json.get('analysis', [])
        for item in analysis:
            analysis_type = item.get('type', '')
            body = item.get('body', {})
            
            if analysis_type == 'transcript':
                transcript = body.get('text', '')
            elif analysis_type == 'summary':
                summary = body.get('text', '')
    except Exception as e:
        logger.warning(f"Error extracting analysis content: {e}")
    
    return {
        'transcript': transcript,
        'summary': summary
    }


def process_single_call(call_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single vCon call record and extract all required fields.
    
    Args:
        call_data: Single call record from the JSON data
        
    Returns:
        Dictionary with all extracted and placeholder fields
    """
    # Extract basic metadata
    metadata = extract_call_metadata(call_data)
    
    # Get vcon_json section
    vcon_json = call_data.get('vcon_json', {})
    
    # Extract various components
    attachments = extract_call_attachments(vcon_json)
    parties = extract_parties(vcon_json)
    analysis = extract_analysis_content(vcon_json)
    
    # Combine all extracted data
    processed_call = {
        **metadata,
        **attachments,
        **parties,
        **analysis,
        # Placeholder columns for future processing
        'sentiment_score': None,
        'issue_category': None,
        'resolution_status': None
    }
    
    return processed_call


def load_vcon_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load vCon data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of call records
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logger.info(f"Successfully loaded {len(data)} call records from {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading file {file_path}: {e}")
        raise


def enrich_row_with_ai(row: pd.Series) -> pd.Series:
    """
    Enrich a single row with AI-generated sentiment, category, and resolution status.
    
    Args:
        row: DataFrame row containing transcript and summary
        
    Returns:
        Updated row with AI-generated values or original values if API fails
    """
    if not openai.api_key:
        return row
    
    transcript = row.get('transcript', '')
    summary = row.get('summary', '')
    
    # Skip if no content to analyze
    if not transcript and not summary:
        return row
    
    # Prepare content for analysis
    content_to_analyze = f"Transcript: {transcript}\n\nSummary: {summary}"
    
    try:
        prompt = """
        Analyze this customer service call and provide:
        1. sentiment_score: A float between -1 (very negative) and 1 (very positive)
        2. issue_category: A short category (1-3 words) like "booking issue", "cancellation", "rescheduling"
        3. resolution_status: One of "resolved", "unresolved", or "escalated"
        
        Respond ONLY with a JSON object in this exact format:
        {
            "sentiment_score": 0.2,
            "issue_category": "booking issue",
            "resolution_status": "resolved"
        }
        
        Customer service call content:
        """ + content_to_analyze[:3000]  # Limit content to avoid token limits
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert customer service analyst. Provide accurate, concise analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        
        try:
            ai_analysis = json.loads(response_text)
            
            # Validate and apply the results
            if 'sentiment_score' in ai_analysis:
                # Ensure sentiment score is within bounds
                sentiment = float(ai_analysis['sentiment_score'])
                row['sentiment_score'] = max(-1.0, min(1.0, sentiment))
            
            if 'issue_category' in ai_analysis:
                category = str(ai_analysis['issue_category']).strip().lower()
                row['issue_category'] = category
            
            if 'resolution_status' in ai_analysis:
                status = str(ai_analysis['resolution_status']).strip().lower()
                if status in ['resolved', 'unresolved', 'escalated']:
                    row['resolution_status'] = status
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse AI response as JSON for call {row.get('call_id', 'unknown')}")
    
    except Exception as e:
        logger.warning(f"AI enrichment failed for call {row.get('call_id', 'unknown')}: {e}")
    
    return row


def enrich_dataframe_with_ai(df: pd.DataFrame, batch_size: int = 10) -> pd.DataFrame:
    """
    Enrich DataFrame with AI-generated analysis in batches.
    
    Args:
        df: DataFrame to enrich
        batch_size: Number of rows to process in each batch
        
    Returns:
        Enriched DataFrame
    """
    if not openai.api_key:
        logger.info("Skipping AI enrichment - no OpenAI API key configured")
        return df
    
    logger.info(f"Starting AI enrichment for {len(df)} records in batches of {batch_size}")
    
    enriched_df = df.copy()
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="AI Enrichment"):
        batch_end = min(i + batch_size, len(df))
        batch_indices = list(range(i, batch_end))
        
        logger.info(f"Processing batch {i//batch_size + 1}: rows {i+1}-{batch_end}")
        
        for idx in batch_indices:
            try:
                enriched_df.iloc[idx] = enrich_row_with_ai(enriched_df.iloc[idx])
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to enrich row {idx}: {e}")
                continue
        
        # Longer delay between batches
        if i + batch_size < len(df):
            logger.info("Pausing between batches to respect rate limits...")
            time.sleep(2)
    
    # Log enrichment statistics
    sentiment_filled = enriched_df['sentiment_score'].notna().sum()
    category_filled = enriched_df['issue_category'].notna().sum()
    status_filled = enriched_df['resolution_status'].notna().sum()
    
    logger.info(f"AI enrichment completed:")
    logger.info(f"  Sentiment scores: {sentiment_filled}/{len(df)} ({sentiment_filled/len(df)*100:.1f}%)")
    logger.info(f"  Issue categories: {category_filled}/{len(df)} ({category_filled/len(df)*100:.1f}%)")
    logger.info(f"  Resolution status: {status_filled}/{len(df)} ({status_filled/len(df)*100:.1f}%)")
    
    return enriched_df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save processed DataFrame to CSV file.
    
    Args:
        df: Processed DataFrame
        output_path: Path where CSV should be saved
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} processed records to {output_path}")
        
        # Log basic statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Log data quality metrics
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.info("Missing data summary:")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.info(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error saving processed data to {output_path}: {e}")
        raise


def main():
    """
    Main function to orchestrate the data processing pipeline.
    """
    # Define file paths
    input_file = Path("data/feel-good-spas-vcons.json")
    output_file = Path("data/processed_calls.csv")
    enriched_output_file = Path("data/processed_calls_enriched.csv")
    
    logger.info("Starting Feel Good Spas vCon data processing...")
    
    try:
        # Load raw data
        logger.info("Loading vCon data...")
        raw_data = load_vcon_data(input_file)
        
        # Process each call record
        logger.info("Processing call records...")
        processed_calls = []
        
        for i, call_data in enumerate(raw_data):
            try:
                processed_call = process_single_call(call_data)
                processed_calls.append(processed_call)
            except Exception as e:
                logger.warning(f"Error processing call record {i}: {e}")
                continue
        
        # Convert to DataFrame
        logger.info("Creating DataFrame...")
        df = pd.DataFrame(processed_calls)
        
        # Define expected column order
        expected_columns = [
            'call_id', 'subject', 'call_created_at', 'call_duration', 
            'location', 'call_type', 'agent_name', 'customer_name',
            'transcript', 'summary', 'sentiment_score', 'issue_category', 
            'resolution_status'
        ]
        
        # Reorder columns to match schema
        df = df.reindex(columns=expected_columns)
        
        # Save basic processed data
        logger.info("Saving basic processed data...")
        save_processed_data(df, output_file)
        
        # AI Enrichment Phase
        logger.info("Starting AI enrichment phase...")
        
        if openai.api_key:
            # Enrich with AI analysis
            enriched_df = enrich_dataframe_with_ai(df, batch_size=10)
            
            # Save enriched data
            logger.info("Saving AI-enriched data...")
            save_processed_data(enriched_df, enriched_output_file)
            
            logger.info("Data processing and AI enrichment completed successfully!")
        else:
            logger.warning("OpenAI API key not found. Skipping AI enrichment.")
            logger.info("To enable AI enrichment, set the OPENAI_API_KEY environment variable.")
            logger.info("Basic data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in data processing: {e}")
        raise


if __name__ == "__main__":
    main() 