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
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        
        # Save processed data
        logger.info("Saving processed data...")
        save_processed_data(df, output_file)
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in data processing: {e}")
        raise


if __name__ == "__main__":
    main() 