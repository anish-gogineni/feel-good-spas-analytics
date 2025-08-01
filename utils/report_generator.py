#!/usr/bin/env python3
"""
Feel Good Spas - PDF Report Generator

Generates professional PDF reports with business insights and recommendations.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import pandas as pd
import io


def calculate_additional_metrics(df):
    """
    Calculate additional metrics beyond the basic KPIs.
    
    Args:
        df (pd.DataFrame): Call data
        
    Returns:
        dict: Additional calculated metrics
    """
    metrics = {}
    
    # Issue category breakdown
    issue_counts = df['issue_category'].value_counts()
    total_calls = len(df)
    
    metrics['top_issue'] = issue_counts.index[0] if len(issue_counts) > 0 else 'N/A'
    metrics['top_issue_pct'] = (issue_counts.iloc[0] / total_calls * 100) if len(issue_counts) > 0 else 0
    
    # Sentiment breakdown
    positive_calls = (df['sentiment_score'] > 0.1).sum()
    negative_calls = (df['sentiment_score'] < -0.1).sum()
    neutral_calls = total_calls - positive_calls - negative_calls
    
    metrics['positive_pct'] = (positive_calls / total_calls * 100) if total_calls > 0 else 0
    metrics['negative_pct'] = (negative_calls / total_calls * 100) if total_calls > 0 else 0
    metrics['neutral_pct'] = (neutral_calls / total_calls * 100) if total_calls > 0 else 0
    
    # Agent performance
    agent_stats = df.groupby('agent_name').agg({
        'sentiment_score': 'mean',
        'call_id': 'count'
    }).reset_index()
    
    if len(agent_stats) > 0:
        best_agent = agent_stats.loc[agent_stats['sentiment_score'].idxmax()]
        metrics['best_agent'] = best_agent['agent_name']
        metrics['best_agent_sentiment'] = best_agent['sentiment_score']
        metrics['total_agents'] = len(agent_stats)
    else:
        metrics['best_agent'] = 'N/A'
        metrics['best_agent_sentiment'] = 0
        metrics['total_agents'] = 0
    
    # Call duration insights
    long_calls = (df['call_duration'] > df['call_duration'].quantile(0.75)).sum()
    metrics['long_calls_pct'] = (long_calls / total_calls * 100) if total_calls > 0 else 0
    
    return metrics


def generate_report_pdf(df, filename="FeelGoodSpas_Insights.pdf"):
    """
    Generate a professional PDF report with business insights.
    
    Args:
        df (pd.DataFrame): Call data from the dashboard
        filename (str): Output filename for the PDF
        
    Returns:
        bytes: PDF content as bytes
    """
    # Create PDF buffer
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#2E86AB')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=HexColor('#2E86AB')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=HexColor('#4A4A4A')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Calculate metrics
    total_calls = len(df)
    avg_sentiment = df['sentiment_score'].mean()
    avg_duration = df['call_duration'].mean() / 60  # Convert to minutes
    resolved_rate = (df['resolution_status'] == 'resolved').mean() * 100
    
    additional_metrics = calculate_additional_metrics(df)
    
    # Content elements
    story = []
    
    # Cover Page
    story.append(Paragraph("Feel Good Spas", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Customer Service Insights Report", heading_style))
    story.append(Spacer(1, 30))
    
    # Date and period
    report_date = datetime.now().strftime("%B %d, %Y")
    data_period = f"{df['call_created_at'].min().strftime('%B %d, %Y')} - {df['call_created_at'].max().strftime('%B %d, %Y')}"
    
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", body_style))
    story.append(Paragraph(f"<b>Data Period:</b> {data_period}", body_style))
    story.append(Paragraph(f"<b>Total Calls Analyzed:</b> {total_calls:,}", body_style))
    
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    executive_summary = f"""
    This report analyzes {total_calls:,} customer service conversations for Feel Good Spas, providing actionable 
    insights to improve customer satisfaction and operational efficiency. Our AI-powered analysis reveals key patterns 
    in customer sentiment, agent performance, and issue resolution that directly impact business outcomes.
    
    The analysis shows a current resolution rate of {resolved_rate:.1f}% with an average sentiment score of 
    {avg_sentiment:.2f}. The top issue category is <b>{additional_metrics['top_issue']}</b>, representing 
    {additional_metrics['top_issue_pct']:.1f}% of all calls. Strategic improvements in these areas could significantly 
    enhance customer experience and reduce operational costs.
    """
    
    story.append(Paragraph(executive_summary, body_style))
    story.append(Spacer(1, 20))
    
    # Key Insights Table
    story.append(Paragraph("Key Business Insights", heading_style))
    
    insights_data = [
        ['Insight Area', 'Business Value', 'Recommended Action', 'Expected Outcome'],
        [
            'Customer Sentiment Analysis',
            'Identify satisfaction trends and at-risk customers',
            'Proactive follow-up for negative sentiment calls',
            'Increase retention 15-20%'
        ],
        [
            'Agent Performance',
            f'Best agent: {additional_metrics["best_agent_sentiment"]:.2f} sentiment score',
            'Train agents using best practices',
            'Improve sentiment +0.2 points'
        ],
        [
            'Issue Categories',
            f'{additional_metrics["top_issue"]} dominates ({additional_metrics["top_issue_pct"]:.1f}%)',
            'Targeted solutions for top categories',
            'Reduce call volume 25%'
        ],
        [
            'Resolution Rate',
            f'Current {resolved_rate:.1f}% needs improvement',
            'Advanced training and decision trees',
            'Achieve 75%+ target'
        ],
        [
            'Call Duration',
            f'{additional_metrics["long_calls_pct"]:.1f}% exceed normal duration',
            'Streamline complex processes',
            'Reduce time 2-3 minutes'
        ]
    ]
    
    insights_table = Table(insights_data, colWidths=[1.8*inch, 1.6*inch, 1.8*inch, 1.4*inch])
    insights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F8F9FA')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, black),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F8F9FA')])
    ]))
    
    story.append(insights_table)
    story.append(Spacer(1, 20))
    
    # KPIs & Metrics Section
    story.append(Paragraph("Key Performance Indicators", heading_style))
    
    kpi_data = [
        ['Metric', 'Current Value', 'Target', 'Status'],
        ['Total Calls Analyzed', f'{total_calls:,}', 'N/A', '✓ Complete'],
        ['Avg Sentiment Score', f'{avg_sentiment:.2f}', '0.35', '🔄 Improving' if avg_sentiment > 0 else '⚠️ Attention'],
        ['Resolution Rate', f'{resolved_rate:.1f}%', '75%', '✓ Good' if resolved_rate > 70 else '⚠️ Improve'],
        ['Avg Call Duration', f'{avg_duration:.1f} min', '5-7 min', '✓ Good' if avg_duration < 8 else '⚠️ High'],
        ['Positive Sentiment', f'{additional_metrics["positive_pct"]:.1f}%', '60%+', '✓ Good' if additional_metrics["positive_pct"] > 50 else '⚠️ Low'],
        ['Agent Team Size', f'{additional_metrics["total_agents"]} agents', 'Optimal', '✓ Staffed']
    ]
    
    kpi_table = Table(kpi_data, colWidths=[2.2*inch, 1.4*inch, 1.0*inch, 1.6*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F8F9FA')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, black),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F8F9FA')])
    ]))
    
    story.append(kpi_table)
    story.append(Spacer(1, 20))
    
    # ROI Projection Summary
    story.append(Paragraph("ROI Projection & Business Impact", heading_style))
    
    # Calculate ROI based on current metrics
    current_resolution_rate = resolved_rate / 100
    target_resolution_rate = 0.75
    improvement_potential = target_resolution_rate - current_resolution_rate
    
    # Estimate annual call volume (assuming current data is representative)
    days_in_data = (df['call_created_at'].max() - df['call_created_at'].min()).days + 1
    daily_calls = total_calls / days_in_data
    annual_calls = daily_calls * 365
    
    # Cost savings calculations
    avg_call_cost = 15  # Estimated cost per call
    repeat_call_reduction = improvement_potential * annual_calls
    annual_savings = repeat_call_reduction * avg_call_cost
    
    roi_text = f"""
    <b>Current Performance:</b> {resolved_rate:.1f}% resolution rate with {total_calls:,} calls analyzed over {days_in_data} days.
    
    <b>Improvement Opportunity:</b> Increasing resolution rate to 75% could prevent approximately {repeat_call_reduction:.0f} 
    repeat calls annually, representing ${annual_savings:,.0f} in cost savings.
    
    <b>Sentiment Impact:</b> Improving average sentiment from {avg_sentiment:.2f} to 0.35 correlates with 15-20% higher 
    customer retention. For a spa business, this translates to significant lifetime value preservation.
    
    <b>Operational Efficiency:</b> Optimizing the top issue category ({additional_metrics['top_issue']}) could reduce 
    overall call volume by 25%, allowing agents to focus on higher-value customer interactions.
    """
    
    story.append(Paragraph(roi_text, body_style))
    story.append(Spacer(1, 20))
    
    # Next Steps / Recommendations
    story.append(Paragraph("Strategic Recommendations", heading_style))
    
    recommendations = f"""
    <b>Immediate Actions (0-30 days):</b>
    • Deploy sentiment monitoring for all calls with alerts for negative experiences
    • Implement best-practice sharing sessions led by top-performing agent: {additional_metrics['best_agent']}
    • Create decision trees for {additional_metrics['top_issue']} issues to improve resolution rates
    
    <b>Short-term Initiatives (1-3 months):</b>
    • Launch comprehensive agent training program focusing on emotional intelligence
    • Develop self-service resources for top 3 issue categories
    • Implement proactive follow-up protocols for unresolved cases
    
    <b>Long-term Strategy (3-12 months):</b>
    • Deploy AI-powered real-time agent assistance during calls
    • Establish customer success programs for high-value clients
    • Create predictive models to identify at-risk customers before they call
    
    <b>Success Metrics to Track:</b>
    • Monthly resolution rate trending toward 75%
    • Average sentiment score improvement to 0.35+
    • 25% reduction in repeat calls for {additional_metrics['top_issue']} issues
    • Customer satisfaction scores correlated with sentiment analysis
    """
    
    story.append(Paragraph(recommendations, body_style))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF content
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content 