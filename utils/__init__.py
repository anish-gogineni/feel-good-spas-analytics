"""
Utils package for Feel Good Spas analytics platform.

This package contains utility functions for report generation and other helper functions.
"""

from .report_generator import generate_report_pdf, calculate_additional_metrics

__all__ = ['generate_report_pdf', 'calculate_additional_metrics'] 