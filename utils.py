"""
Utility functions for data processing and visualization helpers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """Automatically detect date columns in dataframe"""
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                continue
        elif 'date' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
    return date_cols

def auto_aggregate_data(df: pd.DataFrame, 
                        time_col: str = None,
                        value_cols: List[str] = None) -> pd.DataFrame:
    """Automatically aggregate data based on detected patterns"""
    
    if time_col and value_cols:
        # Time-based aggregation
        df[time_col] = pd.to_datetime(df[time_col])
        df['period'] = df[time_col].dt.to_period('M')
        aggregated = df.groupby('period')[value_cols].sum().reset_index()
        aggregated['period'] = aggregated['period'].astype(str)
        return aggregated
    
    return df

def generate_insights(df: pd.DataFrame, chart_spec: Dict) -> List[str]:
    """Generate natural language insights from data"""
    insights = []
    
    if 'x_axis' in chart_spec and 'y_axis' in chart_spec:
        x_col = chart_spec['x_axis']
        y_col = chart_spec['y_axis'][0] if chart_spec['y_axis'] else None
        
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            # Top performer
            top_idx = df[y_col].idxmax()
            insights.append(f"Top performer: {df.loc[top_idx, x_col]} ({df.loc[top_idx, y_col]:,.0f})")
            
            # Bottom performer
            bottom_idx = df[y_col].idxmin()
            insights.append(f"Lowest performer: {df.loc[bottom_idx, x_col]} ({df.loc[bottom_idx, y_col]:,.0f})")
            
            # Trend
            if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                growth = ((df[y_col].iloc[-1] - df[y_col].iloc[0]) / df[y_col].iloc[0] * 100)
                insights.append(f"Overall growth: {growth:+.1f}%")
    
    return insights

def validate_query_complexity(query: str) -> Dict[str, Any]:
    """Validate and classify query complexity"""
    complexity = {
        'level': 'simple',
        'requires_aggregation': False,
        'requires_filtering': False,
        'requires_join': False,
        'time_range': None
    }
    
    # Check for aggregation keywords
    agg_keywords = ['sum', 'total', 'average', 'mean', 'count', 'minimum', 'maximum']
    if any(word in query.lower() for word in agg_keywords):
        complexity['requires_aggregation'] = True
        complexity['level'] = 'medium'
    
    # Check for filtering keywords
    filter_keywords = ['where', 'for', 'in', 'with', 'between', 'after', 'before']
    if any(word in query.lower() for word in filter_keywords):
        complexity['requires_filtering'] = True
        complexity['level'] = 'medium'
    
    # Check for time ranges
    time_keywords = ['month', 'quarter', 'year', 'daily', 'weekly', 'monthly']
    if any(word in query.lower() for word in time_keywords):
        complexity['level'] = 'complex'
    
    return complexity