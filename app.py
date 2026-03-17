"""
Conversational AI for Instant Business Intelligence Dashboards
Amazon Sales Analytics - Main Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from datetime import datetime
import hashlib
import os
from dotenv import load_dotenv
import google.genai as genai
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Load environment variables
load_dotenv('config.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class ChartType(Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX = "box"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    FUNNEL = "funnel"
    GANTT = "gantt"

@dataclass
class VisualizationSuggestion:
    chart_type: ChartType
    title: str
    x_axis: str
    y_axis: List[str]
    aggregation: str
    description: str
    color_by: Optional[str] = None
    facet_by: Optional[str] = None
    additional_params: Dict = None

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_dashboard' not in st.session_state:
        st.session_state.current_dashboard = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'data_context' not in st.session_state:
        st.session_state.data_context = None
    if 'chart_cache' not in st.session_state:
        st.session_state.chart_cache = {}

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Load and preprocess the Amazon sales dataset"""
    try:
        # Read the CSV file
        df = pd.read_csv('sales_data.csv')
        # Basic preprocessing
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['month'] = df['order_date'].dt.month
        df['year'] = df['order_date'].dt.year
        df['quarter'] = df['order_date'].dt.quarter
        df['month_year'] = df['order_date'].dt.strftime('%Y-%m')
        df['profit_margin'] = (df['total_revenue'] - (df['price'] * df['quantity_sold'])) / df['total_revenue'] * 100
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

def get_data_summary(df: pd.DataFrame) -> Dict:
    """Generate comprehensive data summary for context"""
    return {
        'total_records': len(df),
        'date_range': f"{df['order_date'].min().date()} to {df['order_date'].max().date()}",
        'total_revenue': f"${df['total_revenue'].sum():,.2f}",
        'avg_order_value': f"${df['total_revenue'].mean():,.2f}",
        'total_orders': df['order_id'].nunique(),
        'total_products': df['product_id'].nunique(),
        'categories': df['product_category'].unique().tolist(),
        'regions': df['customer_region'].unique().tolist(),
        'payment_methods': df['payment_method'].unique().tolist(),
        'date_columns': ['order_date', 'month', 'year', 'quarter', 'month_year'],
        'numeric_columns': df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'summary_stats': df.describe().to_dict()
    }

# ============================================================================
# LLM INTEGRATION
# ============================================================================

class LLMQueryProcessor:
    def __init__(self, api_key: str = None):
        """Initialize LLM with Gemini API"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            st.warning("Gemini API key not found. Using rule-based fallback.")
            self.client = None
    
    def generate_system_prompt(self, df_summary: Dict) -> str:
        """Generate comprehensive system prompt with data context"""
        return f"""
        You are an AI Business Intelligence assistant specialized in analyzing e-commerce data.
        
        DATASET CONTEXT:
        - Total Records: {df_summary['total_records']} orders
        - Date Range: {df_summary['date_range']}
        - Total Revenue: {df_summary['total_revenue']}
        - Product Categories: {', '.join(df_summary['categories'])}
        - Customer Regions: {', '.join(df_summary['regions'])}
        - Payment Methods: {', '.join(df_summary['payment_methods'])}
        
        AVAILABLE COLUMNS:
        {json.dumps({
            'numeric': df_summary['numeric_columns'],
            'categorical': df_summary['categorical_columns'],
            'date': df_summary['date_columns']
        }, indent=2)}
        
        TASK: Convert user's natural language query into a structured visualization specification.
        
        RESPONSE FORMAT (JSON):
        {{
            "analysis_type": "exploratory|comparative|trend|distribution|correlation",
            "visualizations": [
                {{
                    "chart_type": "bar|line|pie|scatter|heatmap|area|histogram|box|treemap|sunburst",
                    "title": "descriptive title",
                    "description": "brief insight",
                    "x_axis": "column_name",
                    "y_axis": ["column_name1", "column_name2"],
                    "aggregation": "sum|mean|count|min|max",
                    "color_by": "optional_column",
                    "facet_by": "optional_column",
                    "filters": {{"column": "value"}},
                    "insights": ["key finding 1", "key finding 2"]
                }}
            ],
            "key_metrics": ["metric1", "metric2"],
            "filters_applied": {{}}
        }}
        
        RULES:
        1. For time-series -> use line charts
        2. For comparisons -> use bar charts
        3. For distributions -> use histograms or box plots
        4. For proportions -> use pie charts (max 7 slices)
        5. For correlations -> use scatter plots
        6. For hierarchical -> use treemaps or sunburst
        7. Always include descriptive titles and insights
        8. Suggest 2-4 visualizations per query
        """
    
    def parse_query(self, user_query: str, df_summary: Dict) -> Optional[Dict]:
        """Parse user query using Gemini API or fallback"""
        if not self.client:
            return self.rule_based_parser(user_query, df_summary)
        
        try:
            prompt = f"""
            {self.generate_system_prompt(df_summary)}
            
            USER QUERY: "{user_query}"
            
            Generate JSON visualization specification.
            """
            
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
            
        except Exception as e:
            logger.error(f"LLM parsing error: {e}")
            return self.rule_based_parser(user_query, df_summary)
    
    def rule_based_parser(self, user_query: str, df_summary: Dict) -> Dict:
        """Fallback rule-based query parser"""
        query_lower = user_query.lower()
        visualizations = []
        
        # Time series detection
        if any(word in query_lower for word in ['trend', 'over time', 'monthly', 'quarterly', 'yearly']):
            visualizations.append({
                "chart_type": "line",
                "title": "Revenue Trend Over Time",
                "description": "Monthly revenue trends analysis",
                "x_axis": "month_year",
                "y_axis": ["total_revenue"],
                "aggregation": "sum",
                "insights": ["Revenue trends over the analyzed period"]
            })
        
        # Category comparison
        if any(word in query_lower for word in ['compare', 'category', 'by region', 'by payment']):
            visualizations.append({
                "chart_type": "bar",
                "title": "Revenue by Category",
                "description": "Revenue distribution across product categories",
                "x_axis": "product_category",
                "y_axis": ["total_revenue"],
                "aggregation": "sum",
                "insights": ["Category performance comparison"]
            })
        
        # Distribution analysis
        if any(word in query_lower for word in ['distribution', 'rating', 'discount', 'price']):
            visualizations.append({
                "chart_type": "histogram",
                "title": "Product Rating Distribution",
                "description": "Distribution of product ratings",
                "x_axis": "rating",
                "y_axis": ["count"],
                "aggregation": "count",
                "insights": ["Rating distribution pattern"]
            })
        
        # Default dashboard
        if not visualizations:
            visualizations = [
                {
                    "chart_type": "bar",
                    "title": "Revenue by Product Category",
                    "description": "Top performing categories",
                    "x_axis": "product_category",
                    "y_axis": ["total_revenue"],
                    "aggregation": "sum",
                    "insights": ["Identify top revenue categories"]
                },
                {
                    "chart_type": "line",
                    "title": "Monthly Sales Trend",
                    "description": "Revenue trends over time",
                    "x_axis": "month_year",
                    "y_axis": ["total_revenue"],
                    "aggregation": "sum",
                    "insights": ["Seasonal patterns detection"]
                },
                {
                    "chart_type": "pie",
                    "title": "Payment Method Distribution",
                    "description": "Customer payment preferences",
                    "x_axis": "payment_method",
                    "y_axis": ["total_revenue"],
                    "aggregation": "sum",
                    "insights": ["Popular payment methods"]
                }
            ]
        
        return {
            "analysis_type": "exploratory",
            "visualizations": visualizations,
            "key_metrics": ["total_revenue", "avg_rating"],
            "filters_applied": {}
        }

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class VisualizationEngine:
    """Generate Plotly visualizations from specifications"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, viz_spec: Dict) -> go.Figure:
        """Create appropriate chart based on specification"""
        chart_type = viz_spec.get('chart_type', 'bar')
        x_axis = viz_spec.get('x_axis')
        y_axis = viz_spec.get('y_axis', [])
        aggregation = viz_spec.get('aggregation', 'sum')
        color_by = viz_spec.get('color_by')
        facet_by = viz_spec.get('facet_by')
        
        # Prepare data
        if aggregation != 'none':
            if len(y_axis) > 0:
                agg_dict = {col: aggregation for col in y_axis}
                if x_axis and x_axis in df.columns:
                    grouped_df = df.groupby(x_axis).agg(agg_dict).reset_index()
                else:
                    grouped_df = df.agg(agg_dict).to_frame().T
            else:
                grouped_df = df
        else:
            grouped_df = df
        
        # Create chart based on type
        if chart_type == 'line':
            fig = px.line(grouped_df, x=x_axis, y=y_axis, color=color_by,
                          title=viz_spec.get('title', ''), facet_col=facet_by)
        
        elif chart_type == 'bar':
            fig = px.bar(grouped_df, x=x_axis, y=y_axis, color=color_by,
                         title=viz_spec.get('title', ''), facet_col=facet_by,
                         barmode='group')
        
        elif chart_type == 'pie':
            fig = px.pie(grouped_df, values=y_axis[0] if y_axis else None,
                         names=x_axis, title=viz_spec.get('title', ''))
        
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis[0] if y_axis else None,
                             color=color_by, size=y_axis[1] if len(y_axis) > 1 else None,
                             title=viz_spec.get('title', ''), facet_col=facet_by)
        
        elif chart_type == 'heatmap':
            pivot_df = df.pivot_table(index=x_axis, columns=y_axis[0] if y_axis else None,
                                       values=y_axis[1] if len(y_axis) > 1 else 'total_revenue',
                                       aggfunc=aggregation)
            fig = px.imshow(pivot_df, title=viz_spec.get('title', ''))
        
        elif chart_type == 'area':
            fig = px.area(grouped_df, x=x_axis, y=y_axis, color=color_by,
                          title=viz_spec.get('title', ''), facet_col=facet_by)
        
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_axis, color=color_by,
                               title=viz_spec.get('title', ''), facet_col=facet_by)
        
        elif chart_type == 'box':
            fig = px.box(df, x=x_axis, y=y_axis[0] if y_axis else None,
                         color=color_by, title=viz_spec.get('title', ''), facet_col=facet_by)
        
        elif chart_type == 'treemap':
            fig = px.treemap(df, path=[x_axis] if x_axis else None,
                             values=y_axis[0] if y_axis else None,
                             color=color_by, title=viz_spec.get('title', ''))
        
        elif chart_type == 'sunburst':
            fig = px.sunburst(df, path=[x_axis] if x_axis else None,
                              values=y_axis[0] if y_axis else None,
                              color=color_by, title=viz_spec.get('title', ''))
        
        else:
            # Default to bar chart
            fig = px.bar(grouped_df, x=x_axis, y=y_axis, title=viz_spec.get('title', ''))
        
        # Update layout for better appearance
        fig.update_layout(
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add annotations if insights provided
        insights = viz_spec.get('insights', [])
        if insights:
            insight_text = "<br>".join([f"• {insight}" for insight in insights[:2]])
            fig.add_annotation(
                text=insight_text,
                align='left',
                showarrow=False,
                xref='paper', yref='paper',
                x=0, y=-0.2,
                bordercolor='#e0e0e0',
                borderwidth=1,
                borderpad=10,
                bgcolor='#f8f9fa'
            )
        
        return fig
    
    @staticmethod
    def create_kpi_card(value: float, title: str, delta: float = None, format_str: str = "${:,.2f}"):
        """Create KPI metric card"""
        formatted_value = format_str.format(value) if format_str else str(value)
        delta_str = f"{delta:+.1%}" if delta else None
        return st.metric(title, formatted_value, delta_str)

# ============================================================================
# DASHBOARD BUILDER
# ============================================================================

class DashboardBuilder:
    """Build interactive dashboard from visualization specifications"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.viz_engine = VisualizationEngine()
    
    def build_dashboard(self, viz_specs: Dict) -> None:
        """Construct complete dashboard layout"""
        
        # Dashboard header
        st.markdown("""
        <style>
        .dashboard-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        .insight-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="dashboard-header">
            <h1>📊 {viz_specs.get('analysis_type', 'Exploratory').title()} Dashboard</h1>
            <p>Generated from your query: <i>"{st.session_state.current_query}"</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Row
        if viz_specs.get('key_metrics'):
            st.subheader("🎯 Key Performance Indicators")
            cols = st.columns(len(viz_specs['key_metrics']))
            for i, metric in enumerate(viz_specs['key_metrics']):
                if metric in self.df.columns:
                    value = self.df[metric].sum() if metric == 'total_revenue' else self.df[metric].mean()
                    with cols[i]:
                        self.viz_engine.create_kpi_card(value, metric.replace('_', ' ').title())
        
        # Filters Section
        filters_applied = viz_specs.get('filters_applied', {})
        if filters_applied:
            with st.expander("🔍 Applied Filters", expanded=False):
                for key, value in filters_applied.items():
                    st.info(f"**{key}:** {value}")
        
        # Visualizations Grid
        visualizations = viz_specs.get('visualizations', [])
        
        # Create responsive grid layout
        if len(visualizations) == 1:
            cols = [st.container()]
        elif len(visualizations) == 2:
            cols = st.columns(2)
        else:
            # Create 2-column layout for 3-4 charts
            cols = st.columns(2)
        
        for idx, viz_spec in enumerate(visualizations):
            # Calculate position in grid
            if len(visualizations) <= 2:
                with cols[idx]:
                    self._render_chart_card(viz_spec, idx)
            else:
                # Handle multi-row layout
                row_idx = idx // 2
                col_idx = idx % 2
                
                if col_idx == 0:
                    row_cols = st.columns(2)
                
                with row_cols[col_idx]:
                    self._render_chart_card(viz_spec, idx)
        
        # Insights Summary
        all_insights = []
        for viz_spec in visualizations:
            all_insights.extend(viz_spec.get('insights', []))
        
        if all_insights:
            st.markdown("---")
            st.subheader("💡 Key Insights")
            with st.container():
                for i, insight in enumerate(all_insights[:5]):
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong>Insight {i+1}:</strong> {insight}
                    </div>
                    """, unsafe_allow_html=True)
    
    def _render_chart_card(self, viz_spec: Dict, idx: int):
        """Render individual chart with card styling"""
        with st.container():
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            ">
            """, unsafe_allow_html=True)
            
            # Generate and display chart
            fig = self.viz_engine.create_chart(self.df, viz_spec)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
            
            # Chart description
            if viz_spec.get('description'):
                st.caption(f"📌 {viz_spec['description']}")
            
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# CHAT INTERFACE
# ============================================================================

class ChatInterface:
    """Handle chat interactions and query processing"""
    
    def __init__(self, df: pd.DataFrame, llm_processor: LLMQueryProcessor):
        self.df = df
        self.llm = llm_processor
        self.dashboard_builder = DashboardBuilder(df)
    
    def process_query(self, user_query: str) -> Tuple[bool, str]:
        """Process user query and generate dashboard"""
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Parse query
            status_text.text("🤔 Understanding your query...")
            progress_bar.progress(25)
            
            viz_specs = self.llm.parse_query(user_query, get_data_summary(self.df))
            
            if not viz_specs:
                return False, "Could not understand the query. Please try rephrasing."
            
            # Step 2: Apply any filters
            status_text.text("🔍 Applying filters...")
            progress_bar.progress(50)
            
            filtered_df = self.df.copy()
            filters = viz_specs.get('filters_applied', {})
            for col, value in filters.items():
                if col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[col] == value]
            
            # Step 3: Generate visualizations
            status_text.text("📊 Creating visualizations...")
            progress_bar.progress(75)
            
            # Store in session state
            st.session_state.current_dashboard = viz_specs
            st.session_state.current_query = user_query
            st.session_state.filtered_data = filtered_df
            
            # Step 4: Build dashboard
            status_text.text("✨ Building dashboard...")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Render dashboard
            self.dashboard_builder.build_dashboard(viz_specs)
            
            # Add to history
            st.session_state.query_history.append({
                'query': user_query,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'viz_count': len(viz_specs.get('visualizations', []))
            })
            
            return True, "Dashboard generated successfully!"
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            progress_bar.empty()
            status_text.empty()
            return False, f"Error: {str(e)}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Amazon Sales BI Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .sidebar-info {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Amazon Sales BI Assistant</h1>
        <p>Turn natural language into powerful interactive dashboards</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/amazon.png", width=80)
        st.title("📁 Data Source")
        
        # Data upload option
        data_source = st.radio(
            "Choose data source:",
            ["Use Default Dataset", "Upload Custom CSV"]
        )
        
        df = None
        
        if data_source == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"✅ Loaded {len(df)} records")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            else:
                df = load_default_data()
        else:
            df = load_default_data()
        
        if df is not None:
            # Data overview
            st.markdown("---")
            st.subheader("📊 Data Overview")
            
            summary = get_data_summary(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", summary['total_records'])
                st.metric("Total Revenue", summary['total_revenue'])
            with col2:
                st.metric("Categories", len(summary['categories']))
                st.metric("Regions", len(summary['regions']))
            
            # Quick filters
            st.markdown("---")
            st.subheader("🔍 Quick Filters")
            
            selected_category = st.multiselect(
                "Product Category",
                options=summary['categories'],
                default=[]
            )
            
            selected_region = st.multiselect(
                "Customer Region",
                options=summary['regions'],
                default=[]
            )
            
            # Query history
            st.markdown("---")
            st.subheader("📜 Query History")
            
            if st.session_state.query_history:
                for item in reversed(st.session_state.query_history[-5:]):
                    with st.expander(f"🔍 {item['query'][:30]}..."):
                        st.caption(f"Time: {item['timestamp']}")
                        st.caption(f"Visualizations: {item['viz_count']}")
            else:
                st.info("No queries yet")
    
    # Main content area
    if df is not None:
        # Initialize processors
        llm_processor = LLMQueryProcessor()
        chat_interface = ChatInterface(df, llm_processor)
        
        # Query input
        st.markdown("### 💬 Ask about your data")
        
        col1, col2 = st.columns([6, 1])
        with col1:
            user_query = st.text_input(
                "Enter your question in plain English:",
                placeholder="e.g., Show me monthly sales trends by region for top product categories",
                label_visibility="collapsed"
            )
        with col2:
            submit_button = st.button("🚀 Generate", type="primary", use_container_width=True)
        
        # Example queries
        with st.expander("📝 Example queries to try"):
            example_cols = st.columns(3)
            examples = [
                "Show me monthly revenue trends for 2023",
                "Compare sales performance across regions",
                "What are the top 10 products by revenue?",
                "Show discount distribution by category",
                "Analyze customer ratings over time",
                "Display payment method preferences by region"
            ]
            
            for i, example in enumerate(examples):
                with example_cols[i % 3]:
                    if st.button(f"📋 {example}", key=f"example_{i}"):
                        user_query = example
                        submit_button = True
        
        # Process query
        if submit_button and user_query:
            with st.spinner("🤖 Analyzing your query..."):
                success, message = chat_interface.process_query(user_query)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Display current dashboard
        elif st.session_state.current_dashboard:
            # Rebuild dashboard from session state
            filtered_df = st.session_state.get('filtered_data', df)
            dashboard_builder = DashboardBuilder(filtered_df)
            dashboard_builder.build_dashboard(st.session_state.current_dashboard)
        
        else:
            # Welcome screen
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2>👋 Welcome to Amazon Sales BI Assistant</h2>
                <p style="color: #666; font-size: 1.2rem;">
                    Ask questions about your sales data in plain English and get instant, 
                    interactive visualizations. No SQL or technical skills required!
                </p>
                <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 2rem;">
                    <div style="background: #f0f0f0; padding: 1rem; border-radius: 8px; width: 200px;">
                        <h3>📈 Trends</h3>
                        <p>Monthly sales, growth rates</p>
                    </div>
                    <div style="background: #f0f0f0; padding: 1rem; border-radius: 8px; width: 200px;">
                        <h3>💰 Revenue</h3>
                        <p>By category, region, payment</p>
                    </div>
                    <div style="background: #f0f0f0; padding: 1rem; border-radius: 8px; width: 200px;">
                        <h3>⭐ Ratings</h3>
                        <p>Product performance, reviews</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Unable to load data. Please check your data source.")

if __name__ == "__main__":
    main()