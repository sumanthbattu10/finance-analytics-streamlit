"""
Finance Analytics Streamlit App
Author: Sumanth Battu
Description: AI-powered finance analytics dashboard
             for Strategic Finance teams
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Finance Analytics | AI-Powered",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def generate_revenue_data() -> pd.DataFrame:
    """Generate sample revenue data for demo"""
    np.random.seed(42)
    months = pd.date_range(
        start='2024-01-01',
        end='2024-12-31',
        freq='MS'
    )

    arr_base = 10_000_000
    data = []
    for i, month in enumerate(months):
        arr = arr_base * (1 + 0.08) ** i
        new_arr = arr * np.random.uniform(0.06, 0.10)
        churned_arr = arr * np.random.uniform(0.01, 0.03)
        expansion_arr = arr * np.random.uniform(0.02, 0.05)
        nrr = (
            (arr + expansion_arr - churned_arr) / arr * 100
        )
        data.append({
            'month': month,
            'arr': arr,
            'new_arr': new_arr,
            'churned_arr': churned_arr,
            'expansion_arr': expansion_arr,
            'nrr': nrr,
            'bookings': new_arr + expansion_arr,
            'customers': int(100 + i * 8),
        })

    return pd.DataFrame(data)


@st.cache_data
def generate_cost_data() -> pd.DataFrame:
    """Generate sample cost data"""
    np.random.seed(42)
    months = pd.date_range(
        start='2024-01-01',
        end='2024-12-31',
        freq='MS'
    )
    categories = [
        'R&D', 'Sales & Marketing',
        'G&A', 'COGS'
    ]
    data = []
    for month in months:
        for cat in categories:
            base = {
                'R&D': 2_000_000,
                'Sales & Marketing': 3_000_000,
                'G&A': 800_000,
                'COGS': 1_500_000
            }[cat]
            data.append({
                'month': month,
                'category': cat,
                'amount': base * np.random.uniform(
                    0.95, 1.05
                )
            })
    return pd.DataFrame(data)


def render_kpi_metrics(df: pd.DataFrame):
    """Render KPI metrics row"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = (
            (latest['arr'] - prev['arr']) /
            prev['arr'] * 100
        )
        st.metric(
            "Annual Recurring Revenue",
            f"${latest['arr']:,.0f}",
            f"{delta:+.1f}% MoM"
        )

    with col2:
        delta = (
            (latest['nrr'] - prev['nrr'])
        )
        st.metric(
            "Net Revenue Retention",
            f"{latest['nrr']:.1f}%",
            f"{delta:+.1f}pp MoM"
        )

    with col3:
        delta = (
            (latest['bookings'] - prev['bookings']) /
            prev['bookings'] * 100
        )
        st.metric(
            "Bookings",
            f"${latest['bookings']:,.0f}",
            f"{delta:+.1f}% MoM"
        )

    with col4:
        delta = latest['customers'] - prev['customers']
        st.metric(
            "Total Customers",
            f"{latest['customers']:,}",
            f"{delta:+.0f} MoM"
        )


def render_revenue_chart(df: pd.DataFrame):
    """Render revenue waterfall chart"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['month'],
        y=df['arr'],
        name='ARR',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))

    fig.add_trace(go.Bar(
        x=df['month'],
        y=df['new_arr'],
        name='New ARR',
        marker_color='#48bb78'
    ))

    fig.add_trace(go.Bar(
        x=df['month'],
        y=-df['churned_arr'],
        name='Churned ARR',
        marker_color='#fc8181'
    ))

    fig.update_layout(
        title='Revenue Waterfall Analysis',
        xaxis_title='Month',
        yaxis_title='Revenue ($)',
        barmode='relative',
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def render_nrr_chart(df: pd.DataFrame):
    """Render NRR trend chart"""
    fig = px.line(
        df,
        x='month',
        y='nrr',
        title='Net Revenue Retention Trend',
        labels={'nrr': 'NRR (%)', 'month': 'Month'}
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="red",
        annotation_text="100% baseline"
    )

    fig.update_traces(
        line=dict(color='#667eea', width=3)
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_ai_query_interface():
    """Render AI natural language query interface"""
    st.subheader("🤖 AI Finance Agent")
    st.caption(
        "Ask questions in plain English — "
        "powered by prompt-engineered AI agents"
    )

    example_queries = [
        "What drove the QoQ change in ARR?",
        "Show me the top 3 cost categories",
        "What is our NRR trend over the last 6 months?",
        "How does expansion ARR compare to new ARR?"
    ]

    selected = st.selectbox(
        "Try an example query:",
        ["Select a query..."] + example_queries
    )

    query = st.text_input(
        "Or type your own:",
        value=selected if selected != "Select a query..."
        else ""
    )

    if st.button("Ask AI Agent", type="primary"):
        if query:
            with st.spinner(
                "AI agent processing your query..."
            ):
                # Simulate AI agent response
                st.success("✅ AI Agent Response:")
                st.markdown(f"""
**Query:** {query}

**Analysis:** Based on the revenue data analysis:
- Current ARR shows strong growth momentum
  with 8% MoM compound growth rate
- NRR remains above 110% indicating healthy
  expansion revenue from existing customers
- New bookings are trending upward with
  consistent customer acquisition
- Churn rate is within acceptable bounds
  at 1-3% of ARR monthly

**Recommendation:** Continue current growth
trajectory while monitoring churn signals
for early intervention opportunities.

*Generated by AI Finance Agent v1.0*
                """)
        else:
            st.warning("Please enter a query first")


def main():
    """Main app entry point"""
    # Header
    st.title("📊 Finance Analytics Dashboard")
    st.caption(
        "AI-powered | Built with Streamlit + "
        "Snowflake + dbt | Strategic Finance"
    )

    # Load data
    revenue_df = generate_revenue_data()
    cost_df = generate_cost_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    date_range = st.sidebar.selectbox(
        "Time Period",
        ["Last 3 months", "Last 6 months",
         "Last 12 months", "YTD"]
    )

    metric_view = st.sidebar.multiselect(
        "Metrics to Display",
        ["ARR", "NRR", "Bookings", "Churn"],
        default=["ARR", "NRR", "Bookings"]
    )

    # Main content
    tab1, tab2, tab3 = st.tabs([
        "📈 Revenue Analytics",
        "🤖 AI Agent",
        "📊 Cost Analysis"
    ])

    with tab1:
        st.subheader("Revenue KPIs")
        render_kpi_metrics(revenue_df)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            render_revenue_chart(revenue_df)
        with col2:
            render_nrr_chart(revenue_df)

        st.divider()
        st.subheader("Revenue Detail Table")
        display_df = revenue_df.copy()
        display_df['month'] = display_df[
            'month'
        ].dt.strftime('%Y-%m')
        display_df['arr'] = display_df['arr'].apply(
            lambda x: f"${x:,.0f}"
        )
        display_df['nrr'] = display_df['nrr'].apply(
            lambda x: f"{x:.1f}%"
        )
        st.dataframe(
            display_df[[
                'month', 'arr', 'nrr',
                'bookings', 'customers'
            ]],
            use_container_width=True
        )

    with tab2:
        render_ai_query_interface()

    with tab3:
        st.subheader("Cost Analysis")
        fig = px.bar(
            cost_df.groupby(
                'category'
            )['amount'].sum().reset_index(),
            x='category',
            y='amount',
            title='Total Cost by Category',
            color='category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cost Trend Over Time")
        fig2 = px.line(
            cost_df,
            x='month',
            y='amount',
            color='category',
            title='Monthly Cost Trends'
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
