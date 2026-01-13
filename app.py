import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from datetime import datetime
from io import BytesIO
import networkx as nx
import itertools

# -----------------------------------------------------------------------------
# 1. ENTERPRISE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. ADVANCED CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        color: #333;
    }

    :root {
        --primary-navy: #002060;
        --secondary-gold: #C5AD68;
    }

    h1, h2, h3 { color: #002060; font-weight: 700; letter-spacing: -0.5px; }
    h4, h5, h6 { color: #555; font-weight: 600; }

    /* Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 5px solid #C5AD68;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        border-radius: 8px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        color: #002060;
        font-weight: 800;
    }
    
    /* Automated Insight Box */
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #002060;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 15px;
        line-height: 1.6;
        color: #444;
    }
    .insight-highlight {
        color: #002060;
        font-weight: bold;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 2px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #555;
        font-weight: 600;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #002060;
        color: #002060;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. ROBUST DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    extensions = ['*.csv', '*.xlsx', '*.xls']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join("data", ext)))
    
    if not files:
        return None, "No data file detected in 'data/' folder."

    file_path = files[0]
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        # --- CLEANING ---
        df = df.dropna(axis=1, how='all')
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
        df = df.loc[:, ~df.columns.isin(['nan', ''])]
        
        # Date Handling
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if not date_col:
            return None, "Critical: 'Date' column is missing."
            
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df[date_col].dt.year == 2025]
        
        if df.empty:
            return None, "No data records found for 2025."

        # Standardization
        if 'IF' in df.columns:
            df['IF'] = pd.to_numeric(df['IF'], errors='coerce').fillna(0.0)
            
        text_cols = ['Status', 'Journal.Ranking', 'Authors', 'Journal', 'Type.of.study', 'PI']
        for c in text_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().replace(['nan', 'NaN', 'None', 'NA'], 'N/A')
                if c != 'Journal.Ranking':
                    df[c] = df[c].str.title()
                else:
                    df[c] = df[c].str.upper()

        return df, None, date_col
        
    except Exception as e:
        return None, f"Data Parsing Error: {str(e)}", None

# -----------------------------------------------------------------------------
# 4. ADVANCED VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------
def plot_gauge(value, target, title, suffix=""):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 14, 'color': '#555'}},
        delta = {'reference': target, 'increasing': {'color': "#002060"}, 'decreasing': {'color': "#999"}},
        number = {'suffix': suffix, 'font': {'color': "#002060"}},
        gauge = {
            'axis': {'range': [0, max(target*1.2, value*1.2)], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': "#002060"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#f0f0f0",
            'steps': [
                {'range': [0, target*0.7], 'color': '#e6e6e6'},
                {'range': [target*0.7, target], 'color': '#C5AD68'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target
            }
        }
    ))
    fig.update_layout(height=180, margin=dict(t=30, r=30, l=30, b=10))
    return fig

def generate_smart_insights(df, date_col, total_pubs, avg_if):
    insights = []
    
    # Volume Insight
    insights.append(f"We have processed <span class='insight-highlight'>{total_pubs}</span> research projects in FY2025.")
    
    # High Impact Insight
    if 'Journal.Ranking' in df.columns:
        q1_count = len(df[df['Journal.Ranking'] == 'Q1'])
        insights.append(f"Quality is high, with <span class='insight-highlight'>{q1_count}</span> papers published in top-tier (Q1) journals.")
        
    # Top Author
    auth_col = 'PI' if 'PI' in df.columns else ('Authors' if 'Authors' in df.columns else None)
    if auth_col:
        top_auth = df[auth_col].mode().iloc[0]
        count = len(df[df[auth_col] == top_auth])
        insights.append(f"The leading contributor is <span class='insight-highlight'>{top_auth}</span> with {count} active projects.")
        
    # Momentum (Month)
    if date_col:
        top_month_idx = df[date_col].dt.month.mode().iloc[0]
        top_month_name = datetime(2025, top_month_idx, 1).strftime('%B')
        insights.append(f"Output peaked in <span class='insight-highlight'>{top_month_name}</span>.")

    return " ".join(insights)

def plot_network_graph(df):
    if 'Authors' not in df.columns:
        return None
        
    G = nx.Graph()
    for authors_str in df['Authors'].dropna():
        authors = [a.strip() for a in str(authors_str).split(',') if len(a) > 2]
        if len(authors) > 1:
            for u, v in itertools.combinations(authors, 2):
                if G.has_edge(u, v): G[u][v]['weight'] += 1
                else: G.add_edge(u, v, weight=1)
    
    if len(G.nodes) == 0: return None
    
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(len(G.adj[node]) * 5 + 5)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=False, color='#002060', size=node_size, line_width=1))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# -----------------------------------------------------------------------------
# 5. DASHBOARD UI
# -----------------------------------------------------------------------------
def main():
    # HEADER (Strictly No Emojis, Clean Layout)
    c1, c2, c3 = st.columns([1, 8, 1])
    with c1:
        if os.path.exists("assets/logo_left.png"): st.image("assets/logo_left.png", use_container_width=True)
    with c2:
        st.markdown("<h1 style='text-align: center; margin:0;'>RESEARCH PERFORMANCE DASHBOARD</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #666; margin:0;'>Fiscal Year 2025 Executive Report</h4>", unsafe_allow_html=True)
    with c3:
        if os.path.exists("assets/logo_right.png"): st.image("assets/logo_right.png", use_container_width=True)
            
    st.divider()
    
    df, error_msg, date_col = load_data()
    
    if df is None:
        st.error(error_msg)
        return

    # SIDEBAR
    st.sidebar.markdown("### STRATEGIC TARGETS")
    target_pubs = st.sidebar.number_input("Publication Goal", min_value=1, value=50)
    target_if = st.sidebar.number_input("Impact Factor Goal", min_value=1.0, value=100.0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### FILTER CONTROLS")
    
    # UX: Date Range Picker (Advanced)
    if date_col:
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    
    status_opts = sorted(df['Status'].unique()) if 'Status' in df.columns else []
    sel_status = st.sidebar.multiselect("Status", status_opts, default=status_opts)
    
    rank_opts = sorted(df['Journal.Ranking'].unique()) if 'Journal.Ranking' in df.columns else []
    sel_rank = st.sidebar.multiselect("Journal Ranking", rank_opts, default=rank_opts)
    
    pi_col = 'PI' if 'PI' in df.columns else ('Authors' if 'Authors' in df.columns else None)
    sel_pi = []
    if pi_col:
        pi_opts = sorted(df[pi_col].unique())
        sel_pi = st.sidebar.multiselect("Principal Investigator", pi_opts, default=[])

    # FILTERING
    mask = pd.Series(True, index=df.index)
    if 'Status' in df.columns and sel_status: mask &= df['Status'].isin(sel_status)
    if 'Journal.Ranking' in df.columns and sel_rank: mask &= df['Journal.Ranking'].isin(sel_rank)
    if pi_col and sel_pi: mask &= df[pi_col].isin(sel_pi)
    
    # Date Filtering
    if date_col and len(date_range) == 2:
        mask &= (df[date_col].dt.date >= date_range[0]) & (df[date_col].dt.date <= date_range[1])
    
    df_filtered = df.loc[mask]
    
    # -------------------------------------------------------------------------
    # MAIN TABS (CLEAN TITLES)
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["EXECUTIVE OVERVIEW", "TRENDS & ANALYSIS", "COLLABORATION NET", "DATA REGISTRY"])

    COLOR_NAVY = '#002060'
    COLOR_GOLD = '#C5AD68'
    
    # --- TAB 1: EXECUTIVE OVERVIEW ---
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1. SMART INSIGHTS
        total = len(df_filtered)
        avg_if = df_filtered['IF'].mean() if 'IF' in df_filtered.columns and total > 0 else 0
        
        insight_text = generate_smart_insights(df_filtered, date_col, total, avg_if)
        st.markdown(f"""
        <div class="insight-box">
            <strong>AI SUMMARY:</strong><br>
            {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
        # 2. GAUGES & METRICS
        if_sum = df_filtered['IF'].sum() if 'IF' in df_filtered.columns else 0
        
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.plotly_chart(plot_gauge(total, target_pubs, "Total Output"), use_container_width=True)
        with m2: st.plotly_chart(plot_gauge(if_sum, target_if, "Cumulative IF"), use_container_width=True)
            
        with m3:
            st.metric("Avg Impact Factor", f"{avg_if:.2f}")
            st.markdown("<br>", unsafe_allow_html=True)
            top_q = len(df_filtered[df_filtered['Journal.Ranking']=='Q1']) if 'Journal.Ranking' in df_filtered.columns else 0
            st.metric("Q1 Papers", top_q)

        with m4:
            acc_count = 0
            if 'Status' in df_filtered.columns:
                acc_count = df_filtered['Status'].apply(lambda x: 1 if any(s in str(x) for s in ['Accepted', 'Published']) else 0).sum()
            acc_rate = (acc_count / total * 100) if total > 0 else 0
            st.metric("Acceptance Rate", f"{acc_rate:.1f}%")
            
        st.markdown("---")
        
        r1c1, r1c2 = st.columns([1, 1])
        with r1c1:
            st.markdown("##### Portfolio Composition")
            if 'Status' in df_filtered.columns and 'Type.of.study' in df_filtered.columns:
                sun_data = df_filtered.groupby(['Status', 'Type.of.study']).size().reset_index(name='Count')
                fig_sun = px.sunburst(sun_data, path=['Status', 'Type.of.study'], values='Count', 
                                      color='Status', color_discrete_sequence=[COLOR_NAVY, COLOR_GOLD, '#405887', '#998852'])
                fig_sun.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350)
                st.plotly_chart(fig_sun, use_container_width=True)
        
        with r1c2:
            st.markdown("##### Pipeline Progression")
            if 'Status' in df_filtered.columns:
                s_counts = df_filtered['Status'].value_counts().reset_index()
                s_counts.columns = ['Status', 'Count']
                fig_s = px.bar(s_counts, y='Status', x='Count', orientation='h', text='Count')
                fig_s.update_traces(marker_color=COLOR_NAVY, textposition='outside')
                fig_s.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
                st.plotly_chart(fig_s, use_container_width=True)

    # --- TAB 2: TRENDS & ANALYSIS ---
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        t_c1, t_c2 = st.columns(2)
        
        with t_c1:
            st.markdown("##### Monthly Output Trajectory")
            if date_col:
                trend_df = df_filtered.copy()
                trend_df['Month'] = trend_df[date_col].dt.strftime('%b')
                trend_df['Month_Index'] = trend_df[date_col].dt.month
                monthly = trend_df.groupby(['Month_Index', 'Month']).size().reset_index(name='Publications')
                monthly = monthly.sort_values('Month_Index')
                
                fig_line = px.line(monthly, x='Month', y='Publications', markers=True, line_shape='spline')
                fig_line.update_traces(line_color=COLOR_GOLD, line_width=4, marker_color=COLOR_NAVY)
                fig_line.update_layout(yaxis=dict(range=[0, monthly['Publications'].max()*1.2]))
                st.plotly_chart(fig_line, use_container_width=True)
                
        with t_c2:
            st.markdown("##### Impact Factor vs. Ranking")
            if 'Journal.Ranking' in df_filtered.columns and 'IF' in df_filtered.columns:
                fig_box = px.box(df_filtered, x='Journal.Ranking', y='IF', points="all", color='Journal.Ranking',
                                 color_discrete_map={'Q1':COLOR_NAVY, 'Q2':COLOR_GOLD})
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Leading Contributors (Volume vs Impact)")
        if pi_col and 'IF' in df_filtered.columns:
            df_auth = df_filtered.copy()
            df_auth['Display_Author'] = df_auth[pi_col].apply(lambda x: (x[:20] + '...') if len(str(x)) > 20 else str(x))
            auth_stats = df_auth.groupby('Display_Author').agg(Publications=(pi_col, 'count'), Total_IF=('IF', 'sum')).reset_index().sort_values('Publications', ascending=False).head(12)
            fig_combo = go.Figure()
            fig_combo.add_trace(go.Bar(x=auth_stats['Display_Author'], y=auth_stats['Publications'], name='Publications', marker_color=COLOR_NAVY))
            fig_combo.add_trace(go.Scatter(x=auth_stats['Display_Author'], y=auth_stats['Total_IF'], name='Cum. IF', yaxis='y2', mode='lines+markers', line=dict(color=COLOR_GOLD, width=3)))
            fig_combo.update_layout(yaxis=dict(showgrid=False), yaxis2=dict(overlaying='y', side='right', showgrid=False), legend=dict(orientation='h', y=1.1), height=500)
            st.plotly_chart(fig_combo, use_container_width=True)

    # --- TAB 3: COLLABORATION NETWORK ---
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Co-Authorship Network Graph")
        network_fig = plot_network_graph(df_filtered)
        if network_fig: st.plotly_chart(network_fig, use_container_width=True, height=600)
        else: st.warning("Insufficient author data to generate network graph.")

    # --- TAB 4: DATA REGISTRY ---
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        col_ex, _ = st.columns([1, 4])
        with col_ex:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_filtered.to_excel(writer, index=False, sheet_name='2025_Data')
            st.download_button(label="Download Excel Report", data=buffer, file_name=f"Report_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.ms-excel", type="primary")
        
        # UX: Interactive Table with Column Config
        st.dataframe(
            df_filtered, 
            use_container_width=True,
            column_config={
                "IF": st.column_config.ProgressColumn("Impact Factor", format="%.2f", min_value=0, max_value=df_filtered['IF'].max()),
                "Date": st.column_config.DateColumn("Publication Date")
            }
        )

if __name__ == "__main__":
    main()