# Research Performance Dashboard 2025

**System Status**: Production Ready | **Version**: 2.0 (Enterprise Edition)

---

## 1. System Overview

The **Research Performance Dashboard** is a specialized analytics platform engineered for the **Department of Otolaryngology-Head & Neck Surgery**. It serves as the central intelligence hub for tracking, visualizing, and exporting research productivity data for the fiscal year 2025.

Designed for executive leadership, this application transforms raw publication data into actionable insights through a multi-layered interface, strictly adhering to the "Navy Blue & Gold" corporate identity.

### Key Capabilities

- **Executive KPIs**: Real-time tracking of Total Output, Cumulative Impact Factor, and Acceptance Rates.
- **Trend Analysis**: Spline-interpolated trajectory of monthly research output.
- **Quality Assessment**: Dual-axis analysis of Principal Investigators (Volume vs. Impact).
- **Data Sovereignty**: Advanced Excel (`.xlsx`) export engine for offline reporting.

---

## 2. Analytics Modules

The application is structured into three distinct operational modules (Tabs):

### 📊 Module 1: Executive Overview

High-level summary for quick decision making.

- **KPI Cards**: 4-Point metric system (Output, Cum. IF, Avg. IF, Acceptance Rate).
- **Quality Donut**: Quartile-based (Q1-Q4) distribution of journal quality.
- **Pipeline Funnel**: Visualization of research phases from 'Proposal' to 'Published'.

### 📈 Module 2: Trends & Analysis

Deep-dive analytics to identify patterns and leaders.

- **Monthly Momentum**: Time-series visualization of publication frequency.
- **Leaderboard Matrix**: Combo-chart contrasting an author's _Publication Count_ (Bar) against their _Cumulative Impact Factor_ (Line), identifying high-impact contributors.

### 📋 Module 3: Data Registry

Granular data management.

- **Searchable Grid**: Full interactive access to the underlying dataset.
- **Report Generation**: One-click generation of formatted Excel reports.

---

## 3. Technical Specifications

### 3.1 Architecture

- **Frontend**: Streamlit (v1.30+)
- **Visualization**: Plotly Express & Graph Objects (Interactive)
- **Data Engine**: Pandas (Vectorized processing)
- **Export Engine**: XlsxWriter

### 3.2 Directory Structure

```
Research_Dashboard/
├── .streamlit/
│   └── config.toml      # strict theme configuration
├── assets/
│   ├── logo_left.png    # institutional branding
│   └── logo_right.png   # departmental branding
├── data/
│   └── [DATA_FILE]      # auto-detected .csv or .xlsx
├── app.py               # core application logic
├── requirements.txt     # environment dependencies
└── README.md            # system documentation
```

---

## 4. Installation & Deployment

### Prerequisite

- Python 3.9+ environment.

### Step 1: initialization

```bash
# Clone repository
git clone [repository_url]

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Provisioning

Place your 2025 data file in the `data/` directory.

- **Supported Formats**: `.csv`, `.xlsx`
- **Required Columns**: `Date`, `Status`
- **Optional Analytics Columns**: `IF`, `Journal.Ranking`, `PI`, `Authors`, `Journal`

### Step 3: Execution

```bash
streamlit run app.py
```

---

## 5. Support Limits

- **Date Filtering**: The system is hardcoded to filter for the year `2025`.
- **Data Cleaning**: "N/A" values are automatically handled; numerical errors in 'IF' are coerced to 0.0 to prevent system crashes.

---

**CONFIDENTIAL | INTERNAL USE ONLY**
