import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Page config
st.set_page_config(
    page_title="Easy Inventory Manager",
    page_icon="📦",
    layout="wide"
)

# Custom CSS for better visual appeal
st.markdown("""
    <style>
    .tooltip {
        color: #666;
        font-size: 14px;
        font-style: italic;
    }
    .big-number {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to show tooltips
def show_help(text):
    st.markdown(f'<div class="tooltip">💡 {text}</div>', unsafe_allow_html=True)

# Sidebar with simple navigation
st.sidebar.title("📋 Menu")
page = st.sidebar.radio("What would you like to see?", 
    ["Home & Overview", 
     "Product Analysis", 
     "Supplier Information", 
     "Lead Time & Procurement",
     "Weather Impact Analysis",
     "Future Predictions"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.models = {}

def load_data():
    """Load and preprocess the inventory data"""
    if st.session_state.data is None:
        try:
            df = pd.read_csv('data/retail_store_inventory_with_vendors.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Calculate easy-to-understand metrics
            df['DaysToStockout'] = np.ceil(df['Inventory Level'] / df['Units Sold'].where(df['Units Sold'] > 0, 1))
            df['StockoutRisk'] = (df['Inventory Level'] < df['Demand Forecast']).astype(int)
            df['Stock Status'] = pd.cut(
                df['DaysToStockout'],
                bins=[-np.inf, 7, 14, np.inf],
                labels=['Critical', 'Warning', 'Good']
            )
            df['Daily Sales'] = df['Units Sold']
            df['Total Value'] = df['Inventory Level'] * df['Price']
            
            # Lead Time Analysis calculations
            df['Procurement Lead Time'] = df['Vendor Avg Delivery Days']
            df['Safety Stock Days'] = df['Procurement Lead Time'] * 1.5  # 1.5x lead time as safety buffer
            df['Reorder Point'] = df['Units Sold'] * (df['Procurement Lead Time'] + df['Safety Stock Days'])
            df['Lead Time Risk'] = (df['Inventory Level'] < df['Reorder Point']).astype(int)
            
            # Improved Suggested Order calculation
            # Target stock level = 30 days of sales + safety buffer
            df['Target Stock Level'] = df['Units Sold'] * 30 + df['Reorder Point']
            df['Suggested Order Qty'] = np.maximum(
                0, 
                np.where(
                    df['DaysToStockout'] <= 7,  # Critical items
                    df['Target Stock Level'] - df['Inventory Level'],  # Bring to target level
                    np.maximum(
                        df['Reorder Point'] - df['Inventory Level'],  # At minimum bring to reorder point
                        df['Units Sold'] * 14  # Or 14 days worth of stock
                    )
                )
            )
            
            # Weather impact simulation (product-specific sensitivity)
            df['Month'] = df['Date'].dt.month
            df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                                         9: 'Fall', 10: 'Fall', 11: 'Fall'})
            
            # Create product-specific weather sensitivity based on category
            category_weather_sensitivity = {
                'Electronics': {'Winter': 1.2, 'Spring': 1.0, 'Summer': 1.1, 'Fall': 1.0},
                'Clothing': {'Winter': 1.4, 'Spring': 1.1, 'Summer': 0.8, 'Fall': 1.2},
                'Toys': {'Winter': 1.3, 'Spring': 1.0, 'Summer': 1.5, 'Fall': 1.1},
                'Furniture': {'Winter': 0.9, 'Spring': 1.3, 'Summer': 1.2, 'Fall': 1.0},
                'Groceries': {'Winter': 1.1, 'Spring': 1.0, 'Summer': 0.95, 'Fall': 1.05}
            }
            
            # Apply category-specific weather multipliers
            def get_weather_multiplier(row):
                category = row['Category']
                season = row['Season']
                if category in category_weather_sensitivity:
                    return category_weather_sensitivity[category].get(season, 1.0)
                else:
                    # Default seasonal pattern for unknown categories
                    return {'Winter': 1.1, 'Spring': 1.0, 'Summer': 1.2, 'Fall': 1.0}.get(season, 1.0)
            
            df['Weather Multiplier'] = df.apply(get_weather_multiplier, axis=1)
            df['Weather Adjusted Demand'] = df['Demand Forecast'] * df['Weather Multiplier']
            df['Weather Impact Score'] = (df['Weather Multiplier'] - 1) * 100  # Percentage impact
            
            st.session_state.data = df
            return df
        except Exception as e:
            st.error("Oops! There was a problem loading your data. Please make sure your file is in the correct location and format.")
            return None
    return st.session_state.data

def home_page():
    st.title("📦 Easy Inventory Manager")
    st.write("Welcome! Let's help you manage your inventory the simple way.")
    
    df = load_data()
    if df is None:
        return

    # Simple Status Overview
    st.header("Quick Overview")
    show_help("This section shows you the most important numbers at a glance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        low_stock = len(df[df['DaysToStockout'] <= 7])
        st.markdown("### 🚨 Needs Attention")
        st.markdown(f'<p class="big-number">{low_stock} items</p>', unsafe_allow_html=True)
        show_help("Items with 7 days or less of stock remaining. These require immediate reordering to prevent stockouts. Critical threshold based on current daily sales rate.")

    with col2:
        watch_items = len(df[(df['DaysToStockout'] > 7) & (df['DaysToStockout'] <= 14)])
        st.markdown("### ⚠️ Watch List")
        st.markdown(f'<p class="big-number">{watch_items} items</p>', unsafe_allow_html=True)
        show_help("Items with 8-14 days of stock remaining. Plan reorders within the next week to maintain healthy inventory levels.")

    with col3:
        healthy_items = len(df[df['DaysToStockout'] > 14])
        st.markdown("### ✅ Healthy Stock")
        st.markdown(f'<p class="big-number">{healthy_items} items</p>', unsafe_allow_html=True)
        show_help("Items with more than 14 days of stock. These products have adequate inventory levels for normal operations.")

    with col4:
        total_value = (df['Inventory Level'] * df['Price']).sum()
        st.markdown("### 💰 Total Value")
        st.markdown(f'<p class="big-number">${total_value:,.0f}</p>', unsafe_allow_html=True)
        show_help("Total monetary value of all inventory on hand. This represents your working capital tied up in stock.")
    
    # Items that need attention
    st.header("🚨 Items That Need Your Attention")
    show_help("Critical inventory items requiring immediate action. Suggested orders are calculated based on target stock levels (30 days supply + safety buffer) minus current inventory.")
    
    critical_items = df[df['DaysToStockout'] <= 7].sort_values('DaysToStockout')
    if not critical_items.empty:
        for _, row in critical_items.head().iterrows():
            with st.expander(f"🔴 {row['Product ID']} - {row['Category']} - {int(row['DaysToStockout'])} days of stock left"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Current Stock:** {int(row['Inventory Level'])} units")
                    st.write(f"**Daily Sales Rate:** {int(row['Units Sold'])} units/day")
                    st.write(f"**Target Stock Level:** {int(row['Target Stock Level'])} units")
                with col2:
                    st.write(f"**Suggested Order:** {int(row['Suggested Order Qty'])} units")
                    st.write(f"**Lead Time:** {int(row['Vendor Avg Delivery Days'])} days")
                    st.write(f"**Best Supplier:** {row['Vendor']}")
                    
                # Add calculation explanation
                st.info(f"""
                **Order Calculation:** Target stock ({int(row['Target Stock Level'])} units) - Current stock ({int(row['Inventory Level'])} units) = {int(row['Suggested Order Qty'])} units needed.
                Target stock = 30 days supply + safety buffer for lead time coverage.
                """)
    else:
        st.success("Great news! No items need immediate attention.")

    # Lead Time & Weather Alerts
    st.header("⚡ Procurement & Weather Alerts")
    show_help("Real-time alerts for procurement timing and weather-based demand changes. Lead time alerts show items below calculated reorder points, while weather impact shows category-specific seasonal adjustments.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚚 Lead Time Alerts")
        lead_time_risk_items = df[df['Lead Time Risk'] == 1]
        if not lead_time_risk_items.empty:
            st.warning(f"⚠️ {len(lead_time_risk_items)} items below reorder point!")
            show_help("These items have dropped below their calculated reorder points and need immediate ordering to avoid stockouts during supplier lead time.")
            for _, row in lead_time_risk_items.head(3).iterrows():
                days_to_order = max(0, (row['Reorder Point'] - row['Inventory Level']) / row['Units Sold'])
                st.write(f"• **{row['Product ID']}** - Order needed in {int(days_to_order)} days")
        else:
            st.success("✅ All items above reorder points")
            show_help("All products currently have sufficient stock to cover lead times plus safety buffer.")
    
    with col2:
        st.subheader("🌤️ Weather Impact")
        
        # Get current season weather-sensitive items
        current_month = pd.to_datetime('today').month
        current_season = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}[current_month]
        
        # Category-specific weather sensitivity lookup
        category_weather_sensitivity = {
            'Electronics': {'Winter': 1.2, 'Spring': 1.0, 'Summer': 1.1, 'Fall': 1.0},
            'Clothing': {'Winter': 1.4, 'Spring': 1.1, 'Summer': 0.8, 'Fall': 1.2},
            'Toys': {'Winter': 1.3, 'Spring': 1.0, 'Summer': 1.5, 'Fall': 1.1},
            'Furniture': {'Winter': 0.9, 'Spring': 1.3, 'Summer': 1.2, 'Fall': 1.0},
            'Groceries': {'Winter': 1.1, 'Spring': 1.0, 'Summer': 0.95, 'Fall': 1.05}
        }
        
        # Calculate current season impact for each category
        current_weather_impacts = []
        for category in df['Category'].unique():
            if category in category_weather_sensitivity:
                multiplier = category_weather_sensitivity[category].get(current_season, 1.0)
            else:
                multiplier = {'Winter': 1.1, 'Spring': 1.0, 'Summer': 1.2, 'Fall': 1.0}.get(current_season, 1.0)
            
            impact_score = (multiplier - 1) * 100
            if abs(impact_score) > 2:  # Show impacts greater than 2% (lowered threshold)
                current_weather_impacts.append({
                    'category': category, 
                    'impact': impact_score,
                    'products': len(df[df['Category'] == category])
                })
        
        # Sort by impact magnitude
        current_weather_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        if current_weather_impacts:
            avg_impact = sum([abs(x['impact']) for x in current_weather_impacts]) / len(current_weather_impacts)
            if avg_impact > 10:
                st.warning(f"🌡️ Current {current_season} season significantly affecting {len(current_weather_impacts)} categories")
            else:
                st.info(f"🌤️ Mild {current_season} weather impact on {len(current_weather_impacts)} categories")
            
            show_help(f"Weather impact varies by product category. Current {current_season} season affects different product types differently - some categories see increased demand while others decrease.")
            
            for impact_data in current_weather_impacts[:3]:
                impact_direction = "increase" if impact_data['impact'] > 0 else "decrease"
                st.write(f"• **{impact_data['category']}** ({impact_data['products']} products) - {abs(impact_data['impact']):.1f}% {impact_direction}")
        else:
            st.success("🌤️ Minimal weather impact expected")
            show_help(f"Current {current_season} seasonal conditions are not significantly affecting product demand patterns.")

    # Simple Stock Level Chart
    st.header("📊 Stock Levels by Category")
    show_help("Visual breakdown of your inventory distribution across product categories. Larger bars indicate categories with more total units in stock. Use this to identify which categories dominate your inventory investment.")
    
    category_data = df.groupby('Category').agg({
        'Inventory Level': 'sum',
        'Total Value': 'sum',
        'Product ID': 'nunique'
    }).reset_index()
    
    fig = px.bar(
        category_data,
        x='Category',
        y='Inventory Level',
        color='Total Value',
        title="Total Inventory Units by Category (colored by total value)",
        labels={
            'Inventory Level': 'Total Units in Stock',
            'Category': 'Product Category',
            'Total Value': 'Total Value ($)'
        },
        text='Inventory Level'
    )
    fig.update_traces(texttemplate='%{text:,.0f} units', textposition='outside')
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary insights
    highest_value_category = category_data.loc[category_data['Total Value'].idxmax(), 'Category']
    highest_units_category = category_data.loc[category_data['Inventory Level'].idxmax(), 'Category']
    
    st.info(f"""
    **📈 Inventory Insights:**
    - **Highest Value Category:** {highest_value_category} (${category_data.loc[category_data['Category'] == highest_value_category, 'Total Value'].iloc[0]:,.0f})
    - **Most Units Category:** {highest_units_category} ({category_data.loc[category_data['Category'] == highest_units_category, 'Inventory Level'].iloc[0]:,.0f} units)
    - **Product Lines per Category:** {category_data['Product ID'].iloc[0]} unique products each
    - **Total Inventory Value:** ${category_data['Total Value'].sum():,.0f} across all categories
    """)
    
    # Show category distribution
    st.subheader("📊 Category Distribution Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Inventory Units by Category:**")
        for _, row in category_data.sort_values('Inventory Level', ascending=False).iterrows():
            percentage = (row['Inventory Level'] / category_data['Inventory Level'].sum()) * 100
            st.write(f"• {row['Category']}: {row['Inventory Level']:,.0f} units ({percentage:.1f}%)")
    
    with col2:
        st.write("**Value Distribution by Category:**")
        for _, row in category_data.sort_values('Total Value', ascending=False).iterrows():
            percentage = (row['Total Value'] / category_data['Total Value'].sum()) * 100
            st.write(f"• {row['Category']}: ${row['Total Value']:,.0f} ({percentage:.1f}%)")

def product_analysis():
    st.title("📈 Product Analysis")
    st.write("""
    This page helps you understand how your products are performing. You can filter by category 
    and see detailed information about stock levels and sales trends.
    """)
    
    df = load_data()
    if df is None:
        return
    
    # Get most recent data for current analysis
    latest_date = df['Date'].max()
    current_data = df[df['Date'] == latest_date].copy()
    
    # Filters
    st.header("1️⃣ Select What You Want to See")
    show_help("Use these dropdowns to focus on specific categories or stores. Analysis shows current data with historical trends.")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox(
            "Choose a Product Category",
            options=['All Categories'] + list(df['Category'].unique())
        )
    
    with col2:
        selected_store = st.selectbox(
            "Choose a Store Location",
            options=['All Stores'] + list(df['Store ID'].unique())
        )
    
    # Filter data (use full dataset for trends, current data for metrics)
    filtered_df = df.copy()
    filtered_current = current_data.copy()
    
    if selected_category != 'All Categories':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        filtered_current = filtered_current[filtered_current['Category'] == selected_category]
    if selected_store != 'All Stores':
        filtered_df = filtered_df[filtered_df['Store ID'] == selected_store]
        filtered_current = filtered_current[filtered_current['Store ID'] == selected_store]
    
    # Current Overview
    st.header("2️⃣ Current Status Overview")
    show_help(f"Current inventory status as of {latest_date.strftime('%B %d, %Y')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(filtered_current['Product ID'].unique())
        st.markdown("### 📦 Total Products")
        st.markdown(f'<p class="big-number">{total_products}</p>', unsafe_allow_html=True)
        show_help("Number of unique products in selected category/store")

    with col2:
        total_current_stock = filtered_current['Inventory Level'].sum()
        st.markdown("### 📊 Current Stock")
        st.markdown(f'<p class="big-number">{total_current_stock:,}</p>', unsafe_allow_html=True)
        show_help("Total units currently in stock")

    with col3:
        avg_daily_sales = filtered_current['Units Sold'].mean()
        st.markdown("### 📈 Avg Daily Sales")
        st.markdown(f'<p class="big-number">{avg_daily_sales:.1f}</p>', unsafe_allow_html=True)
        show_help("Average daily sales per product")

    with col4:
        total_value = (filtered_current['Inventory Level'] * filtered_current['Price']).sum()
        st.markdown("### 💰 Current Value")
        st.markdown(f'<p class="big-number">${total_value:,.0f}</p>', unsafe_allow_html=True)
        show_help("Total value of current inventory")
    
    # Simple Stock Overview
    st.header("3️⃣ Current Stock by Category")
    st.write("Simple view of how much stock you have in each category right now.")
    
    # Get current stock by category (simple aggregation)
    if len(filtered_current) > 0:
        category_stock = filtered_current.groupby('Category')['Inventory Level'].sum().reset_index()
        category_stock = category_stock.sort_values('Inventory Level', ascending=True)
        
        fig = px.bar(
            category_stock,
            x='Inventory Level',
            y='Category',
            orientation='h',
            title="Current Stock by Category",
            text='Inventory Level'
        )
        fig.update_traces(texttemplate='%{text:,} units', textposition='outside')
        fig.update_layout(
            yaxis_title="Category",
            xaxis_title="Total Units in Stock"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for selected filters")
    
    # Simple Product Comparison
    st.header("4️⃣ Sales vs Stock Comparison")
    st.write("Simple comparison: How much you're selling vs how much stock you have.")
    
    if len(filtered_current) > 0:
        # Get top 10 products by sales for a meaningful comparison
        top_selling = filtered_current.nlargest(10, 'Units Sold')[['Product ID', 'Category', 'Units Sold', 'Inventory Level']]
        
        if len(top_selling) > 0:
            fig = go.Figure()
            
            # Add sales bars
            fig.add_trace(go.Bar(
                x=top_selling['Product ID'],
                y=top_selling['Units Sold'],
                name='Daily Sales',
                marker_color='lightblue'
            ))
            
            # Add stock bars
            fig.add_trace(go.Bar(
                x=top_selling['Product ID'],
                y=top_selling['Inventory Level'],
                name='Current Stock',
                marker_color='orange'
            ))
            
            fig.update_layout(
                title="Sales vs Stock for Top Selling Products",
                xaxis_title="Product",
                yaxis_title="Units",
                barmode='group',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **How to read this:** Blue bars = daily sales, Orange bars = current stock. If orange is much bigger than blue, you might have too much stock. If orange is smaller than blue, you might run out soon.")
        else:
            st.info("No sales data available for comparison.")
    else:
        st.info("No data available for selected filters")
    
    # Top Performers Analysis
    st.header("5️⃣ Simple Performance Summary")
    show_help("Easy-to-understand rankings of your products")
    
    if len(filtered_current) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Best Sellers")
            st.write("Products with highest daily sales:")
            top_sales = filtered_current.nlargest(5, 'Units Sold')[['Product ID', 'Category', 'Units Sold', 'Inventory Level', 'Store ID']]
            for i, (_, row) in enumerate(top_sales.iterrows(), 1):
                days_left = row['Inventory Level'] / row['Units Sold'] if row['Units Sold'] > 0 else float('inf')
                st.write(f"**{i}. {row['Product ID']}** ({row['Category']})")
                st.write(f"   Sales: {row['Units Sold']} units/day | Stock: {row['Inventory Level']} units")
        
        with col2:
            st.subheader("📦 Highest Stock")
            st.write("Products with most inventory:")
            high_stock = filtered_current.nlargest(5, 'Inventory Level')[['Product ID', 'Category', 'Units Sold', 'Inventory Level', 'Store ID']]
            
            for i, (_, row) in enumerate(high_stock.iterrows(), 1):
                st.write(f"**{i}. {row['Product ID']}** ({row['Category']})")
                st.write(f"   Stock: {row['Inventory Level']} units | Sales: {row['Units Sold']} units/day")
    else:
        st.info("No products found with the selected filters.")
    
    # Detailed Product Information
    st.header("6️⃣ Individual Product Details")
    show_help("Pick any product to see its specific information")
    
    if len(filtered_current) > 0:
        selected_product = st.selectbox(
            "Choose a Product:",
            options=filtered_current['Product ID'].unique(),
            format_func=lambda x: f"{x} ({filtered_current[filtered_current['Product ID']==x]['Category'].iloc[0]})"
        )
        
        # Get detailed data for selected product
        product_detail = filtered_current[filtered_current['Product ID'] == selected_product].iloc[0]
        
        # Simple metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Stock", f"{int(product_detail['Inventory Level']):,} units")
            st.metric("Daily Sales", f"{int(product_detail['Units Sold']):,} units")
            
        with col2:
            days_of_stock = product_detail['Inventory Level'] / product_detail['Units Sold'] if product_detail['Units Sold'] > 0 else float('inf')
            st.metric("Days Left", f"{int(days_of_stock) if days_of_stock != float('inf') else '∞'} days")
            st.metric("Price", f"${product_detail['Price']:.2f}")
            
        with col3:
            total_value = product_detail['Inventory Level'] * product_detail['Price']
            st.metric("Stock Value", f"${total_value:,.2f}")
            st.metric("Store", product_detail['Store ID'])
        
        # Simple product status overview
        product_history = filtered_df[filtered_df['Product ID'] == selected_product].copy()
        
        if len(product_history) > 1:
            st.subheader(f"📊 {selected_product} - Quick Look")
            st.write("Simple comparison of what you have vs what you sell:")
            
            # Create very simple comparison: What you have vs What you sell
            stock_value = int(product_detail['Inventory Level'])
            sales_value = int(product_detail['Units Sold'])
            
            # Simple horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=['What You Have in Stock', 'What You Sell Daily'],
                x=[stock_value, sales_value],
                orientation='h',
                marker_color=['green', 'blue'],
                text=[f'{stock_value} units', f'{sales_value} units'],
                textposition='inside'
            ))
            
            fig.update_layout(
                title=f"{selected_product} - Simple Comparison",
                xaxis_title="Number of Units",
                yaxis_title="",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Simple interpretation
            if stock_value > sales_value * 14:
                st.success("✅ You have plenty of stock (more than 2 weeks worth)")
            elif stock_value > sales_value * 7:
                st.warning("⚠️ You have good stock (1-2 weeks worth)")
            else:
                st.error("🚨 You're running low on stock (less than 1 week)")
                
            st.info(f"💡 **Simple math:** You have {stock_value} units and sell {sales_value} per day. That's about {int(stock_value/sales_value) if sales_value > 0 else '∞'} days of stock.")
    else:
        st.info("No products found with the selected filters.")

def supplier_page():
    st.title("🤝 Supplier Information")
    st.write("""
    Here you can see how well your suppliers are performing. We look at things like delivery speed,
    order volumes, and reliability to help you choose the best suppliers for your business.
    """)
    
    df = load_data()
    if df is None:
        return
    
    # Calculate detailed supplier metrics
    try:
        supplier_metrics = df.groupby('Vendor').agg({
            'Vendor Avg Delivery Days': 'mean',
            'Units Ordered': 'sum',
            'StockoutRisk': 'mean',
            'Product ID': 'nunique'
        }).reset_index()
        
        # Ensure we have data
        if supplier_metrics.empty:
            st.error("No supplier data found in the dataset.")
            return
            
        # Calculate normalized scores (0-100 scale) with error handling
        # Delivery Score: Faster delivery = higher score
        supplier_metrics['Delivery Score'] = (
            (1 / supplier_metrics['Vendor Avg Delivery Days'].clip(lower=0.1)) /
            (1 / supplier_metrics['Vendor Avg Delivery Days'].clip(lower=0.1)).max() * 100
        ).fillna(0)
        
        # Volume Score: Higher order volume = higher score
        supplier_metrics['Volume Score'] = (
            supplier_metrics['Units Ordered'] / supplier_metrics['Units Ordered'].max() * 100
        ).fillna(0)
        
        # Reliability Score: Lower stockout risk = higher score
        supplier_metrics['Reliability Score'] = (
            (1 - supplier_metrics['StockoutRisk'].clip(0, 1)) * 100
        ).fillna(0)
        
        # Overall Score with weighted average (keeping more precision)
        supplier_metrics['Overall Score'] = (
            supplier_metrics['Delivery Score'] * 0.4 +
            supplier_metrics['Volume Score'] * 0.3 +
            supplier_metrics['Reliability Score'] * 0.3
        ).round(1)  # Round to 1 decimal place instead of whole numbers
        
    except Exception as e:
        st.error(f"Error calculating supplier metrics: {str(e)}")
        return
    
    # Supplier Rankings
    st.header("1️⃣ Overall Supplier Performance")
    show_help("Compare your suppliers based on delivery speed (40%), order volume (30%), and reliability (30%). Higher scores indicate better performing suppliers.")
    
    st.write("""
    **How to read this chart:**
    - 🟢 **Excellent (80-100)**: Top-tier suppliers
    - 🟡 **Good (60-79)**: Reliable suppliers 
    - 🔴 **Needs Improvement (Below 60)**: Consider finding alternatives
    """)
    
    if len(supplier_metrics) > 0:
        fig = px.bar(
            supplier_metrics.sort_values('Overall Score', ascending=True),
            y='Vendor',
            x='Overall Score',
            orientation='h',
            title="Supplier Performance Scores",
            text='Overall Score',
            color='Overall Score',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig.update_traces(texttemplate='%{text:.1f}')  # Show 1 decimal place
        fig.update_layout(
            yaxis_title="Supplier Name",
            xaxis_title="Performance Score (0-100)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add performance summary
        avg_score = supplier_metrics['Overall Score'].mean()
        best_supplier = supplier_metrics.loc[supplier_metrics['Overall Score'].idxmax(), 'Vendor']
        worst_supplier = supplier_metrics.loc[supplier_metrics['Overall Score'].idxmin(), 'Vendor']
        
        st.info(f"""
        **📊 Performance Summary:**
        - **Average Score:** {avg_score:.1f}/100
        - **Best Supplier:** {best_supplier} ({supplier_metrics.loc[supplier_metrics['Overall Score'].idxmax(), 'Overall Score']:.1f}/100)
        - **Needs Attention:** {worst_supplier} ({supplier_metrics.loc[supplier_metrics['Overall Score'].idxmin(), 'Overall Score']:.1f}/100)
        """)
    else:
        st.warning("No supplier data available to display.")
    
    # Detailed Supplier Cards
    st.header("2️⃣ Detailed Supplier Information")
    show_help("Click on any supplier below to see their detailed performance metrics and recommendations.")
    
    if len(supplier_metrics) > 0:
        for _, supplier in supplier_metrics.sort_values('Overall Score', ascending=False).iterrows():
            # Determine supplier status
            overall_score = supplier['Overall Score']  # Keep as float for more precision
            if overall_score >= 80:
                status_icon = "🟢"
                status_text = "Excellent"
            elif overall_score >= 60:
                status_icon = "🟡"
                status_text = "Good"
            else:
                status_icon = "🔴"
                status_text = "Needs Improvement"
                
            with st.expander(f"{status_icon} {supplier['Vendor']} - {status_text} ({overall_score:.1f}/100)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🚚 Delivery Speed")
                    score = int(supplier['Delivery Score'])
                    color = 'green' if score >= 70 else 'orange' if score >= 50 else 'red'
                    st.markdown(f"**Score:** <span style='color: {color}'>{score}/100</span>", unsafe_allow_html=True)
                    st.write(f"Average delivery time: {supplier['Vendor Avg Delivery Days']:.1f} days")
                
                with col2:
                    st.markdown("### 📦 Order Volume")
                    score = int(supplier['Volume Score'])
                    color = 'green' if score >= 70 else 'orange' if score >= 50 else 'red'
                    st.markdown(f"**Score:** <span style='color: {color}'>{score}/100</span>", unsafe_allow_html=True)
                    st.write(f"Total units ordered: {supplier['Units Ordered']:,.0f}")
                
                with col3:
                    st.markdown("### ⭐ Reliability")
                    score = int(supplier['Reliability Score'])
                    color = 'green' if score >= 70 else 'orange' if score >= 50 else 'red'
                    st.markdown(f"**Score:** <span style='color: {color}'>{score}/100</span>", unsafe_allow_html=True)
                    st.write(f"Products supplied: {supplier['Product ID']} items")
                
                # Performance recommendations
                st.markdown("### 📈 Performance Analysis")
                
                # Generate specific recommendations
                recommendations = []
                if supplier['Delivery Score'] < 70:
                    recommendations.append("🚚 **Delivery**: Consider negotiating faster delivery options")
                if supplier['Volume Score'] < 70:
                    recommendations.append("📦 **Volume**: Explore bulk ordering opportunities")
                if supplier['Reliability Score'] < 70:
                    recommendations.append("⭐ **Reliability**: Monitor stock levels more closely")
                    
                if overall_score >= 80:
                    st.success("✅ **Excellent Performance**: Top-tier supplier meeting all expectations")
                elif overall_score >= 60:
                    st.info("ℹ️ **Good Performance**: Reliable supplier with minor improvement areas")
                    if recommendations:
                        st.write("**Suggestions for improvement:**")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                else:
                    st.warning("⚠️ **Needs Attention**: Performance review recommended")
                    if recommendations:
                        st.write("**Priority improvements needed:**")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                    
    else:
        st.warning("⚠️ No supplier data available. Please check your data file.")
    
    # Tips for interpretation
    st.header("3️⃣ How to Use This Information")
    st.info("""
    🎯 What makes a good supplier?
    - Delivery Score: Lower delivery times are better
    - Volume Score: Higher order volumes show reliability
    - Reliability Score: Based on preventing stock-outs
    
    💡 Tips:
    - Consider multiple suppliers for critical items
    - Look for suppliers with scores above 70 for new orders
    - Monitor trends in delivery times
    """)

def predictions_page():
    st.title("🔮 Future Predictions")
    st.write("""
    Let us help you plan for the future! We'll analyze your data to predict:
    - Which products might run out of stock
    - How much you should order
    - When you should place new orders
    """)
    
    df = load_data()
    if df is None:
        return
    
    # Train models if not already trained
    if 'reorder_model' not in st.session_state.models:
        with st.spinner("📊 Analyzing your data to make predictions..."):
            # Prepare features
            features = ['Inventory Level', 'Units Sold', 'Price', 'Vendor Avg Delivery Days']
            X = df[features]
            y_reorder = df['StockoutRisk']
            
            # Train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y_reorder)
            
            st.session_state.models['reorder_model'] = model
            st.session_state.models['scaler'] = scaler
    
    # Product Selection
    st.header("1️⃣ Choose a Product")
    show_help("Select a product to see predictions about its future stock needs")
    
    selected_product = st.selectbox(
        "Which product would you like to analyze?",
        options=df['Product ID'].unique(),
        format_func=lambda x: f"{x} ({df[df['Product ID']==x]['Category'].iloc[0]})"
    )
    
    # Get current product data
    product_data = df[df['Product ID'] == selected_product].iloc[-1]
    
    # Make predictions
    features = ['Inventory Level', 'Units Sold', 'Price', 'Vendor Avg Delivery Days']
    X_pred = product_data[features].values.reshape(1, -1)
    X_pred_scaled = st.session_state.models['scaler'].transform(X_pred)
    reorder_prob = st.session_state.models['reorder_model'].predict(X_pred_scaled)[0]
    
    # Display predictions in a more user-friendly way
    st.header("2️⃣ What We Predict")
    
    # Risk Level
    risk_level = "High" if reorder_prob > 0.7 else "Medium" if reorder_prob > 0.3 else "Low"
    risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
    
    st.markdown(f"""
    ### Stock Risk Level: <span style='color: {risk_color}'>{risk_level}</span>
    """, unsafe_allow_html=True)
    
    # Recommendations based on risk
    if risk_level == "High":
        st.error("⚠️ Action Needed Soon!")
    elif risk_level == "Medium":
        st.warning("👀 Keep an Eye on This")
    else:
        st.success("✅ Looking Good!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_stock = int(product_data['Inventory Level'])
        st.metric("Current Stock", f"{current_stock:,} units")
    
    with col2:
        daily_sales = int(product_data['Units Sold'])
        st.metric("Daily Sales", f"{daily_sales:,} units")
    
    with col3:
        days_until_stockout = int(product_data['DaysToStockout'])
        st.metric(
            "Days Until Reorder Needed",
            f"{days_until_stockout} days",
            delta=f"{days_until_stockout - 14}" if days_until_stockout < 14 else None,
            delta_color="inverse"
        )
    
    # Order Recommendations
    st.header("3️⃣ What You Should Do")
    
    recommended_qty = int(max(0, product_data['Demand Forecast'] - product_data['Inventory Level']))
    if recommended_qty > 0:
        st.write(f"""
        Based on our analysis, we recommend:
        1. Order **{recommended_qty:,} units** of this product
        2. Place the order within **{min(7, days_until_stockout)} days**
        3. Expected delivery time: **{int(product_data['Vendor Avg Delivery Days'])} days**
        """)
    else:
        st.write("No immediate action needed. Stock levels are healthy! 🌟")
    
    # Simple Stock Overview
    st.header("4️⃣ Simple Stock Overview")
    show_help("Easy-to-read comparison of your current stock vs what you need")
    
    product_history = df[df['Product ID'] == selected_product].copy()
    
    if len(product_history) > 0:
        # Get the latest data for this product
        latest_data = product_history.iloc[-1]
        
        # Calculate simple metrics
        current_stock = int(latest_data['Inventory Level'])
        daily_sales = int(latest_data['Units Sold']) if latest_data['Units Sold'] > 0 else 1
        
        # Calculate different stock scenarios
        one_week_stock = daily_sales * 7
        two_week_stock = daily_sales * 14
        one_month_stock = daily_sales * 30
        
        # Create simple bar chart showing what you have vs what you need
        categories = ['What You Have Now', '1 Week Supply', '2 Week Supply', '1 Month Supply']
        values = [current_stock, one_week_stock, two_week_stock, one_month_stock]
        colors = ['blue', 'yellow', 'orange', 'green']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v:,} units' for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"{selected_product} - Stock Comparison",
            xaxis_title="Stock Type",
            yaxis_title="Number of Units",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Simple interpretation
        if current_stock >= one_month_stock:
            st.success("✅ **Excellent!** You have more than a month of stock")
        elif current_stock >= two_week_stock:
            st.info("👍 **Good!** You have 2-4 weeks of stock")
        elif current_stock >= one_week_stock:
            st.warning("⚠️ **Watch Out!** You have 1-2 weeks of stock")
        else:
            st.error("🚨 **Order Now!** You have less than 1 week of stock")
        
        # Add simple explanation
        st.info(f"""
        💡 **Simple Explanation:**
        - **Blue bar (What you have):** Your current inventory
        - **Yellow bar (1 week):** Stock needed for 1 week
        - **Orange bar (2 weeks):** Stock needed for 2 weeks  
        - **Green bar (1 month):** Stock needed for 1 month
        
        **Goal:** Keep your blue bar higher than the orange bar for safe operations!
        """)
    else:
        st.info("No historical data available for this product.")
    
    # Tips box
    st.info("""
    💡 **How to Read This Page**
    
    - **Risk Level**: Shows how likely you are to run out of stock
    - **Days Until Reorder**: How long your current stock will last
    - **Recommended Order**: How much you should order to maintain healthy stock
    - **Chart**: Blue line should stay above the red dotted line for safety
    """)

def lead_time_procurement_page():
    st.title("📦 Simple Order Manager")
    st.write("""
    **Easy ordering made simple!** Just tell me what to order and when to order it.
    No complicated charts - just clear answers to help you manage your inventory.
    """)
    
    df = load_data()
    if df is None:
        st.error("❌ No data available. Please check your data file.")
        return
    
    # Super simple math
    try:
        # Only look at items with sales
        items = df[df['Units Sold'] > 0].copy()
        if len(items) == 0:
            st.error("❌ No sales data found.")
            return
        
        # Simple calculation: How many days will stock last?
        items['Days_Left'] = (items['Inventory Level'] / items['Units Sold']).round(0)
        
        # Simple status
        items['Simple_Status'] = items['Days_Left'].apply(lambda x: 
            'Order Today!' if x <= 3 else 
            'Order This Week' if x <= 7 else 
            'Order Soon' if x <= 14 else 
            'Good for Now'
        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return

    # 1. SIMPLE OVERVIEW
    st.header("📊 Quick Status Check")
    show_help("Simple count of how many items need attention at different urgency levels")
    
    # Count items by urgency
    today_items = len(items[items['Simple_Status'] == 'Order Today!'])
    week_items = len(items[items['Simple_Status'] == 'Order This Week'])
    soon_items = len(items[items['Simple_Status'] == 'Order Soon'])
    good_items = len(items[items['Simple_Status'] == 'Good for Now'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🔴 Order Today!")
        st.markdown(f'<p class="big-number">{today_items} items</p>', unsafe_allow_html=True)
        show_help("Critical - 3 days or less of stock remaining")
        if today_items > 0:
            st.error("🚨 URGENT!")
    
    with col2:
        st.markdown("### 🟠 Order This Week")
        st.markdown(f'<p class="big-number">{week_items} items</p>', unsafe_allow_html=True)
        show_help("Important - 4-7 days of stock remaining")
        if week_items > 0:
            st.warning("⚠️ Soon")
    
    with col3:
        st.markdown("### 🟡 Order Soon")
        st.markdown(f'<p class="big-number">{soon_items} items</p>', unsafe_allow_html=True)
        show_help("Watch - 8-14 days of stock remaining")
    
    with col4:
        st.markdown("### 🟢 Good for Now")
        st.markdown(f'<p class="big-number">{good_items} items</p>', unsafe_allow_html=True)
        show_help("Healthy - More than 14 days of stock")

    # 2. CRITICAL ITEMS LIST
    if today_items > 0:
        st.header("🚨 Order These Items TODAY!")
        st.write("**These items are running out fast - order them immediately!**")
        
        critical_items = items[items['Simple_Status'] == 'Order Today!'].sort_values('Days_Left')
        
        for i, (_, item) in enumerate(critical_items.head(5).iterrows(), 1):
            st.markdown(f"### {i}. {item['Product ID']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📦 Current Stock:**")
                st.write(f"{item['Inventory Level']:,.0f} units")
                st.write(f"**⏰ Days Left:** {item['Days_Left']:.0f}")
            
            with col2:
                st.write("**📈 Daily Sales:**")
                st.write(f"{item['Units Sold']:.0f} units/day")
                st.write(f"**💰 Price:** ${item['Price']:.2f}")
            
            with col3:
                # Simple order suggestion
                order_qty = item['Units Sold'] * 30  # 30 days worth
                total_cost = order_qty * item['Price']
                st.write("**💡 Suggested Order:**")
                st.write(f"{order_qty:.0f} units")
                st.write(f"**Cost:** ${total_cost:,.0f}")
            
            st.markdown("---")

    # 3. WEEKLY PLANNING
    if week_items > 0:
        st.header("📅 Plan These Orders This Week")
        st.write("**Good planning - these items need ordering within 7 days**")
        
        weekly_items = items[items['Simple_Status'] == 'Order This Week'].sort_values('Days_Left')
        
        # Simple table format
        st.write("**Quick Reference List:**")
        for _, item in weekly_items.head(10).iterrows():
            order_qty = item['Units Sold'] * 30
            cost = order_qty * item['Price']
            st.write(f"• **{item['Product ID']}** - {item['Days_Left']:.0f} days left - Order {order_qty:.0f} units (${cost:,.0f})")

    # 4. SIMPLE SUPPLIER INFO
    st.header("🚚 Supplier Quick Reference")
    st.write("**Which suppliers are fastest for urgent orders?**")
    
    if 'Vendor' in items.columns:
        # Simple supplier comparison
        supplier_speed = items.groupby('Vendor')['Vendor Avg Delivery Days'].mean().sort_values()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Fastest Suppliers")
            st.write("*Call these first for urgent orders:*")
            for supplier, days in supplier_speed.head(3).items():
                st.write(f"• **{supplier}**: {days:.0f} days")
        
        with col2:
            st.subheader("🐌 Slower Suppliers")
            st.write("*Plan ahead for these:*")
            for supplier, days in supplier_speed.tail(3).items():
                st.write(f"• **{supplier}**: {days:.0f} days")
    else:
        st.info("💡 Add supplier information to see delivery comparisons")

    # 5. BUDGET PLANNING
    st.header("💰 Simple Budget Calculator")
    st.write("**How much money do you need for orders?**")
    
    # Calculate simple budgets
    today_budget = 0
    week_budget = 0
    
    for _, item in items.iterrows():
        order_cost = (item['Units Sold'] * 30) * item['Price']
        
        if item['Simple_Status'] == 'Order Today!':
            today_budget += order_cost
        elif item['Simple_Status'] == 'Order This Week':
            week_budget += order_cost
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Urgent Budget", f"${today_budget:,.0f}")
        st.caption("Money needed for urgent orders")
    
    with col2:
        st.metric("🟠 Weekly Budget", f"${week_budget:,.0f}")
        st.caption("Money needed this week")
    
    with col3:
        total_budget = today_budget + week_budget
        st.metric("📊 Total Needed", f"${total_budget:,.0f}")
        st.caption("Total money to set aside")

    # 6. INDIVIDUAL PRODUCT LOOKUP
    st.header("🔍 Check Any Product")
    st.write("**Pick any product to see when to order it**")
    
    product_list = items['Product ID'].tolist()
    selected_product = st.selectbox("Choose a product to check:", product_list)
    
    if selected_product:
        product = items[items['Product ID'] == selected_product].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Current Status")
            st.write(f"**Product:** {product['Product ID']}")
            st.write(f"**Stock:** {product['Inventory Level']:,.0f} units")
            st.write(f"**Daily Sales:** {product['Units Sold']:.0f} units")
            st.write(f"**Days Left:** {product['Days_Left']:.0f} days")
            st.write(f"**Status:** {product['Simple_Status']}")
        
        with col2:
            st.subheader("💡 What to Do")
            
            days_coverage = st.selectbox("How many days of stock do you want?", [14, 21, 30, 45, 60], index=2)
            
            target_stock = product['Units Sold'] * days_coverage
            current_stock = product['Inventory Level']
            order_needed = max(0, target_stock - current_stock)
            order_cost = order_needed * product['Price']
            
            if order_needed > 0:
                st.write(f"**Order:** {order_needed:.0f} units")
                st.write(f"**Cost:** ${order_cost:,.0f}")
                st.write(f"**This gives you:** {days_coverage} days of stock")
                
                if product['Days_Left'] <= 3:
                    st.error("🚨 Order immediately!")
                elif product['Days_Left'] <= 7:
                    st.warning("⚠️ Order this week")
                else:
                    st.success("✅ Plan ahead")
            else:
                st.success("✅ No order needed - you have enough stock!")

    # 7. SIMPLE ACTION PLAN
    st.header("📝 Your Action Plan")
    st.write("**Simple checklist for managing orders**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 Daily Tasks:**
        1. Check red items (Order Today!)
        2. Call suppliers for urgent items
        3. Place critical orders
        4. Update order tracking
        """)
    
    with col2:
        st.info("""
        **📅 Weekly Tasks:**
        • Review orange items (Order This Week)
        • Plan next week's orders
        • Check delivery schedules
        • Update budget planning
        """)
    
    # Simple tips
    st.header("💡 Simple Tips")
    st.success("""
    **🏆 Best Practices:**
    • Order when you have 7 days of stock left
    • Keep 30 days of stock for popular items
    • Call your fastest suppliers first for urgent orders
    • Check this page every Monday morning
    """)



def weather_impact_page():
    st.title("🌤️ Simple Weather Guide")
    st.write("""
    **Easy seasonal planning!** Understand how weather affects what people buy, 
    so you can stock up on the right things at the right time.
    """)
    
    df = load_data()
    if df is None:
        st.error("❌ No data available. Please check your data file.")
        return
    
    # Get current season - August 2025
    current_month = 8  # August
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}
    current_season = season_map[current_month]

    # 1. CURRENT SEASON OVERVIEW
    st.header("🌤️ What Season Is It?")
    st.write(f"**Today is August 7, 2025 - We're in {current_season} season!**")
    
    # Simple season explanation
    if current_season == "Summer":
        st.success("☀️ **Summer Time!** People buy more toys, outdoor items, and cooling products.")
        season_emoji = "☀️"
        season_items = ["Toys", "Electronics (AC/Fans)", "Outdoor Furniture", "Summer Clothing"]
        demand_change = "+25%"
    elif current_season == "Winter":
        st.info("❄️ **Winter Time!** People buy more warm clothes and holiday gifts.")
        season_emoji = "❄️"
        season_items = ["Clothing", "Electronics (Heaters)", "Holiday Items", "Comfort Products"]
        demand_change = "+15%"
    elif current_season == "Fall":
        st.warning("🍂 **Fall Time!** Back to school and preparing for winter.")
        season_emoji = "🍂"
        season_items = ["Clothing", "Electronics", "School Supplies", "Furniture"]
        demand_change = "+10%"
    else:  # Spring
        st.success("🌸 **Spring Time!** Spring cleaning and home improvement season.")
        season_emoji = "🌸"
        season_items = ["Furniture", "Home Improvement", "Gardening", "Cleaning Products"]
        demand_change = "+5%"

    # 2. WHAT'S POPULAR RIGHT NOW
    st.header(f"{season_emoji} What's Popular in {current_season}?")
    show_help(f"These product categories typically see increased demand during {current_season}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Hot Categories")
        for item in season_items:
            st.write(f"• **{item}** - {demand_change} more sales expected")
    
    with col2:
        st.subheader("📊 Quick Numbers")
        st.metric("Expected Demand Increase", demand_change)
        st.metric("Categories Affected", len(season_items))
        st.metric("Planning Window", "2-3 months ahead")

    # 3. YOUR INVENTORY IMPACT
    st.header("📦 How This Affects Your Inventory")
    st.write("**Simple analysis of your current stock for seasonal items**")
    
    try:
        # Define what categories are affected by current season
        seasonal_categories = {
            'Summer': ['Toys', 'Electronics', 'Furniture'],
            'Winter': ['Clothing', 'Electronics', 'Toys'],
            'Spring': ['Furniture', 'Groceries'],
            'Fall': ['Clothing', 'Electronics']
        }
        
        current_seasonal_cats = seasonal_categories.get(current_season, [])
        
        # Check your inventory for these categories
        seasonal_items = df[df['Category'].isin(current_seasonal_cats)].copy()
        
        if len(seasonal_items) > 0:
            # Simple calculations with proper handling of zero sales
            total_seasonal_value = (seasonal_items['Inventory Level'] * seasonal_items['Price']).sum()
            total_seasonal_units = seasonal_items['Inventory Level'].sum()
            
            # Calculate days left only for items with sales, avoid division by zero
            items_with_sales = seasonal_items[seasonal_items['Units Sold'] > 0]
            if len(items_with_sales) > 0:
                days_left_values = items_with_sales['Inventory Level'] / items_with_sales['Units Sold']
                avg_days_left = days_left_values.mean()
                days_left_display = f"{avg_days_left:.0f} days"
                
                # Determine status based on average days left
                if avg_days_left < 14:
                    status_message = "⚠️ **Action Needed:** Your seasonal items are running low! Consider ordering more for the current season."
                    status_type = "warning"
                elif avg_days_left < 30:
                    status_message = "👀 **Watch Closely:** Seasonal items have moderate stock. Plan your next orders."
                    status_type = "info"
                else:
                    status_message = "✅ **Looking Good:** You have plenty of seasonal stock."
                    status_type = "success"
            else:
                # No sales data available
                days_left_display = "No sales data"
                status_message = "📊 **No Sales Data:** Unable to calculate stock duration. Monitor these items closely."
                status_type = "info"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Seasonal Items Value", f"${total_seasonal_value:,.0f}")
                st.caption("Total value of seasonal products")
            
            with col2:
                st.metric("Seasonal Items Stock", f"{total_seasonal_units:,.0f} units")
                st.caption("Total units of seasonal products")
            
            with col3:
                st.metric("Stock Duration", days_left_display)
                st.caption("How long seasonal stock will last")
            
            # Quick recommendations
            if status_type == "warning":
                st.warning(status_message)
            elif status_type == "info":
                st.info(status_message)
            else:
                st.success(status_message)
        
        else:
            st.info("💡 No seasonal categories found in your inventory for current season analysis.")
    
    except Exception as e:
        st.warning("Unable to analyze seasonal impact on your specific inventory.")

    # 4. SEASONAL PLANNING CALENDAR
    st.header("📅 Simple Seasonal Planning")
    st.write("**Plan ahead! Here's when to stock up for each season:**")
    
    # Create simple planning guide
    planning_guide = {
        "Spring (Mar-May)": {
            "Stock up in": "January-February",
            "Hot items": "Furniture, Home improvement, Gardening supplies",
            "Tip": "People clean and redecorate homes"
        },
        "Summer (Jun-Aug)": {
            "Stock up in": "April-May", 
            "Hot items": "Toys, Outdoor furniture, Cooling electronics",
            "Tip": "Kids on vacation, hot weather needs"
        },
        "Fall (Sep-Nov)": {
            "Stock up in": "July-August",
            "Hot items": "Clothing, Electronics, Back-to-school items", 
            "Tip": "Back to school and preparing for winter"
        },
        "Winter (Dec-Feb)": {
            "Stock up in": "October-November",
            "Hot items": "Clothing, Electronics, Holiday gifts",
            "Tip": "Cold weather and holiday shopping"
        }
    }
    
    for season, info in planning_guide.items():
        is_current = current_season in season
        if is_current:
            st.success(f"**{season}** ← We are here!")
        else:
            st.write(f"**{season}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"📦 **Stock up:** {info['Stock up in']}")
        with col2:
            st.write(f"🔥 **Hot items:** {info['Hot items']}")
        with col3:
            st.write(f"💡 **Why:** {info['Tip']}")
        
        if not is_current:
            st.write("---")

    # 5. SIMPLE WEATHER TIPS
    st.header("🎯 Quick Weather Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📈 Smart Ordering:**
        • Order seasonal items 2-3 months early
        • Watch weather forecasts for surprises
        • Keep 20% extra stock for hot seasons
        • Don't overstock off-season items
        """)
    
    with col2:
        st.success("""
        **🏆 Best Practices:**
        • Summer = More toys & outdoor items
        • Winter = More clothes & electronics  
        • Spring = More furniture & home items
        • Fall = Back-to-school rush
        """)

    # 6. QUICK SEASONAL CHECKER
    st.header("🔍 Check Any Product Category")
    st.write("**See how seasons affect different product types**")
    
    if len(df) > 0:
        available_categories = df['Category'].unique().tolist()
        selected_category = st.selectbox("Choose a category to check:", available_categories)
        
        if selected_category:
            # Simple seasonal impact for selected category
            seasonal_impact = {
                'Electronics': {
                    'Summer': '+10% (Air conditioning, fans)',
                    'Winter': '+20% (Heaters, holiday gifts)', 
                    'Spring': 'Normal (Steady demand)',
                    'Fall': '+15% (Back to school, new models)'
                },
                'Clothing': {
                    'Summer': '-20% (Less layers needed)',
                    'Winter': '+40% (Warm clothes needed)',
                    'Spring': '+10% (New season wardrobe)', 
                    'Fall': '+20% (Back to school, layers)'
                },
                'Toys': {
                    'Summer': '+30% (Kids on vacation)',
                    'Winter': '+25% (Holiday gifts)',
                    'Spring': 'Normal (Steady demand)',
                    'Fall': '+10% (Back to school)'
                },
                'Furniture': {
                    'Summer': '+20% (Outdoor furniture)',
                    'Winter': '-10% (Less moving/decorating)',
                    'Spring': '+30% (Spring cleaning, redecorating)',
                    'Fall': 'Normal (Steady demand)'
                },
                'Groceries': {
                    'Summer': '-5% (People eat out more)',
                    'Winter': '+5% (More home cooking)',
                    'Spring': 'Normal (Steady demand)',
                    'Fall': 'Normal (Steady demand)'
                }
            }
            
            if selected_category in seasonal_impact:
                st.write(f"**{selected_category} - Seasonal Impact:**")
                
                for season, impact in seasonal_impact[selected_category].items():
                    if season == current_season:
                        st.success(f"• **{season}** (Current): {impact}")
                    else:
                        st.write(f"• {season}: {impact}")
            else:
                st.info(f"No specific seasonal data for {selected_category}, but most products see +10-15% increase during peak seasons.")

    # 7. SIMPLE ACTION PLAN
    st.header("📝 Your Weather Action Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **� This Month ({current_season}):**
        • Focus on seasonal hot items
        • Check stock levels for popular categories  
        • Plan orders for next season
        • Monitor weather-sensitive products
        """)
    
    with col2:
        next_season = {
            'Summer': 'Fall', 'Fall': 'Winter', 
            'Winter': 'Spring', 'Spring': 'Summer'
        }[current_season]
        
        st.success(f"""
        **📅 Plan for {next_season}:**
        • Start ordering {next_season.lower()} items in 2 months
        • Research trending products
        • Clear out off-season inventory
        • Prepare storage for seasonal items
        """)

# Page Router
if page == "Home & Overview":
    home_page()
elif page == "Product Analysis":
    product_analysis()
elif page == "Supplier Information":
    supplier_page()
elif page == "Lead Time & Procurement":
    lead_time_procurement_page()
elif page == "Weather Impact Analysis":
    weather_impact_page()
else:  # Future Predictions
    predictions_page()
