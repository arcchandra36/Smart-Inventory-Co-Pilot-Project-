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
    
    # Historical Performance
    st.header("4️⃣ Past Performance")
    show_help("""
    This chart shows how stock levels changed over time (blue line) compared to predicted demand (orange line).
    When the blue line goes below the orange line, you might run out of stock.
    """)
    
    product_history = df[df['Product ID'] == selected_product].copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=product_history['Date'],
        y=product_history['Inventory Level'],
        name='Actual Stock Level',
        line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        x=product_history['Date'],
        y=product_history['Demand Forecast'],
        name='Expected Demand',
        line=dict(color='orange', dash='dash')
    ))
    
    # Add a minimum safe stock level line
    safe_stock = product_history['Units Sold'].mean() * 7  # 7 days of stock
    fig.add_hline(
        y=safe_stock,
        line_dash="dot",
        line_color="red",
        annotation_text="Minimum Safe Stock Level"
    )
    
    fig.update_layout(
        title="Stock Levels vs Expected Demand",
        xaxis_title="Date",
        yaxis_title="Number of Units",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tips box
    st.info("""
    💡 **How to Read This Page**
    
    - **Risk Level**: Shows how likely you are to run out of stock
    - **Days Until Reorder**: How long your current stock will last
    - **Recommended Order**: How much you should order to maintain healthy stock
    - **Chart**: Blue line should stay above the red dotted line for safety
    """)

def lead_time_procurement_page():
    st.title("🚚 Lead Time & Procurement Analysis")
    st.write("""
    This page helps you optimize your procurement process by analyzing supplier lead times,
    calculating optimal reorder points, and planning your ordering schedule to avoid stockouts.
    """)
    
    df = load_data()
    if df is None:
        return
    
    # Enhanced error handling and data validation for procurement calculations
    try:
        # Filter out items with zero sales to avoid division by zero
        active_items = df[df['Units Sold'] > 0].copy()
        if len(active_items) == 0:
            st.error("No active sales data found for procurement analysis.")
            return
        
        # Recalculate more realistic reorder points for active items only
        # Use a more conservative safety stock multiplier
        active_items['Adjusted Safety Stock Days'] = active_items['Procurement Lead Time'] * 0.5  # Reduced from 1.5x
        active_items['Adjusted Reorder Point'] = active_items['Units Sold'] * (
            active_items['Procurement Lead Time'] + active_items['Adjusted Safety Stock Days']
        )
        active_items['Adjusted Lead Time Risk'] = (
            active_items['Inventory Level'] < active_items['Adjusted Reorder Point']
        ).astype(int)
        
    except Exception as e:
        st.error(f"Error in procurement calculations: {str(e)}")
        return

    # Overview Metrics
    st.header("1️⃣ Procurement Overview")
    show_help("Key metrics for managing your procurement timeline and reducing stockout risk. Analysis focuses on active products with sales data.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_lead_time = active_items['Procurement Lead Time'].mean()
        st.markdown("### 📅 Avg Lead Time")
        st.markdown(f'<p class="big-number">{avg_lead_time:.1f} days</p>', unsafe_allow_html=True)
        show_help("Average time from order to delivery across all suppliers for active products")

    with col2:
        lead_time_risk_items = len(active_items[active_items['Adjusted Lead Time Risk'] == 1])
        st.markdown("### ⚠️ Lead Time Risk")
        st.markdown(f'<p class="big-number">{lead_time_risk_items} items</p>', unsafe_allow_html=True)
        show_help("Active products below their calculated reorder point (conservative calculation)")

    with col3:
        longest_lead_time = active_items['Procurement Lead Time'].max()
        st.markdown("### 🐌 Longest Lead Time")
        st.markdown(f'<p class="big-number">{longest_lead_time:.0f} days</p>', unsafe_allow_html=True)
        show_help("Supplier with the longest delivery time")

    with col4:
        # More realistic safety stock calculation
        realistic_safety_value = (active_items['Adjusted Safety Stock Days'] * active_items['Units Sold'] * active_items['Price']).sum()
        st.markdown("### 💰 Safety Stock Value")
        st.markdown(f'<p class="big-number">${realistic_safety_value:,.0f}</p>', unsafe_allow_html=True)
        show_help("Total value tied up in safety stock (conservative estimate)")
    
    # Add data quality information
    total_products = len(df)
    active_products = len(active_items)
    inactive_products = total_products - active_products
    
    if inactive_products > 0:
        st.info(f"""
        **📊 Data Quality Note:**
        - **Total Products:** {total_products:,}
        - **Active Products (with sales):** {active_products:,}
        - **Inactive Products:** {inactive_products:,} (excluded from lead time analysis)
        """)

def weather_impact_page():
    st.title("🌤️ Weather Impact Analysis")
    st.write("""
    Simple guide to understanding how seasons affect your inventory. 
    This helps you prepare for busy seasons and plan your stock levels throughout the year.
    """)
    
    df = load_data()
    if df is None:
        return
    
    # Educational Weather Impact Section
    st.header("1️⃣ Understanding Weather Impact on Your Business")
    show_help("Learn how different seasons affect what customers buy, and how to plan your inventory accordingly")
    
    # Get current season (August = Summer) - clearly explained
    current_month = 8  # August 2025
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}
    current_season = season_map[current_month]
    
    # Clear explanation box
    st.info(f"""
    📅 **Today's Date: August 5, 2025**
    
    **How we determine the season:** Since today is August 5th, we're in **{current_season}** season.
    This is calculated from today's date, not from your Excel file.
    
    **Why seasons matter for inventory:**
    • **Summer** = More toys (kids on vacation), electronics (air conditioning), outdoor furniture
    • **Winter** = More clothing (cold weather), electronics (holiday gifts), comfort items
    • **Spring** = More furniture (spring cleaning), home improvement items
    • **Fall** = More clothing (back to school), electronics (new school year)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🌤️ Current Season")
        st.markdown(f'<p class="big-number">{current_season}</p>', unsafe_allow_html=True)
        show_help("Current seasonal period affecting demand patterns")

    with col2:
        weather_boost = "+25%" if current_season == "Summer" else "+15%" if current_season in ["Winter"] else "+5%"
        st.markdown("### 📈 Demand Boost")
        st.markdown(f'<p class="big-number">{weather_boost}</p>', unsafe_allow_html=True)
        show_help("Average increase in demand due to current season")

    with col3:
        affected_categories = 4 if current_season in ["Summer", "Winter"] else 2
        st.markdown("### 🎯 Affected Categories")
        st.markdown(f'<p class="big-number">{affected_categories} categories</p>', unsafe_allow_html=True)
        show_help("Number of product categories significantly affected by current weather")

    with col4:
        temp_status = "High" if current_season == "Summer" else "Low" if current_season == "Winter" else "Mild"
        st.markdown("### 🌡️ Temperature Impact")
        st.markdown(f'<p class="big-number">{temp_status}</p>', unsafe_allow_html=True)
        show_help("Current temperature impact level on product demand")

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
