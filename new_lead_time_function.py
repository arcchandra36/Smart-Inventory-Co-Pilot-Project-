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
