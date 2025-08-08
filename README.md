# 📦 Smart Inventory Co-Pilot

A comprehensive inventory management system built with Python and Streamlit that provides intelligent insights, predictions, and recommendations for retail inventory management.

## 🚀 Features

### Core Functionality
- **📊 Interactive Dashboard** - Real-time inventory overview with key metrics
- **📈 Product Analysis** - Detailed performance analysis with filtering and visualization
- **🤝 Supplier Management** - Performance scoring and recommendation system
- **📦 Lead Time & Procurement** - Smart ordering suggestions with urgency levels
- **🌤️ Weather Impact Analysis** - Seasonal demand forecasting and planning
- **🔮 Future Predictions** - ML-powered stock predictions and recommendations

### Technical Features
- **Machine Learning Integration** - Random Forest models for demand forecasting
- **Real-time Analytics** - Dynamic calculations and visualizations
- **User-friendly Interface** - Intuitive navigation with helpful tooltips
- **Comprehensive Backup System** - PowerShell-based backup and recovery tools
- **Data Protection** - Multiple backup strategies and recovery procedures

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python 3.x
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, StandardScaler)
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Data Storage**: CSV files
- **Backup System**: PowerShell scripts

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- PowerShell (for backup scripts)

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arcchandra36/Smart-Inventory-Co-Pilot-Project-.git
   cd Smart-Inventory-Co-Pilot-Project-
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

## 📂 Project Structure

```
SMART INVENTORY CO-PILOT/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── app_backup_v1.py       # Backup version 1
│   ├── app_backup_v2.py       # Backup version 2
│   └── app_backup_final.py    # Final backup version
├── data/
│   └── retail_store_inventory_with_vendors.csv  # Sample inventory data
├── models/
│   ├── quantity_model.pkl     # Trained ML model for quantity prediction
│   ├── reorder_model.pkl      # Trained ML model for reorder points
│   └── scaler.pkl            # StandardScaler for feature normalization
├── backups/                   # Timestamped backup files
├── create_backup.ps1          # PowerShell backup script
├── recover_app.ps1            # PowerShell recovery script
├── CODE_PROTECTION_GUIDE.md   # Comprehensive backup guide
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## 🎯 Usage

### Home & Overview
- View critical inventory metrics at a glance
- Identify items needing immediate attention
- Monitor lead time and weather alerts
- Analyze inventory distribution by category

### Product Analysis
- Filter products by category and store location
- Compare sales vs stock levels
- Identify top performers and slow movers
- Detailed individual product analysis

### Supplier Information
- Performance scoring based on delivery speed, volume, and reliability
- Detailed supplier metrics and recommendations
- Comparative analysis for supplier selection

### Lead Time & Procurement
- Simple order management with urgency levels
- Budget planning and cost calculations
- Supplier delivery time comparisons
- Individual product order recommendations

### Weather Impact Analysis
- Seasonal demand forecasting
- Category-specific weather sensitivity analysis
- Planning calendar for seasonal inventory
- Real-time seasonal impact on current inventory

### Future Predictions
- ML-powered stock risk assessment
- Order quantity recommendations
- Visual comparison of stock scenarios
- Predictive analytics for inventory planning

## 🔐 Data Protection

The project includes a comprehensive backup system:

- **Automated Backups**: PowerShell scripts for regular backups
- **Multiple Versions**: Version control for critical files
- **Recovery Tools**: Easy restoration procedures
- **Documentation**: Detailed protection guide

### Running Backups
```powershell
# Create a backup
powershell -ExecutionPolicy Bypass -File "create_backup.ps1"

# Recover from backup
powershell -ExecutionPolicy Bypass -File "recover_app.ps1"
```

## 📊 Sample Data

The project includes sample retail inventory data with:
- 200+ products across 5 categories (Electronics, Clothing, Toys, Furniture, Groceries)
- Multiple store locations
- Vendor information and delivery metrics
- Historical sales and inventory levels
- Demand forecasting data

## 🤖 Machine Learning Models

### Features Used
- Inventory Level
- Units Sold
- Price
- Vendor Average Delivery Days

### Models
- **Random Forest Regressor**: For stockout risk prediction
- **StandardScaler**: For feature normalization
- **Custom Algorithms**: For demand forecasting and reorder point calculation

## 🎨 User Interface

- **Responsive Design**: Works on different screen sizes
- **Intuitive Navigation**: Easy-to-use sidebar menu
- **Interactive Visualizations**: Plotly charts with hover information
- **Helpful Tooltips**: Guidance throughout the application
- **Status Indicators**: Color-coded alerts and recommendations

## 🔄 Updates and Maintenance

- Regular backup creation recommended
- Model retraining with new data
- Seasonal parameter adjustments
- Performance monitoring and optimization

## 📈 Future Enhancements

- [ ] Integration with real inventory management systems
- [ ] Advanced forecasting algorithms
- [ ] Mobile-responsive design improvements
- [ ] Real-time data synchronization
- [ ] Advanced reporting features
- [ ] Multi-user support with role-based access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is created for educational purposes. Feel free to use and modify as needed.

## 👤 Author

**Akash Chandra**
- GitHub: [@arcchandra36](https://github.com/arcchandra36)
- Email: arcchandra369@gmail.com

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Plotly for excellent visualization tools
- Scikit-learn for machine learning capabilities
- Python community for the robust ecosystem

---

**Note**: This project was developed as part of an academic assignment to demonstrate inventory management concepts and data analysis techniques.
