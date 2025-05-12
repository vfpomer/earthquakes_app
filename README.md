# 🌍 Earthquake Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Data](https://img.shields.io/badge/Data-USGS-yellow.svg)

Advanced seismic activity visualization and analysis platform leveraging Streamlit and Python. Offers real-time interactive analysis of earthquake data through dynamic visualizations and statistical computations.

![Earthquakes App img](img\image.jpg)

## 🚀 Key Features

### 📊 Data Analysis
- **Real-time Filtering System**
  - Temporal: Custom date range selection
  - Spatial: Geographic region filtering
  - Seismic: Magnitude and depth parameters
  - Categorical: Event type classification

### 🗺️ Visualization Components
- **Geographic Analysis**
  ```python
  - Interactive global mapping with magnitude representation
  - Heat maps showing activity concentration
  - DBSCAN-based cluster analysis
  - Region-based statistical summaries
  ```

- **Temporal Analysis**
  ```python
  - Time series visualization
  - Daily/weekly pattern recognition
  - Hourly distribution charts
  - Trend identification algorithms
  ```

- **Statistical Analysis**
  ```python
  - Magnitude/depth distributions
  - Correlation matrices
  - Regional comparisons
  - Advanced statistical metrics
  ```

## 🛠️ Technical Stack

### Core Dependencies
```python
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
statsmodels>=0.13.0
```

### Data Structure
```json
{
    "time": "datetime64[ns]",
    "latitude": "float64",
    "longitude": "float64",
    "depth": "float64",
    "magnitude": "float64",
    "place": "string",
    "type": "string"
}
```

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/earthquakes_app.git
cd earthquakes_app

# Create virtual environment
python -m venv earthquakesenv

# Activate environment
source earthquakesenv/bin/activate  # Linux/Mac
earthquakesenv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Application Launch
```bash
# Ensure data file is present in data/
streamlit run app.py
```

## 📊 Analysis Modules

### 1. General Summary
- Comprehensive metrics overview
- Statistical distributions
- Top active regions ranking

### 2. Geographic Analysis
- Interactive global mapping
- Density visualization
- Cluster identification

### 3. Temporal Analysis
- Time-based pattern recognition
- Periodic analysis
- Trend detection algorithms

### 4. Advanced Analytics
- Correlation studies
- Regional comparisons
- Custom metric analysis

## 🔧 Configuration

The application supports various configuration options through `config.yaml`:

```yaml
data:
  source: "data/all_month.csv"
  update_interval: 3600

visualization:
  default_map_style: "dark"
  color_scheme: "viridis"
  plot_height: 600
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork repository
2. Create feature branch (`git checkout -b feature/Enhancement`)
3. Commit changes (`git commit -m 'Add Enhancement'`)
4. Push branch (`git push origin feature/Enhancement`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)


