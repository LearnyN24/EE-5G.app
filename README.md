# ML-based Energy Efficiency Optimization for 5G Networks and Beyond

This Streamlit app demonstrates the use of machine learning (ML) techniques for optimizing energy efficiency in 5G and future wireless networks. It provides a secure, multi-page interface for:

- Data-driven energy consumption analysis and prediction
- Interactive ML model training and visualization
- Real-time AI-powered insights using Llama-3.1
- Future research directions in 5G energy optimization

## Features
- **Secure User Authentication**: Login/Registration system with encrypted password storage
- **Data Analysis & ML**: 
  - CSV data upload and preprocessing
  - Automated feature selection
  - Linear regression model training
  - Performance visualization with actual vs predicted plots
- **Interactive AI Assistant**: Real-time Q&A about energy efficiency using Llama-3.1
- **Modern UI/UX**:
  - Responsive navigation sidebar
  - User profile management
  - Clean, professional styling
- **Research Insights**:
  - Future directions in federated learning
  - Digital twins for network simulation
  - Cross-layer ML approaches
  - Applications of generative AI
  - Quantum ML possibilities

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```

## Notes
- The real-time inference uses the API i created
- User data is stored securely with SHA-256 password hashing
- The ML demo supports any CSV dataset with numeric features
- Visualizations include interactive scatter plots and performance metrics