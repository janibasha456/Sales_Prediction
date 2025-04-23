import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# Configuration
st.set_page_config(page_title="Sales AI Predictor", page_icon="📈", layout="wide")

# Model loading function
@st.cache_resource
def load_model(model_name):
    model_mapping = {
        'Linear Regression': 'linear_model.pkl',
        'Random Forest': 'rf_model.pkl',
        'LightGBM': 'lgb_model.pkl',
        'Decision Tree': 'dt_model.pkl',
        'KNN': 'knn_model.pkl'
    }
    return joblib.load(model_mapping[model_name])

# Load error metrics (replace with actual values)
error_metrics = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'LightGBM', 'Decision Tree', 'KNN'],
    'MSE': [14940.223047941236, 5.711765214241612, 26.385809480287097, 21.222844162138347, 86.10433925715962],
    'MAE':  [80.76403252193506, 0.20033375238656517, 3.2131346019990907, 0.60078278748715, 5.081625583786166],
    'R²': [0.8099106237423851, 0.9999273273308289, 0.9996642846596021, 0.9997299747670267, 0.9989044661455223],
    'MAPE': [280.54445710691624, 0.04758166523516064, 3.289453543973639, 0.12641181803389076, 4.259948521922456]    # Your MAPE values
}).set_index('Model')

# Feature importance placeholder (replace with actual data)
feature_importance = pd.DataFrame({
    'Feature': ['Unit Price', 'Quantity', 'Promotion', 'Holiday', 'Weekend',
                'GDP', 'Inflation', 'Unemployment', 'Sentiment'],
    'Importance': np.random.rand(9)
}).sort_values('Importance', ascending=False)

# ==============================================
# Sidebar Configuration
# ==============================================
with st.sidebar:
    st.header("⚙️ Configuration")
    model_option = st.radio(
        "Select Prediction Model",
        error_metrics.index.tolist(),
        help="Choose the machine learning model for prediction"
    )
    
    st.header("📥 Input Features")
    
    # Common features
    unit_price = st.number_input('Unit Price ($)', 0.0, 1000.0, 49.99, 0.1)
    quantity_sold = st.number_input('Quantity Sold', 0, 1000, 100)
    promotion = st.selectbox('Promotion', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    holiday = st.selectbox('Holiday', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    is_weekend = st.selectbox('Weekend', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    
    # Economic features for complex models
    if model_option in ['Random Forest', 'LightGBM', 'Decision Tree', 'KNN']:
        st.subheader("🌐 Economic Indicators")
        gdp_growth = st.number_input('GDP Growth (%)', -10.0, 20.0, 2.5, 0.1)
        inflation_rate = st.number_input('Inflation Rate (%)', 0.0, 50.0, 2.0, 0.1)
        unemployment_rate = st.number_input('Unemployment Rate (%)', 0.0, 30.0, 5.0, 0.1)
        market_sentiment = st.slider('Market Sentiment', 0.0, 10.0, 7.0)

# ==============================================
# Main Interface
# ==============================================
st.title("📊 Advanced Sales Prediction System")

# Load model
model = load_model(model_option)

# Prepare input features
common_features = [unit_price, quantity_sold, promotion, holiday, is_weekend]
if model_option in ['Random Forest', 'LightGBM', 'Decision Tree', 'KNN']:
    economic_features = [gdp_growth, inflation_rate, unemployment_rate, market_sentiment]
    input_features = np.array([common_features + economic_features])
    feature_names = [
        'Unit Price', 'Quantity', 'Promotion', 'Holiday', 'Weekend',
        'GDP Growth', 'Inflation', 'Unemployment', 'Sentiment'
    ]
else:
    input_features = np.array([common_features])
    feature_names = ['Unit Price', 'Quantity', 'Promotion', 'Holiday', 'Weekend']

# ==============================================
# Prediction Section
# ==============================================
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("🔮 Prediction Panel")
    if st.button('Calculate Sales Prediction', use_container_width=True):
        try:
            prediction = model.predict(input_features)[0]
            st.metric("Predicted Sales", f"${prediction:,.2f}", 
                     help="Estimated total sales based on input features")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

with col2:
    st.subheader("📋 Feature Summary")
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_features[0]
    })
    st.dataframe(feature_df.set_index('Feature'), use_container_width=True)

# ==============================================
# Advanced Analytics
# ==============================================
st.header("📈 Advanced Analytics")

# Tabs for different analysis sections
tab1, tab2, tab3 = st.tabs(["Model Performance", "Error Analysis", "Feature Impact"])

with tab1:
    st.subheader("Model Comparison")
    
    # Metric selection
    metric = st.selectbox("Select Metric", error_metrics.columns.tolist())
    
    # Interactive bar chart
    fig = px.bar(
        error_metrics.reset_index(),
        x='Model',
        y=metric,
        color='Model',
        text=metric,
        title=f"{metric} Comparison"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.dataframe(
        error_metrics.style.format({
            'MAE': '{:.1f}',
            'MSE': '{:,.0f}',
            'R²': '{:.2%}',
            'MAPE': '{:.2%}'
        }),
        use_container_width=True
    )

with tab2:
    st.subheader("Error Diagnostics")
    
    col1, col2 = st.columns(2)
    with col1:
        # Residual analysis placeholder
        st.write("**Residual Distribution**")
        residual_data = pd.DataFrame({
            'Models': np.random.choice(error_metrics.index.tolist(), 500),
            'Residuals': np.random.normal(0, 1, 500)
        })
        fig = px.box(residual_data, x='Models', y='Residuals')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Error correlation
        st.write("**Error Metric Relationships**")
        fig = px.scatter_matrix(error_metrics.reset_index(), dimensions=error_metrics.columns)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Feature Impact Analysis")
    
    # Feature importance visualization
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance Ranking'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation matrix placeholder
    st.write("**Feature Correlation Matrix**")
    corr_matrix = pd.DataFrame(np.random.randn(9, 9), 
                              columns=feature_importance.Feature,
                              index=feature_importance.Feature)
    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

# ==============================================
# Documentation & Export
# ==============================================
st.header("📚 Documentation & Tools")

expander = st.expander("📖 Technical Documentation")
with expander:
    st.markdown("""
    ## Model Performance Metrics
    - **MAE (Mean Absolute Error):** Average absolute difference between predictions and actual values
    - **MSE (Mean Squared Error):** Squared average of prediction errors
    - **R² (R-Squared):** Proportion of variance explained by the model
    - **MAPE (Mean Absolute Percentage Error):** Percentage representation of MAE

    ## Model Selection Guide
    | Model Type          | Best Use Case                          |
    |---------------------|----------------------------------------|
    | Linear Regression   | Baseline models, linear relationships |
    | Tree-Based Models   | Complex non-linear patterns           |
    | KNN                 | Local pattern recognition             |
    """)

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Export Input Features",
        data=pd.DataFrame([input_features[0]], columns=feature_names).to_csv(),
        file_name="sales_prediction_inputs.csv"
    )

with col2:
    st.download_button(
        label="📥 Export Performance Metrics",
        data=error_metrics.to_csv(),
        file_name="model_performance.csv"
    )