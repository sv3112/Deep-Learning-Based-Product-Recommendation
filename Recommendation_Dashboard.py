import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure Streamlit page
st.set_page_config(
    page_title="Customer-Centric Banking: Tailored Recommendations Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TwoTowerRecommendationSystem:
    def __init__(self, embedding_dim=64, hidden_dims=[128, 64], dropout_rate=0.3):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.user_tower = None
        self.item_tower = None
        self.model = None
        self.user_encoder = StandardScaler()
        self.item_encoder = StandardScaler()
        self.user_feature_names = None
        self.item_feature_names = None
        
    def build_user_tower(self, user_feature_dim):
        """Build the user tower network"""
        inputs = keras.Input(shape=(user_feature_dim,), name='user_features')
        x = inputs
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output embedding - using L2 normalization
        user_embedding = layers.Dense(self.embedding_dim, name='user_embedding')(x)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(user_embedding)
        
        return keras.Model(inputs=inputs, outputs=user_embedding, name='user_tower')
    
    def build_item_tower(self, item_feature_dim):
        """Build the item tower network"""
        inputs = keras.Input(shape=(item_feature_dim,), name='item_features')
        x = inputs
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output embedding - using L2 normalization
        item_embedding = layers.Dense(self.embedding_dim, name='item_embedding')(x)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item_embedding)
        
        return keras.Model(inputs=inputs, outputs=item_embedding, name='item_tower')
    
    def build_two_tower_model(self, user_feature_dim, item_feature_dim):
        """Build the complete two-tower model"""
        # Build towers
        self.user_tower = self.build_user_tower(user_feature_dim)
        self.item_tower = self.build_item_tower(item_feature_dim)
        
        # Inputs
        user_input = keras.Input(shape=(user_feature_dim,), name='user_input')
        item_input = keras.Input(shape=(item_feature_dim,), name='item_input')
        
        # Get embeddings
        user_emb = self.user_tower(user_input)
        item_emb = self.item_tower(item_input)
        
        # Compute similarity (dot product)
        similarity = layers.Dot(axes=1, name='similarity')([user_emb, item_emb])
        
        # Apply sigmoid to get probability
        output = layers.Activation('sigmoid', name='recommendation_score')(similarity)
        
        self.model = keras.Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='two_tower_recommendation'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['mae', 'mse', 'accuracy']
        )
    
    def fit(self, user_features, item_features, targets, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        # Normalize features
        user_features_scaled = self.user_encoder.fit_transform(user_features)
        item_features_scaled = self.item_encoder.fit_transform(item_features)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train the model
        history = self.model.fit(
            [user_features_scaled, item_features_scaled],
            targets,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return history
    
    def get_user_embeddings(self, user_features):
        """Get user embeddings"""
        user_features_scaled = self.user_encoder.transform(user_features)
        return self.user_tower.predict(user_features_scaled, verbose=0)
    
    def get_item_embeddings(self, item_features):
        """Get item embeddings"""
        item_features_scaled = self.item_encoder.transform(item_features)
        return self.item_tower.predict(item_features_scaled, verbose=0)
    
    def predict_similarity(self, user_features, item_features):
        """Predict similarity scores"""
        user_features_scaled = self.user_encoder.transform(user_features)
        item_features_scaled = self.item_encoder.transform(item_features)
        return self.model.predict([user_features_scaled, item_features_scaled], verbose=0)
    
    def get_feature_importance_gradient(self, user_features, item_features):
        """Fast gradient-based feature importance (alternative to SHAP)"""
        try:
            user_features_scaled = self.user_encoder.transform(user_features)
            item_features_scaled = self.item_encoder.transform(item_features)
            
            # Convert to tensors
            user_tensor = tf.convert_to_tensor(user_features_scaled, dtype=tf.float32)
            item_tensor = tf.convert_to_tensor(item_features_scaled, dtype=tf.float32)
            
            # Compute gradients
            with tf.GradientTape() as tape:
                tape.watch(user_tensor)
                tape.watch(item_tensor)
                prediction = self.model([user_tensor, item_tensor])
            
            # Get gradients
            user_gradients = tape.gradient(prediction, user_tensor)
            item_gradients = tape.gradient(prediction, item_tensor)
            
            # Compute feature importance (gradient * input)
            user_importance = (user_gradients * user_tensor).numpy()
            item_importance = (item_gradients * item_tensor).numpy()
            
            return user_importance, item_importance
            
        except Exception as e:
            print(f"Gradient importance error: {str(e)}")
            return None, None
    
    def get_embedding_contribution(self, user_features, item_features):
        """Analyze embedding contributions for interpretability"""
        try:
            user_emb = self.get_user_embeddings(user_features)
            item_emb = self.get_item_embeddings(item_features)
            
            # Compute element-wise contributions to final similarity
            similarity_contributions = user_emb * item_emb
            
            return {
                'user_embedding': user_emb,
                'item_embedding': item_emb,
                'similarity_contributions': similarity_contributions,
                'total_similarity': np.sum(similarity_contributions, axis=1)
            }
            
        except Exception as e:
            print(f"Embedding contribution error: {str(e)}")
            return None
    
    def recommend_top_k(self, user_features, all_item_features, k=10, exclude_seen=None):
        """Get top-k recommendations for a user"""
        user_features_reshaped = user_features.reshape(1, -1) if user_features.ndim == 1 else user_features
        user_emb = self.get_user_embeddings(user_features_reshaped)
        
        item_embs = self.get_item_embeddings(all_item_features)
        
        similarities = np.dot(user_emb, item_embs.T).flatten()
        
        if exclude_seen is not None:
            similarities[exclude_seen] = -np.inf
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices, top_k_scores

def create_sample_data():
    """Create sample banking data for demonstration"""
    np.random.seed(42)
    
    # Generate sample customers
    n_customers = 100
    n_products = 20
    
    customers = []
    for i in range(n_customers):
        customers.append({
            'CustomerID': f'C{i+1:03d}',
            'Age': np.random.randint(25, 70),
            'Gender': np.random.choice(['Male', 'Female']),
            'Income': np.random.normal(50000, 20000),
            'SpendingScore': np.random.randint(1, 100),
            'CreditLimit': np.random.normal(10000, 5000),
            'LoanAmount': np.random.normal(5000, 3000),
            'DigitalEngagementScore': np.random.randint(1, 100),
            'SavingsAccountBalance': np.random.normal(15000, 8000),
            'TransactionFrequency': np.random.randint(1, 50)
        })
    
    # Generate sample products
    product_names = ['Savings Account', 'Credit Card', 'Personal Loan', 'Home Loan', 
                    'Investment Fund', 'Fixed Deposit', 'Insurance Policy', 'Mutual Fund',
                    'Stock Portfolio', 'Retirement Plan', 'Education Loan', 'Car Loan',
                    'Business Loan', 'Travel Card', 'Gold Investment', 'Real Estate Fund',
                    'Health Insurance', 'Life Insurance', 'Emergency Fund', 'Pension Plan']
    
    product_types = ['Savings', 'Credit', 'Loan', 'Investment', 'Insurance']
    risk_levels = ['Low', 'Medium', 'High']
    
    products = []
    for i in range(n_products):
        products.append({
            'ProductID': f'P{i+1:03d}',
            'ProductName': product_names[i],
            'ProductType': np.random.choice(product_types),
            'RiskLevel': np.random.choice(risk_levels),
            'ExpectedYield': np.random.uniform(2, 15),
            'DurationMonths': np.random.choice([6, 12, 24, 36, 60])
        })
    
    # Generate interactions
    interactions = []
    interaction_types = ['View', 'Click', 'Apply', 'Purchase']
    
    for _ in range(500):  # Generate 500 interactions
        customer = np.random.choice(customers)
        product = np.random.choice(products)
        
        interactions.append({
            'CustomerID': customer['CustomerID'],
            'ProductID': product['ProductID'],
            'InteractionType': np.random.choice(interaction_types),
            'Timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
            **customer,
            **product
        })
    
    return pd.DataFrame(interactions)

def prepare_data_simple(data):
    """Simplified data preparation"""
    # Encode categorical variables
    le = LabelEncoder()
    
    categorical_cols = ['Gender', 'ProductType', 'RiskLevel', 'InteractionType']
    for col in categorical_cols:
        if col in data.columns:
            data[f'{col}_enc'] = le.fit_transform(data[col].astype(str))
    
    # Create user and item features
    user_cols = ['Age', 'Income', 'SpendingScore', 'CreditLimit', 'LoanAmount', 
                'DigitalEngagementScore', 'SavingsAccountBalance', 'TransactionFrequency']
    item_cols = ['ExpectedYield', 'DurationMonths']
    
    # Add encoded categorical features
    if 'Gender Type' in data.columns:
        user_cols.append('Gender Type')
    if 'Product Type' in data.columns:
        item_cols.append('Product Type')
    if 'Risk Level' in data.columns:
        item_cols.append('Risk Level')
    
    # Fill missing values
    for col in user_cols + item_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median() if data[col].dtype in ['int64', 'float64'] else 0)
    
    # Create target (1 for actual interactions, 0 for negative samples)
    data['target'] = 1
    
    return data, user_cols, item_cols

def simple_train_model(data, user_cols, item_cols):
    """Simplified model training"""
    # Prepare features
    user_features = data[['CustomerID'] + user_cols].drop_duplicates(subset=['CustomerID'])
    item_features = data[['ProductID'] + item_cols].drop_duplicates(subset=['ProductID'])
    
    # Create positive and negative samples
    all_combinations = []
    for _, user in user_features.iterrows():
        for _, item in item_features.iterrows():
            # Check if this is a real interaction
            is_interaction = len(data[(data['CustomerID'] == user['CustomerID']) & 
                                    (data['ProductID'] == item['ProductID'])]) > 0
            
            combination = {}
            combination.update({f'user_{col}': user[col] for col in user_cols})
            combination.update({f'item_{col}': item[col] for col in item_cols})
            combination['CustomerID'] = user['CustomerID']
            combination['ProductID'] = item['ProductID']
            combination['target'] = 1 if is_interaction else 0
            
            all_combinations.append(combination)
    
    train_data = pd.DataFrame(all_combinations)
    
    # Balance the dataset
    positive_samples = train_data[train_data['target'] == 1]
    negative_samples = train_data[train_data['target'] == 0].sample(
        n=min(len(positive_samples) * 3, len(train_data[train_data['target'] == 0]))
    )
    
    balanced_data = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
    
    # Prepare final features
    user_feature_cols = [f'user_{col}' for col in user_cols]
    item_feature_cols = [f'item_{col}' for col in item_cols]
    
    X_user = balanced_data[user_feature_cols].values
    X_item = balanced_data[item_feature_cols].values
    y = balanced_data['target'].values
    
    # Build and train model
    model = TwoTowerRecommendationSystem(embedding_dim=32, hidden_dims=[64, 32])
    model.build_two_tower_model(len(user_feature_cols), len(item_feature_cols))
    model.compile_model()
    
    # Store feature names
    model.user_feature_names = user_cols
    model.item_feature_names = item_cols
    
    # Train
    history = model.fit(X_user, X_item, y, epochs=20, batch_size=32)
    
    return model, balanced_data, user_feature_cols, item_feature_cols, history

def generate_recommendations_with_explanations(model, customer_id, data, user_feature_cols, item_feature_cols):
    """Generate recommendations with fast gradient-based explanations"""
    # Get customer features
    customer_data = data[data['CustomerID'] == customer_id]
    if len(customer_data) == 0:
        return None, [], []
    
    customer_data = customer_data.iloc[0]
    
    # Get all unique products
    unique_products = data.drop_duplicates(subset=['ProductID'])
    
    # Get customer's existing interactions
    customer_interactions = data[data['CustomerID'] == customer_id]['ProductID'].unique()
    
    recommendations = []
    explanations = []
    
    for _, product in unique_products.iterrows():
        if product['ProductID'] in customer_interactions:
            continue
            
        # Create feature vectors
        user_vector = np.array([customer_data[col.replace('user_', '')] for col in user_feature_cols])
        item_vector = np.array([product[col.replace('item_', '')] for col in item_feature_cols])
        
        # Predict similarity
        try:
            score = model.predict_similarity(user_vector.reshape(1, -1), item_vector.reshape(1, -1))[0][0]
            
            recommendations.append({
                'product_id': product['ProductID'],
                'product_name': product.get('ProductName', 'Unknown'),
                'product_type': product.get('ProductType', 'Unknown'),
                'risk_level': product.get('RiskLevel', 'Unknown'),
                'expected_yield': product.get('ExpectedYield', 0),
                'duration': product.get('DurationMonths', 0),
                'score': score
            })
            
            # Get fast gradient-based importance
            user_importance, item_importance = model.get_feature_importance_gradient(
                user_vector.reshape(1, -1), 
                item_vector.reshape(1, -1)
            )
            
            # Get embedding contributions
            embedding_contrib = model.get_embedding_contribution(
                user_vector.reshape(1, -1), 
                item_vector.reshape(1, -1)
            )
            
            explanations.append({
                'product_id': product['ProductID'],
                'user_importance': user_importance[0] if user_importance is not None else None,
                'item_importance': item_importance[0] if item_importance is not None else None,
                'embedding_contrib': embedding_contrib,
                'user_features': user_vector,
                'item_features': item_vector
            })
                
        except Exception as e:
            continue
    
    # Sort by score and return top 5
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    top_5_recommendations = recommendations[:5]
    
    # Get corresponding explanations
    top_5_explanations = []
    for rec in top_5_recommendations:
        for exp in explanations:
            if exp['product_id'] == rec['product_id']:
                top_5_explanations.append(exp)
                break
        else:
            top_5_explanations.append(None)
    
    return customer_data, top_5_recommendations, top_5_explanations

def display_gradient_explanation(explanation, product_name, user_feature_names, item_feature_names):
    """Display gradient-based explanation"""
    if not explanation:
        st.write("No explanation available")
        return
    
    st.markdown(f"""
    <div class="explanation-card">
        <h4>🔍 Feature Importance for {product_name}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # User feature importance
    if explanation['user_importance'] is not None:
        st.subheader("👤 User Feature Importance")
        
        user_df = pd.DataFrame({
            'Feature': [f"user_{name}" for name in user_feature_names],
            'Importance': explanation['user_importance'],
            'Value': explanation['user_features']
        })
        user_df['Abs_Importance'] = np.abs(user_df['Importance'])
        user_df = user_df.sort_values('Abs_Importance', ascending=True)
        
        # Create horizontal bar plot
        fig_user = go.Figure()
        colors = ['red' if val < 0 else 'blue' for val in user_df['Importance']]
        
        fig_user.add_trace(go.Bar(
            y=user_df['Feature'],
            x=user_df['Importance'],
            orientation='h',
            marker_color=colors,
            text=[f'{val:.3f}' for val in user_df['Importance']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Value: %{customdata}<br>Importance: %{x:.3f}<extra></extra>',
            customdata=user_df['Value']
        ))
        
        fig_user.update_layout(
            title='User Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=300,
            margin=dict(l=150)
        )
        
        st.plotly_chart(fig_user, use_container_width=True)
    
    # Item feature importance
    if explanation['item_importance'] is not None:
        st.subheader("🎯 Product Feature Importance")
        
        item_df = pd.DataFrame({
            'Feature': [f"item_{name}" for name in item_feature_names],
            'Importance': explanation['item_importance'],
            'Value': explanation['item_features']
        })
        item_df['Abs_Importance'] = np.abs(item_df['Importance'])
        item_df = item_df.sort_values('Abs_Importance', ascending=True)
        
        # Create horizontal bar plot
        fig_item = go.Figure()
        colors = ['red' if val < 0 else 'blue' for val in item_df['Importance']]
        
        fig_item.add_trace(go.Bar(
            y=item_df['Feature'],
            x=item_df['Importance'],
            orientation='h',
            marker_color=colors,
            text=[f'{val:.3f}' for val in item_df['Importance']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Value: %{customdata}<br>Importance: %{x:.3f}<extra></extra>',
            customdata=item_df['Value']
        ))
        
        fig_item.update_layout(
            title='Product Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=300,
            margin=dict(l=150)
        )
        
        st.plotly_chart(fig_item, use_container_width=True)
    
    # Embedding contributions
    if explanation['embedding_contrib']:
        st.subheader("🧠 Embedding Analysis")
        
        contrib = explanation['embedding_contrib']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Similarity Score", f"{contrib['total_similarity'][0]:.3f}")
            
        with col2:
            # Show top contributing embedding dimensions
            top_dims = np.argsort(np.abs(contrib['similarity_contributions'][0]))[-3:][::-1]
            st.write("**Top Contributing Dimensions:**")
            for i, dim in enumerate(top_dims):
                contribution = contrib['similarity_contributions'][0][dim]
                st.write(f"Dim {dim}: {contribution:.3f}")

# Main Application
def main():
    st.markdown('<h1 class="main-header">🏦 Customer-Centric Banking: Tailored Recommendations Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Navigation")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Data loading section
    st.header("📊 Data Management")
    
    # Option to use sample data or upload file
    data_option = st.radio("Choose data source:", ["Use Sample Data", "Upload Excel File"])
    
    if data_option == "Use Sample Data":
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample banking data..."):
                st.session_state.data = create_sample_data()
                st.session_state.data_loaded = True
                st.success("Sample data generated successfully!")
    
    else:
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_excel(uploaded_file)
                st.session_state.data_loaded = True
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Display data if loaded
    if st.session_state.data_loaded:
        st.subheader("Data Overview")
        st.write(f"Shape: {st.session_state.data.shape}")
        st.dataframe(st.session_state.data.head())
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", st.session_state.data['CustomerID'].nunique())
        with col2:
            st.metric("Total Products", st.session_state.data['ProductID'].nunique())
        with col3:
            st.metric("Total Interactions", len(st.session_state.data))
        with col4:
            st.metric("Avg Age", f"{st.session_state.data['Age'].mean():.1f}")
        
        # Model training section
        st.header("🤖 Model Training")
        
        if st.button("Train Recommendation Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Prepare data
                    processed_data, user_cols, item_cols = prepare_data_simple(st.session_state.data.copy())
                    
                    # Train model
                    model, train_data, user_feature_cols, item_feature_cols, history = simple_train_model(
                        processed_data, user_cols, item_cols
                    )
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.train_data = train_data
                    st.session_state.user_feature_cols = user_feature_cols
                    st.session_state.item_feature_cols = item_feature_cols
                    st.session_state.history = history
                    st.session_state.model_trained = True
                    
                    st.success("Model trained successfully with fast gradient-based explanations!")
                    
                    # Show training metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Loss", f"{history.history['loss'][-1]:.4f}")
                    with col2:
                        st.metric("Final Accuracy", f"{history.history['accuracy'][-1]:.3f}")
                    with col3:
                        st.metric("Epochs Trained", len(history.history['loss']))
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.error("Please check your data format and try again.")
            
        
        # Recommendations section
        if st.session_state.model_trained:
            st.header("🎯 Generate Personalized Recommendations")
            
            # Customer selection
            customer_ids = st.session_state.data['CustomerID'].unique()
            selected_customer = st.selectbox("Select Customer ID:", customer_ids)
            
            if st.button("Generate Recommendations"):
                with st.spinner("Generating recommendations and SHAP explanations..."):
                    try:
                        customer_info, recommendations, shap_explanations = generate_recommendations_with_explanations(
                            st.session_state.model,
                            selected_customer,
                            st.session_state.data,
                            st.session_state.user_feature_cols,
                            st.session_state.item_feature_cols
                        )
                        
                        if customer_info is not None:
                            # Display customer profile
                            st.subheader(f"👤 Customer Profile: {selected_customer}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Age:** {customer_info.get('Age', 'N/A')}")
                                st.write(f"**Gender:** {customer_info.get('Gender', 'N/A')}")
                                st.write(f"**Income:** ${customer_info.get('Income', 0):,.0f}")
                            
                            with col2:
                                st.write(f"**Spending Score:** {customer_info.get('SpendingScore', 'N/A')}")
                                st.write(f"**Credit Limit:** ${customer_info.get('CreditLimit', 0):,.0f}")
                                #st.write(f"**Digital Engagement:** {customer_info.get('DigitalEngagementScore', 'N/A')}")
                                st.write(f"**Digital Engagement:** {customer_info.get('DigitalEngagementScore', 'N/A'):.2f}")
                            
                            with col3:
                                st.write(f"**Savings Balance:** ${customer_info.get('SavingsAccountBalance', 0):,.0f}")
                                st.write(f"**Transaction Frequency:** {customer_info.get('TransactionFrequency', 'N/A')}")
                                st.write(f"**Loan Amount:** ${customer_info.get('LoanAmount', 0):,.0f}")
                            
                            # Display recommendations with SHAP
                            st.subheader("🎯 Top Recommendations with SHAP Analysis")
                            
                            if recommendations:
                                for i, (rec, shap_exp) in enumerate(zip(recommendations, shap_explanations), 1):
                                    with st.expander(f"#{i} - {rec['product_name']} (Score: {rec['score']:.3f})"):
                                        # Basic recommendation info
                                        col1, col2 = st.columns([1, 2])
                                        
                                        with col1:
                                            st.write(f"**Product ID:** {rec['product_id']}")
                                            st.write(f"**Type:** {rec['product_type']}")
                                            st.write(f"**Risk Level:** {rec['risk_level']}")
                                            st.write(f"**Recommendation Score:** {rec['score']:.3f}")
                                        if 'shap_ready' not in st.session_state:
                                            st.session_state.shap_ready = False
                                        with col2:
                                            if shap_exp is not None and st.session_state.shap_ready:
                                                st.write("**SHAP Feature Importance:**")
                                                
                                                # Create SHAP waterfall plot
                                                try:
                                                    fig_shap = plt.figure(figsize=(10, 6))
                                                    shap.plots.waterfall(shap_exp, show=False)
                                                    st.pyplot(fig_shap)
                                                    plt.close()
                                                except Exception as e:
                                                    st.write("SHAP visualization not available")
                                                    
                                                # Display top SHAP values
                                                if hasattr(shap_exp, 'values') and hasattr(shap_exp, 'data'):
                                                    shap_df = pd.DataFrame({
                                                        'Feature': shap_exp.feature_names if hasattr(shap_exp, 'feature_names') else range(len(shap_exp.values)),
                                                        'SHAP Value': shap_exp.values,
                                                        'Feature Value': shap_exp.data
                                                    })
                                                    shap_df['Abs_SHAP'] = abs(shap_df['SHAP Value'])
                                                    shap_df = shap_df.sort_values('Abs_SHAP', ascending=False).head(5)
                                                    
                                                    st.write("**Top 5 Most Important Features:**")
                                                    for _, row in shap_df.iterrows():
                                                        direction = "↑" if row['SHAP Value'] > 0 else "↓"
                                                        st.write(f"{direction} **{row['Feature']}**: {row['Feature Value']:.2f} (SHAP: {row['SHAP Value']:.3f})")
                                            else:
                                                st.write("SHAP explanations not available for this recommendation")
                                        
                                        # Additional product details
                                        st.write("**Why this recommendation?**")
                                        if rec['product_type'] == 'Credit Card':
                                            st.write("- Based on your spending patterns and credit profile")
                                            st.write("- Matches your current financial capacity")
                                        elif rec['product_type'] == 'Savings Account':
                                            st.write("- Aligned with your savings goals and income level")
                                            st.write("- Suitable for your transaction frequency")
                                        elif rec['product_type'] == 'Investment':
                                            st.write("- Matches your risk tolerance and investment capacity")
                                            st.write("- Appropriate for your age and financial goals")
                                        elif rec['product_type'] == 'Loan':
                                            st.write("- Based on your creditworthiness and income")
                                            st.write("- Suitable for your current debt situation")
                            else:
                                st.info("No recommendations generated. Please check the model training.")
                        else:
                            st.error("Customer information not found. Please select a valid customer ID.")
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        st.error("Please ensure the model is properly trained and try again.")
            
            
    # Footer
    st.markdown("---")
    st.markdown("**Banking Recommendation System** - Explainable AI for Financial Services")
    st.markdown("Built with Streamlit, and Deep Learning")

if __name__ == "__main__":
    main()
                  
