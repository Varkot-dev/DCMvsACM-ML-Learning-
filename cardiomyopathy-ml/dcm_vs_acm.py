import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
from sklearn.feature_selection import RFE, SelectFromModel, RFECV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import time
import numpy as np
import sys
from scipy import stats
import mygene
from statsmodels.stats.multitest import multipletests
import umap
import gseapy as gp
from sklearn.decomposition import PCA
import networkx as nx
from scipy.stats import spearmanr
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime
import os
import shap
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        attn_output, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + ff_output)
        return x, attn_weights

class GeneTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(256) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add sequence dimension
        
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn = block(x)
            attention_weights.append(attn)
            
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.classifier(x)
        
        if return_attention:
            return x, attention_weights
        return x

def train_deep_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {accuracy:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
    
    return best_model

def create_gene_network(X, gene_names, threshold=0.7):
    """Create a gene correlation network."""
    # Calculate correlation matrix
    corr_matrix, p_values = spearmanr(X)
    
    # Create network
    G = nx.Graph()
    
    # Add edges for highly correlated genes
    for i in range(len(gene_names)):
        for j in range(i+1, len(gene_names)):
            if abs(corr_matrix[i,j]) > threshold:
                G.add_edge(gene_names[i], gene_names[j], 
                          weight=abs(corr_matrix[i,j]))
    
    return G

def plot_gene_network(G, output_file):
    """Plot gene correlation network."""
    plt.figure(figsize=(15, 15))
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    node_sizes = [v * 3000 for v in centrality.values()]
    
    # Calculate edge weights
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u,v in edges]
    
    # Spring layout with adjusted parameters
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Gene Correlation Network\n(Edges show correlations > 0.7)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def perform_gsea(gene_list, gene_scores, output_dir='gsea_results'):
    """Perform Gene Set Enrichment Analysis."""
    # Prepare ranked gene list
    ranked_genes = pd.Series(gene_scores, index=gene_list)
    ranked_genes = ranked_genes.sort_values(ascending=False)
    
    # Run enrichment analysis using enrichr instead of gsea
    enr = gp.enrichr(gene_list=gene_list,
                     gene_sets=['KEGG_2021_Human',
                               'GO_Biological_Process_2021',
                               'WikiPathways_2021_Human',
                               'MSigDB_Hallmark_2020'],
                     organism='Human',
                     outdir=output_dir)
    
    return enr.results

def plot_attention_weights(attention_weights, gene_names, layer_idx=0, head_idx=0, top_k=20, output_file=None):
    """Plot attention weights for a specific layer and head."""
    # Get attention weights for specified layer and head
    attn = attention_weights[layer_idx][head_idx].cpu().detach().numpy()
    
    # Get top k genes based on maximum attention weight
    max_attn = np.max(attn, axis=1)
    top_k_indices = np.argsort(max_attn)[-top_k:]
    
    # Create heatmap for top k genes
    plt.figure(figsize=(12, 10))
    sns.heatmap(attn[top_k_indices][:, top_k_indices],
                xticklabels=[gene_names[i] for i in top_k_indices],
                yticklabels=[gene_names[i] for i in top_k_indices],
                cmap='YlOrRd')
    
    plt.title(f'Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_gene_interactions(attention_weights, gene_names, layer_idx=0, threshold=0.5, output_file=None):
    """Create a network visualization of gene interactions based on attention weights."""
    # Average attention weights across heads
    avg_attn = torch.mean(attention_weights[layer_idx], dim=0).cpu().detach().numpy()
    
    # Create network
    G = nx.Graph()
    
    # Add edges for strong attention weights
    for i in range(len(gene_names)):
        for j in range(i+1, len(gene_names)):
            if avg_attn[i,j] > threshold:
                G.add_edge(gene_names[i], gene_names[j],
                          weight=float(avg_attn[i,j]))
    
    # Plot network
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    node_sizes = [v * 3000 for v in centrality.values()]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color='lightblue', alpha=0.6)
    
    # Draw edges with varying width based on attention weight
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u,v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.3)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"Gene Interaction Network (Layer {layer_idx+1})\nEdges show attention > {threshold}")
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return G

def analyze_attention_patterns(model, val_loader, gene_names, device):
    """Analyze attention patterns across the validation set."""
    model.eval()
    all_attention_weights = []
    max_batch_size = val_loader.batch_size
    
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            
            # If this batch is smaller than max_batch_size, pad it
            if batch_X.size(0) < max_batch_size:
                padding_size = max_batch_size - batch_X.size(0)
                padding = torch.zeros((padding_size, batch_X.size(1)), device=device)
                batch_X = torch.cat([batch_X, padding], dim=0)
            
            _, attention_weights = model(batch_X, return_attention=True)
            all_attention_weights.append([w.cpu() for w in attention_weights])
    
    # Average attention weights across batches
    avg_attention = []
    for layer_idx in range(len(all_attention_weights[0])):
        layer_attention = torch.mean(torch.stack([batch[layer_idx] for batch in all_attention_weights]), dim=0)
        avg_attention.append(layer_attention)
    
    return avg_attention

def create_results_pdf(results_dict, file_paths, gene_results, gsea_results):
    """Create a comprehensive PDF report of all results."""
    doc = SimpleDocTemplate(
        "cardiomyopathy_analysis_report.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Collect the elements that will make up our document
    elements = []
    styles = getSampleStyleSheet()
    
    # Create custom style for headers
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    
    # Title
    elements.append(Paragraph("Cardiomyopathy Analysis Report", header_style))
    elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Dataset Information
    elements.append(Paragraph("1. Dataset Overview", header_style))
    elements.append(Paragraph(f"Total cells analyzed: {len(results_dict['y_balanced'])}", styles['Normal']))
    elements.append(Paragraph(f"Number of features: {results_dict['X_balanced'].shape[1]}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Model Performance
    elements.append(Paragraph("2. Model Performance", header_style))
    for name, res in results_dict['results'].items():
        elements.append(Paragraph(f"\n{name} Results:", styles['Heading2']))
        elements.append(Paragraph(f"Cross-validation ROC-AUC: {res['cv_scores'].mean():.3f} (±{res['cv_scores'].std() * 2:.3f})", styles['Normal']))
        elements.append(Paragraph("Classification Report:", styles['Normal']))
        elements.append(Paragraph(f"<pre>{classification_report(results_dict['y_test'], res['y_pred'], target_names=results_dict['le'].classes_)}</pre>", styles['Code']))
    
    # Visualizations
    elements.append(Paragraph("3. Visualizations", header_style))
    
    # Add all generated plots
    for file_path in file_paths:
        if file_path.endswith(('.png', '.jpg')):
            img = Image(file_path)
            img.drawHeight = 4*inch
            img.drawWidth = 6*inch
            elements.append(img)
            elements.append(Spacer(1, 12))
    
    # Gene Analysis
    elements.append(Paragraph("4. Gene Analysis", header_style))
    
    # Top Genes Table
    elements.append(Paragraph("Top 20 Significant Genes:", styles['Heading2']))
    top_genes = gene_results.head(20)
    gene_data = [[col for col in top_genes.columns]] + \
                [[str(round(x, 4)) if isinstance(x, float) else str(x) for x in row] 
                 for row in top_genes.values]
    gene_table = Table(gene_data)
    gene_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(gene_table)
    elements.append(Spacer(1, 12))
    
    # Pathway Analysis
    elements.append(Paragraph("5. Pathway Analysis", header_style))
    if gsea_results:
        for pathway_db, results in gsea_results.items():
            elements.append(Paragraph(f"\n{pathway_db} Top Pathways:", styles['Heading2']))
            top_pathways = results.head(10)
            pathway_data = [[col for col in ['Term', 'NES', 'FDR q-val', 'Genes']]] + \
                          [[row['Term'], f"{row['NES']:.3f}", f"{row['FDR q-val']:.2e}", row['Leading_edge'][:100] + '...'] 
                           for _, row in top_pathways.iterrows()]
            pathway_table = Table(pathway_data)
            pathway_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(pathway_table)
            elements.append(Spacer(1, 12))
    
    # Network Analysis
    if os.path.exists('attention_plots'):
        elements.append(Paragraph("6. Network Analysis", header_style))
        for file_name in os.listdir('attention_plots'):
            if file_name.endswith('.txt'):
                with open(os.path.join('attention_plots', file_name), 'r') as f:
                    content = f.read()
                elements.append(Paragraph(f"Network Statistics - {file_name}", styles['Heading2']))
                elements.append(Paragraph(f"<pre>{content}</pre>", styles['Code']))
                elements.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(elements)
    print("\nComprehensive PDF report generated: cardiomyopathy_analysis_report.pdf")

try:
    print("Starting analysis...")
    total_start_time = time.time()

    # 1. Load and prepare data
    print("\nLoading data from h5ad file...")
    start_time = time.time()
    adata_cm = sc.read_h5ad("cardiomyocytes.h5ad")
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    # Take 10k cells
    print("\nTaking a random subset of 10000 cells...")
    subset_idx = np.random.choice(adata_cm.shape[0], size=10000, replace=False)
    adata_cm = adata_cm[subset_idx].copy()
    print(f"Subset shape: {adata_cm.shape}")

    # Print initial distribution
    print("\nInitial disease distribution:")
    print(adata_cm.obs['disease'].value_counts())

    # Filter for DCM and ACM
    print("\nFiltering for DCM and ACM samples...")
    adata_cm = adata_cm[adata_cm.obs['disease'].isin(['dilated cardiomyopathy', 'arrhythmogenic right ventricular cardiomyopathy'])]
    print("\nDisease distribution after filtering:")
    print(adata_cm.obs['disease'].value_counts())

    # 2. Prepare expression matrix
    print("\nPreparing expression matrix...")
    X = adata_cm.X.toarray()
    y = adata_cm.obs['disease'].astype(str)

    # Label encoding for classification
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("\nClass encoding:")
    for i, label in enumerate(le.classes_):
        print(f"{label}: {i}")

    # 3. Feature selection
    print("\nPerforming feature selection...")
    
    # Use optimal number of HVG (2500 based on previous optimization)
    sc.pp.highly_variable_genes(adata_cm, n_top_genes=2500)
    hvg_mask = adata_cm.var['highly_variable']
    X_hvg = adata_cm[:, hvg_mask].X.toarray()
    gene_ids = adata_cm[:, hvg_mask].var_names.tolist()

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hvg)

    # 4. Balance classes using SMOTE
    print("\nBalancing classes using SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y_encoded)
    print("Class distribution after balancing:")
    unique, counts = np.unique(y_balanced, return_counts=True)
    for label, count in zip(le.classes_, counts):
        print(f"{label}: {count}")

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    # 6. Model Training and Evaluation
    base_models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    param_grids = {
        'Random Forest': {
            'n_estimators': [100],  # 100 was sufficient in previous runs
            'max_depth': [15],      # 15 performed best previously
            'min_samples_split': [5] # 5 was optimal
        },
        'XGBoost': {
            'n_estimators': [100],  # 100 was sufficient
            'max_depth': [6],       # 6 performed well
            'learning_rate': [0.1]  # 0.1 was consistently better
        },
        'Logistic Regression': {
            'C': [1.0],            # 1.0 was optimal
            'max_iter': [1000]     # Required for convergence
        }
    }

    # Results storage
    results = {}
    best_models = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3 folds

    for name, model in base_models.items():
        print(f"\nOptimizing {name}...")
        start_time = time.time()
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Store best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Cross-validation on full dataset
        cv_scores = cross_val_score(best_model, X_balanced, y_balanced, cv=cv, scoring='roc_auc')
        print(f"Final cross-validation ROC-AUC: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")
        
        # Train final model
        best_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Store results
        results[name] = {
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': grid_search.best_params_,
            'training_time': time.time() - start_time
        }

    # 7. Plot Results
    # Confusion Matrix for each model
    plt.figure(figsize=(15, 5))
    for i, (name, res) in enumerate(results.items(), 1):
        plt.subplot(1, 3, i)
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d',
                   xticklabels=le.classes_, yticklabels=le.classes_,
                   cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'][:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Biological Interpretation
    print("\nPerforming biological interpretation...")
    
    # Convert ENSEMBL IDs to gene names
    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(gene_ids, scopes='ensembl.gene', fields=['symbol', 'name'], species='human')
    
    # Create gene name mapping
    gene_names = {}
    for info in gene_info:
        if 'symbol' in info:
            gene_names[info.get('query', '')] = info['symbol']
        else:
            gene_names[info.get('query', '')] = info.get('query', '')

    # SHAP Analysis for XGBoost
    print("\nCalculating SHAP values for XGBoost...")
    explainer = shap.TreeExplainer(best_models['XGBoost'])
    shap_values = explainer.shap_values(X_test)
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, 
                     feature_names=[gene_names.get(g, g) for g in gene_ids],
                     show=False,
                     max_display=20)
    plt.title('Top 20 Genes Driving Classification (SHAP Analysis)')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Get Logistic Regression coefficients
    lr_coef = pd.DataFrame({
        'gene': [gene_names.get(g, g) for g in gene_ids],
        'coefficient': best_models['Logistic Regression'].coef_[0]
    })
    lr_coef['abs_coef'] = abs(lr_coef['coefficient'])
    lr_coef = lr_coef.sort_values('abs_coef', ascending=False)

    # Plot top genes from Logistic Regression
    plt.figure(figsize=(12, 8))
    plt.bar(range(20), lr_coef['coefficient'][:20])
    plt.xticks(range(20), lr_coef['gene'][:20], rotation=45, ha='right')
    plt.title('Top 20 Genes Driving Classification (Logistic Regression)')
    plt.xlabel('Genes')
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.savefig('logreg_top_genes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed results
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Results Summary\n")
        f.write("============================\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"Total cells analyzed: {len(y_balanced)}\n")
        f.write(f"Number of features (genes): {X_balanced.shape[1]}\n")
        f.write(f"Total runtime: {(time.time() - total_start_time)/60:.1f} minutes\n\n")
        
        for name, res in results.items():
            f.write(f"\n{name} Results:\n")
            f.write("-" * (len(name) + 9) + "\n")
            f.write(f"Cross-validation ROC-AUC: {res['cv_scores'].mean():.3f} (±{res['cv_scores'].std() * 2:.3f})\n")
            f.write(f"Training time: {res['training_time']/60:.1f} minutes\n")
            f.write("\nBest Parameters:\n")
            for param, value in res['best_params'].items():
                f.write(f"{param}: {value}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, res['y_pred'], target_names=le.classes_))
            f.write("\n" + "="*50 + "\n")
        
        # Save top genes from both analyses
        f.write("\nTop 20 Genes (SHAP Analysis - XGBoost):\n")
        shap_importance = np.abs(shap_values).mean(0)
        shap_df = pd.DataFrame({
            'gene': [gene_names.get(g, g) for g in gene_ids],
            'importance': shap_importance
        })
        shap_df = shap_df.sort_values('importance', ascending=False)
        for i, row in shap_df.head(20).iterrows():
            f.write(f"{i+1}. {row['gene']}: {row['importance']:.4f}\n")
        
        f.write("\nTop 20 Genes (Logistic Regression):\n")
        for i, row in lr_coef.head(20).iterrows():
            f.write(f"{i+1}. {row['gene']}: {row['coefficient']:.4f}\n")

    print("\nAnalysis completed! Results saved in:")
    print("1. confusion_matrices.png - Confusion matrices for all models")
    print("2. roc_curves.png - ROC curves for all models")
    print("3. shap_summary.png - SHAP analysis of top genes")
    print("4. logreg_top_genes.png - Logistic Regression feature importance")
    print("5. classification_report.txt - Detailed classification metrics and gene lists")
    print(f"\nTotal runtime: {(time.time() - total_start_time)/60:.1f} minutes")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("Error occurred at:")
    import traceback
    traceback.print_exc()
