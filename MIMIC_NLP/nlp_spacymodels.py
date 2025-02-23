import pandas as pd
import numpy as np
import spacy
import scispacy
from scispacy.linking import EntityLinker
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from tqdm import tqdm
from collections import Counter
from spacy import displacy
from IPython.display import HTML, display
from tqdm.auto import tqdm
import networkx as nx
import seaborn as sns
import string
from nltk.corpus import stopwords
import nltk
import re
import ssl

# Download NLTK stop words if not already downloaded and resolve ssl error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

#Load datasets
print("Loading data...")
notes_df = pd.read_csv('./data/NOTEEVENTS.csv.gz', low_memory=False)
print("\nNOTEEVENTS Sample:")
print(notes_df.head())
print("\nNOTEEVENTS Shape:", notes_df.shape)

diagnoses_df = pd.read_csv('./data/DIAGNOSES_ICD.csv.gz', low_memory=False)
print("\nDIAGNOSES Sample:")
print(diagnoses_df.head())
print("\nDIAGNOSES Shape:", diagnoses_df.shape)

# Diabetes Analysis
print("\nFiltering for diabetes patients...")
diabetes_patients = diagnoses_df[diagnoses_df['ICD9_CODE'].str.startswith('250', na=False)]
print("\nDiabetes Patients Sample:")
print(diabetes_patients.head())
print("\nDiabetes Patients Count:", len(diabetes_patients))

print("\nMerging diabetes datasets...")
diabetes_notes = pd.merge(diabetes_patients, notes_df, on=['SUBJECT_ID', 'HADM_ID'])
print("\nDiabetes Notes Sample:")
print(diabetes_notes.head())
print(f"Merged dataset has {len(diabetes_notes)} rows")

diabetes_sample_notes = diabetes_notes[diabetes_notes['CATEGORY'] == 'Discharge summary'].sample(n=500, random_state=42)

# Load both models
print("\nLoading Spacy models...")
nlp_sci = spacy.load("en_core_sci_lg")
nlp_web = spacy.load("en_core_web_sm")

# Add UMLS linker to scientific model
nlp_sci.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

def extract_and_display_entities(text, model, title):
    """
    Extract and display entities in a format suitable for both notebook and script environments
    """
    doc = model(text)
    
    print(f"\n{title} Entities:")
    print("-" * 100)
    
    # Create a formatted text display of entities
    for ent in doc.ents:
        # Color coding for console
        color = '\033[92m'  # green
        end_color = '\033[0m'
        print(f"{color}Entity:{end_color} {ent.text:<50} {color}Label:{end_color} {ent.label_:<20}")
    
    # Group entities by label
    entities_by_label = {}
    for ent in doc.ents:
        if ent.label_ not in entities_by_label:
            entities_by_label[ent.label_] = []
        entities_by_label[ent.label_].append(ent.text)

    # Display summary statistics
    print("\nEntity Type Statistics:")
    print("-" * 100)
    for label, entities in entities_by_label.items():
        print(f"Type: {label:<20} Count: {len(entities):<5} Examples: {', '.join(entities[:3])}")

    # Create visualization using matplotlib
    if len(doc.ents) > 0:
        # Count entity types
        label_counts = Counter([ent.label_ for ent in doc.ents])
        
        # Create bar plot of entity types
        plt.figure(figsize=(12, 6))
        bars = plt.bar(label_counts.keys(), label_counts.values())
        plt.title(f'Entity Types Distribution in {title}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    return [(ent.text, ent.label_) for ent in doc.ents]


# Process sample text with both models
print("\nExtracting entities...")
sample_text = diabetes_sample_notes['TEXT'].iloc[0]
sci_entities = extract_and_display_entities(sample_text, nlp_sci, "Scientific Model")
web_entities = extract_and_display_entities(sample_text, nlp_web, "Web Model")

# Optional: Add entity network visualization
def visualize_entity_network(entities, title):
    """
    Create a network visualization of entities and their relationships
    """
    G = nx.Graph()
    
    # Add nodes for each entity
    for entity, label in entities:
        G.add_node(entity, label=label)
        
    # Add edges between entities that appear close to each other in text
    for i in range(len(entities)-1):
        G.add_edge(entities[i][0], entities[i+1][0])
    
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    labels = nx.get_node_attributes(G, 'label')
    unique_labels = set(labels.values())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    node_colors = [color_map[labels[node]] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color_map[label],
                                 label=label, markersize=10)
                      for label in unique_labels]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title(f'Entity Relationships Network - {title}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize entity networks
visualize_entity_network(sci_entities, "Scientific Model")
visualize_entity_network(web_entities, "Web Model")

# Add summary comparison
print("\nComparison of Models:")
print("-" * 100)
sci_labels = Counter([label for _, label in sci_entities])
web_labels = Counter([label for _, label in web_entities])

print("\nScientific Model found:")
for label, count in sci_labels.most_common():
    print(f"- {count} {label} entities")

print("\nWeb Model found:")
for label, count in web_labels.most_common():
    print(f"- {count} {label} entities")

# Enable progress bar for pandas
tqdm.pandas()

def clean_text(text):
    """
    Comprehensive text cleaning function
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    return ''

def preprocess_text(text, nlp_model):
    """
    Comprehensive text preprocessing function
    """
    # Get stop words from both NLTK and spaCy
    stop_words = set(stopwords.words('english'))
    spacy_stop_words = nlp_model.Defaults.stop_words
    all_stop_words = stop_words.union(spacy_stop_words)
    
    # Clean the text first
    text = clean_text(text)
    
    # Process with spaCy
    doc = nlp_model(text)
    
    # Get tokens that:
    # - are not punctuation
    # - are not space
    # - are not stop words
    # - are not single characters
    # - have length > 2
    tokens = [token.text for token in doc 
             if not token.is_punct 
             and not token.is_space
             and token.text.lower() not in all_stop_words
             and len(token.text) > 2]
    
    return tokens

def create_word2vec_and_tsne(notes_df, nlp_model, condition_name):
    print(f"\nProcessing {condition_name} notes...")
    
    # Preprocess all texts
    print("Tokenizing and cleaning texts...")
    notes_df['tokens'] = notes_df['TEXT'].progress_apply(
        lambda x: preprocess_text(x, nlp_model)
    )
    
    # Remove empty token lists
    notes_df = notes_df[notes_df['tokens'].map(len) > 0]
    
    print("Creating Word2Vec model...")
    w2v_model = Word2Vec(sentences=notes_df['tokens'], 
                        vector_size=100, 
                        window=5, 
                        min_count=2,  # Increased min_count to filter rare words
                        workers=4)
    
    # Visualization of similar words
    visualize_similar_words(w2v_model, condition_name)
    
    # TSNE Visualization
    print("\nCreating t-SNE visualization...")
    labels = []
    tokens = []
    
    # Get most common words for visualization
    all_words = [word for token_list in notes_df['tokens'] for word in token_list]
    word_freq = Counter(all_words)
    most_common = [word for word, count in word_freq.most_common(100)]
    
    for word in most_common:
        if word in w2v_model.wv:
            tokens.append(w2v_model.wv[word])
            labels.append(word)
    
    # Convert tokens list to numpy array
    tokens_array = np.array(tokens)
    
    tsne_model = TSNE(perplexity=min(40, len(tokens_array)-1), 
                      n_components=2, 
                      init='pca', 
                      max_iter=2500, 
                      random_state=23)
    
    new_values = tsne_model.fit_transform(tokens_array)
    
    plt.figure(figsize=(16, 16))
    plt.scatter(new_values[:, 0], new_values[:, 1])
    
    for i, label in enumerate(labels):
        plt.annotate(label,
                    xy=(new_values[i, 0], new_values[i, 1]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    
    plt.title(f"t-SNE visualization for {condition_name}\n(Most Common Words)")
    plt.show()

def visualize_similar_words(w2v_model, condition_name):
    try:
        # Get similar words
        similar_words = w2v_model.wv.most_similar(condition_name.lower(), topn=10)
        words, scores = zip(*similar_words)
        
        # Bar Plot
        plt.figure(figsize=(12, 6))
        colors = sns.color_palette("husl", len(words))
        bars = plt.barh(range(len(words)), scores, align='center', color=colors)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Similarity Score')
        plt.title(f'Top 10 Words Most Similar to "{condition_name}"')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

        # Network Graph with improved styling
        plt.figure(figsize=(12, 12))
        G = nx.Graph()
        
        # Add central node
        G.add_node(condition_name, size=3000, color='#ff6b6b')  # Coral red
        
        # Add similar words as nodes and edges
        for word, score in similar_words:
            G.add_node(word, size=2000 * score, color='#4ecdc4')  # Turquoise
            G.add_edge(condition_name, word, weight=score)

        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes with custom styling
        node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                             node_color=node_colors, 
                             alpha=0.7, 
                             edgecolors='white')
        
        # Draw edges with varying thickness and custom styling
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                             alpha=0.5, 
                             edge_color='#95a5a6')
        
        # Add labels with custom styling
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, 
                              font_size=10, 
                              font_weight='bold',
                              font_color='#2c3e50')
        
        plt.title(f'Word Similarity Network for "{condition_name}"', 
                 pad=20, 
                 fontsize=14, 
                 fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Heatmap with improved styling
        plt.figure(figsize=(12, 8))
        similarity_matrix = np.zeros((len(words), len(words)))
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                try:
                    similarity_matrix[i, j] = w2v_model.wv.similarity(word1, word2)
                except KeyError:
                    similarity_matrix[i, j] = 0

        sns.heatmap(similarity_matrix, 
                   xticklabels=words, 
                   yticklabels=words, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='YlOrRd',
                   square=True,
                   cbar_kws={'label': 'Similarity Score'})
        
        plt.title(f'Word Similarity Heatmap for Top 10 Words Similar to "{condition_name}"')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Print similarity scores
        print(f"\nMost similar words for {condition_name}:")
        for word, score in similar_words:
            print(f"{word}: {score:.4f}")

    except KeyError:
        print(f"'{condition_name}' not found in vocabulary")


# Process both conditions

print("\nProcessing Diabetes notes using Scientific Model...")
create_word2vec_and_tsne(diabetes_sample_notes, nlp_sci, "Diabetes")

# print("\nProcessing Diabetes notes using Web Model...")
# create_word2vec_and_tsne(diabetes_sample_notes, nlp_web, "Diabetes")

print("\nAnalysis complete!")
