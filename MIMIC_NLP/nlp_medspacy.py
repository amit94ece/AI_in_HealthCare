import medspacy
from medspacy.ner import TargetRule
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from collections import Counter
import seaborn as sns
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
import ssl
from tqdm.auto import tqdm

# Enable progress bar for pandas
tqdm.pandas()

def extract_and_display_entities(text, model, title):
    # Add rules for target concept extraction
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [
        TargetRule("CVA", "PROBLEM"),
        TargetRule("respiratory secretions", "SYMPTOM"),
        TargetRule("vomiting", "SYMPTOM"),
        TargetRule("aspiration pneumonia", "DIAGNOSIS"),
        TargetRule("Vancomycin", "MEDICATION"),
        TargetRule("Levofloxacin", "MEDICATION"),
        TargetRule("Flagyl", "MEDICATION"),
        TargetRule("diabetes", "CONDITION"),
        TargetRule("type 2 diabetes", "CONDITION"),
        TargetRule("t2dm", "CONDITION"),
        TargetRule("insulin", "MEDICATION"),
        TargetRule("metformin", "MEDICATION"),
        TargetRule("glipizide", "MEDICATION"),
        TargetRule("a1c", "LAB"),
        TargetRule("hemoglobin a1c", "LAB"),
        TargetRule("blood glucose", "LAB"),
        TargetRule("fasting glucose", "LAB"),
        TargetRule("hyperglycemia", "CONDITION"),
        TargetRule("hypoglycemia", "CONDITION"),
        TargetRule("diabetic nephropathy", "CONDITION"),
        TargetRule("diabetic retinopathy", "CONDITION"),
        TargetRule("diabetic neuropathy", "CONDITION")
    ]
    target_matcher.add(target_rules)

    doc = model(text)
    
    print(f"\n{title} Entities:")
    print("-" * 100)

    # Print extracted entities
    print("Extracted Entities:")
    for ent in doc.ents:
        print(f"Text: {ent.text}, Label: {ent.label_}")

    # Visualize entities
    from collections import Counter
    entity_labels = [ent.label_ for ent in doc.ents]
    label_counts = Counter(entity_labels)

    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title('Entity Types Distribution')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('medspacy_entity_distribution.png')
    plt.close()

    print("Entity distribution plot saved as 'medspacy_entity_distribution.png'")

    return [(ent.text, ent.label_) for ent in doc.ents]

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

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    return ''

def preprocess_text(text, nlp_model):
    """
    Preprocess text for Word2Vec
    """
    try:
        # Clean text
        text = clean_text(text)
        
        # Process with spaCy
        doc = nlp_model(text)
        
        # Get tokens that:
        # - are not punctuation
        # - are not space
        # - are not stop words
        # - have length > 2
        tokens = [token.text.lower() for token in doc 
                 if not token.is_punct 
                 and not token.is_space
                 and not token.is_stop
                 and len(token.text) > 2]
        
        return tokens
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return []
    
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

def create_word2vec_and_tsne(notes_df, nlp_model, condition_name):
    """
    Create Word2Vec model and t-SNE visualization
    """
    print(f"\nProcessing {condition_name} notes...")
    
    try:
        # Tokenize texts with progress bar
        print("Tokenizing texts...")
        # First ensure TEXT column exists and is string type
        notes_df['TEXT'] = notes_df['TEXT'].fillna('')
        notes_df['TEXT'] = notes_df['TEXT'].astype(str)
        
        # Now apply tokenization with progress bar
        notes_df['tokens'] = notes_df['TEXT'].apply(
            lambda x: preprocess_text(x, nlp_model)
        )
        
        # Remove empty token lists
        notes_df = notes_df[notes_df['tokens'].map(len) > 0]
        
        if len(notes_df) == 0:
            print("No valid texts found after preprocessing")
            return None
        
        print("Creating Word2Vec model...")
        w2v_model = Word2Vec(sentences=notes_df['tokens'].tolist(), 
                           vector_size=100, 
                           window=5, 
                           min_count=2,
                           workers=4)
        
        # Most similar words
        print(f"\nMost similar words for {condition_name}:")
        try:
            similar_words = w2v_model.wv.most_similar(condition_name.lower(), topn=10)
            for word, score in similar_words:
                print(f"{word}: {score:.4f}")
        except KeyError:
            print(f"'{condition_name}' not found in vocabulary")
        
        # Visualization of similar words
        visualize_similar_words(w2v_model, condition_name)

        # TSNE Visualization
        print("\nCreating t-SNE visualization...")
        tokens = []
        labels = []
        
        # Get most common words
        all_words = [word for token_list in notes_df['tokens'] for word in token_list]
        word_freq = Counter(all_words)
        most_common = [word for word, count in word_freq.most_common(100)]
        
        for word in most_common:
            if word in w2v_model.wv:
                tokens.append(w2v_model.wv[word])
                labels.append(word)
        
        if len(tokens) == 0:
            print("No words found in Word2Vec model")
            return None
        
        tokens_array = np.array(tokens)
        
        tsne_model = TSNE(perplexity=min(40, len(tokens_array)-1), 
                         n_components=2, 
                         init='pca', 
                         n_iter=2500, 
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
        
        plt.title(f"t-SNE visualization for {condition_name}")
        plt.tight_layout()
        plt.show()
        
        return w2v_model
        
    except Exception as e:
        print(f"Error in create_word2vec_and_tsne: {str(e)}")
        return None
    
# Main execution
if __name__ == "__main__":
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')

    # Load medspacy model
    nlp = medspacy.load()

    # Load datasets
    print("Loading data...")
    notes_df = pd.read_csv('./data/NOTEEVENTS.csv.gz', low_memory=False)
    diagnoses_df = pd.read_csv('./data/DIAGNOSES_ICD.csv.gz', low_memory=False)

    # Diabetes Analysis
    print("\nFiltering for diabetes patients...")
    diabetes_patients = diagnoses_df[diagnoses_df['ICD9_CODE'].str.startswith('250', na=False)]
    diabetes_notes = pd.merge(diabetes_patients, notes_df, on=['SUBJECT_ID', 'HADM_ID'])
    diabetes_sample_notes = diabetes_notes[diabetes_notes['CATEGORY'] == 'Discharge summary'].sample(n=500, random_state=42)

    sample_text = diabetes_sample_notes['TEXT'].iloc[0]
    medspacy_entities = extract_and_display_entities(sample_text, nlp, "MedSpaCy Model")

    visualize_entity_network(medspacy_entities, "MedSpaCy Model")

    create_word2vec_and_tsne(diabetes_sample_notes, nlp, "Diabetes")

    print("\nAnalysis complete!")