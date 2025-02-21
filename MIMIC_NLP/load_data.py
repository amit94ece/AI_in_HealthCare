# import pandas as pd
# import numpy as np
# import spacy
# import scispacy
# from scispacy.linking import EntityLinker
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from gensim.models import Word2Vec
# from tqdm import tqdm

# print("Loading data...")
# notes_df = pd.read_csv('./data/NOTEEVENTS.csv.gz', low_memory=False)
# diagnoses_df = pd.read_csv('./data/DIAGNOSES_ICD.csv.gz', low_memory=False)
# print("Data loaded successfully")

# print("Filtering for heart failure patients...")
# hf_patients = diagnoses_df[diagnoses_df['ICD9_CODE'].str.startswith('428', na=False)]
# print(f"Found {len(hf_patients)} heart failure patients")

# print("Merging datasets...")
# hf_notes = pd.merge(hf_patients, notes_df, on='HADM_ID')
# print(f"Merged dataset has {len(hf_notes)} rows")

# print("Sampling discharge summaries...")
# sample_notes = hf_notes[hf_notes['CATEGORY'] == 'Discharge summary'].sample(n=500, random_state=42)
# print(f"Sampled {len(sample_notes)} discharge summaries")

# print("Loading Spacy models...")
# nlp = spacy.load("en_core_sci_lg")
# nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
# nlp_biobert = spacy.load("en_biobert_ner_symptom")
# print("Spacy models loaded")

# def extract_entities(text):
#     doc = nlp(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]

# def extract_symptoms(text):
#     doc = nlp_biobert(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]

# print("Extracting entities and symptoms...")
# tqdm.pandas()
# sample_notes['entities'] = sample_notes['TEXT'].progress_apply(extract_entities)
# sample_notes['symptoms'] = sample_notes['TEXT'].progress_apply(extract_symptoms)
# print("Entity and symptom extraction complete")

# print("Counting entities and symptoms...")
# all_entities = [entity for entities in sample_notes['entities'] for entity in entities]
# all_symptoms = [symptom for symptoms in sample_notes['symptoms'] for symptom in symptoms]
# entity_counts = Counter(all_entities)
# symptom_counts = Counter(all_symptoms)

# print("\nTop 20 most common entities:")
# for entity, count in entity_counts.most_common(20):
#     print(f"{entity}: {count}")

# print("\nTop 20 most common symptoms:")
# for symptom, count in symptom_counts.most_common(20):
#     print(f"{symptom}: {count}")

# print("\nCreating Word2Vec model...")
# sample_notes['tokens'] = sample_notes['TEXT'].apply(lambda x: [token.text for token in nlp(x)])
# w2v_model = Word2Vec(sentences=sample_notes['tokens'], vector_size=100, window=5, min_count=1, workers=4)
# print("Word2Vec model created")

# print("\nCalculating word similarities...")
# similar_words = w2v_model.wv.similar_by_word('Sinus rhythm')
# print("Similar words to 'Sinus rhythm':")
# for word, score in similar_words[:10]:
#     print(f"{word}: {score:.4f}")

# def tsne_plot(model):
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []

#     for word in model.wv.index_to_key:
#         tokens.append(model.wv[word])
#         labels.append(word)
    
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
        
#     plt.figure(figsize=(16, 16)) 
#     for i in range(len(x)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()

# print("\nCreating t-SNE plot...")
# tsne_plot(w2v_model)
# print("t-SNE plot created")

# print("Analysis complete!")

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

# def check_spacy_version():
#     """Check if spaCy version is compatible and install if needed"""
#     try:
#         import spacy
#         current_version = spacy.__version__
#         if not current_version.startswith('3.7'):
#             print("Installing compatible spaCy version...")
#             import subprocess
#             subprocess.run(["pip", "install", "-U", "spacy==3.7.4"], check=True)
#             subprocess.run(["pip", "install", "-U", "scispacy==0.5.5"], check=True)
#             subprocess.run([
#                 "pip", "install",
#                 "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz"
#             ], check=True)
#             print("Please restart the script to use the newly installed versions")
#             exit()
#     except Exception as e:
#         print(f"Error checking/installing spaCy: {str(e)}")
#         exit()

# # Check spaCy version before proceeding
# check_spacy_version()

print("Loading data...")
notes_df = pd.read_csv('./data/NOTEEVENTS.csv.gz', low_memory=False)
diagnoses_df = pd.read_csv('./data/DIAGNOSES_ICD.csv.gz', low_memory=False)
print("Data loaded successfully")

print("Filtering for heart failure patients...")
hf_patients = diagnoses_df[diagnoses_df['ICD9_CODE'].str.startswith('428', na=False)]
print(f"Found {len(hf_patients)} heart failure patients")

print("Merging datasets...")
hf_notes = pd.merge(hf_patients, notes_df, on='HADM_ID')
print(f"Merged dataset has {len(hf_notes)} rows")

print("Sampling discharge summaries...")
sample_notes = hf_notes[hf_notes['CATEGORY'] == 'Discharge summary'].sample(n=500, random_state=42)
print(f"Sampled {len(sample_notes)} discharge summaries")

print("Loading Spacy model...")
try:
    nlp = spacy.load("en_core_sci_lg")
    print("Loaded en_core_sci_lg model")
except OSError:
    print("Model not found. Please ensure you've installed the required models.")
    exit()

# Add UMLS entity linker if needed
try:
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
except Exception as e:
    print(f"Note: UMLS linking not available - {str(e)}")

def extract_entities(text):
    """Extract entities from text using spaCy model."""
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return []

print("Extracting entities...")
tqdm.pandas()
sample_notes['entities'] = sample_notes['TEXT'].progress_apply(extract_entities)
print("Entity extraction complete")

print("Counting entities...")
all_entities = [entity for entities in sample_notes['entities'] for entity in entities]
entity_counts = Counter(all_entities)

print("\nTop 20 most common entities:")
for entity, count in entity_counts.most_common(20):
    print(f"{entity}: {count}")

print("\nCreating Word2Vec model...")
def preprocess_text(text):
    try:
        doc = nlp(text)
        return [token.text for token in doc if not token.is_punct and not token.is_space]
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return []

sample_notes['tokens'] = sample_notes['TEXT'].progress_apply(preprocess_text)
w2v_model = Word2Vec(sentences=sample_notes['tokens'], vector_size=100, window=5, min_count=1, workers=4)
print("Word2Vec model created")

print("\nCalculating word similarities...")
try:
    similar_words = w2v_model.wv.most_similar('heart', topn=10)
    print("Similar words to 'heart':")
    for word, score in similar_words:
        print(f"{word}: {score:.4f}")
except KeyError:
    print("Note: 'heart' not found in vocabulary. Try another common medical term.")

def tsne_plot(model):
    """Creates and TSNE model and plots it"""
    try:
        labels = []
        tokens = []

        # Limit to top 100 most frequent words for visualization
        vocab = model.wv.index_to_key[:100]
        
        for word in vocab:
            tokens.append(model.wv[word])
            labels.append(word)
        
        # Convert tokens list to numpy array
        tokens_array = np.array(tokens)
        
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', max_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens_array)

        x = new_values[:, 0]  # Get all x coordinates
        y = new_values[:, 1]  # Get all y coordinates
            
        plt.figure(figsize=(16, 16)) 
        plt.scatter(x, y)
        
        # Add labels to points
        for i, label in enumerate(labels):
            plt.annotate(label,
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        
        plt.title("t-SNE visualization of word embeddings")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.show()
        
    except Exception as e:
        print(f"Error creating t-SNE plot: {str(e)}")
        raise  # This will show the full error traceback

print("\nCreating t-SNE plot...")
tsne_plot(w2v_model)
print("t-SNE plot created")


print("Analysis complete!")
