from keybert import KeyBERT
import joblib
model = KeyBERT('distilbert-base-nli-mean-tokens')
joblib.dump(model, r'.\model_files\keybert.p')
