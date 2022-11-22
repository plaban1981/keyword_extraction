from keybert import KeyBERT
import joblib
import os
import json
def preprocess_function(data_path,content_type = None ):
    #with open(data_path,"r") as f:
    #  data = f.read()
    text = data_path.read()
    return text

# 
def predict_function(input_data,model): 
    kwords = model.extract_keywords(input_data,
                                    stop_words='english',
                                    use_mmr=True, 
                                    diversity=0.5,
                                    top_n=10)
    keywords = [ i[0] for i in kwords]
    return keywords
#
def model_load_function(model_file_path):
    model_path = os.path.join(model_file_path,'keybert.p')
    Model = joblib.load(model_path)
    return Model
#
def postprocess_function(predictions,content_type = None ):
           return json.dumps({"response": "Top 10 keywords detected  {}".format(predictions)})
#test the script is working
"""
if __name__ == "__main__":
    data_path = r".\Data\sample.txt"
    input_data = preprocess_function(data_path)
    model_file_path = r".\model_files"
    Model = model_load_function(model_file_path)
    summary = predict_function(input_data,Model)
    output = postprocess_function(summary)
    print(output)
"""