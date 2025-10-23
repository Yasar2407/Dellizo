from transformers import pipeline

def load_classifier(model_name: str):
    return pipeline("text-classification", model=model_name, return_all_scores=True)

def pick_best_label(pipe_output):
    scores = pipe_output[0]
    best = max(scores, key=lambda x: x['score'])
    return best['label'], float(best['score'])
