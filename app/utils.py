def map_label_to_emotion(hf_label: str) -> str:
    mapping = {
        "joy": "joy",
        "happiness": "joy",
        "love": "love",
        "sadness": "sadness",
        "anger": "anger",
        "fear": "fear",
        "anxiety": "fear",
        "neutral": "neutral",
        "surprise": "surprise",
        "disgust": "disgust",
    }
    return mapping.get(hf_label.lower(), "neutral")
