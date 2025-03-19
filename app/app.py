import gradio as gr
from transformers import pipeline

ner_pipe = pipeline("token-classification", model="samrawal/bert-base-uncased_clinical-ner", device=-1)
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# Custom label mapping
LABEL_MAPPING = {
    "B-problem": "Symptoms",
    "I-problem": "Symptoms",
    "B-treatment": "Treatment",
    "I-treatment": "Treatment",
    "B-test": "Diagnosis",
    "I-test": "Diagnosis",
}

def reconstruct_entities(ner_results):
    OUTPUT = {"Symptoms": [], "Treatment": [], "Diagnosis": []}
    current_entity = None
    
    for entry in ner_results:
        word = entry["word"].replace("##", "")  # Remove subword markers
        entity_type = entry["entity"]
        start, end = entry["start"], entry["end"]

        # Map the entity to our custom category
        mapped_category = LABEL_MAPPING.get(entity_type, None)

        if mapped_category:
            if "B-" in entity_type or (current_entity and current_entity["type"] != mapped_category):
                if current_entity:
                    OUTPUT[current_entity["type"]].append(current_entity["word"])
                current_entity = {"word": word, "type": mapped_category, "start": start, "end": end}
            elif "I-" in entity_type and current_entity and current_entity["type"] == mapped_category:
                if entry["word"].startswith("##"):
                    current_entity["word"] += word
                else:
                    current_entity["word"] += " " + word
                current_entity["end"] = end
    
    if current_entity:
        OUTPUT[current_entity["type"]].append(current_entity["word"])
    
    return OUTPUT

def extract_medical_entities(text):
    ner_results = ner_pipe(text)
    return reconstruct_entities(ner_results)

def classify_intent(text):
    intent_labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern"]
    intent_result = zero_shot(text, candidate_labels=intent_labels)

    sentiment_labels = ["Anxious", "Neutral", "Reassured"]
    sentiment_result = zero_shot(text, candidate_labels=sentiment_labels)

    return {
      "Sentiment": sentiment_result["labels"][0],
      "Intent": intent_result["labels"][0]
    }

# Gradio UI
ner_demo = gr.Interface(
    fn=extract_medical_entities,
    inputs=gr.Textbox(lines=5, placeholder="Enter medical text here..."),
    outputs=gr.JSON(),
    title="Medical Entity Extractor",
    description="Enter a medical text to extract Symptoms, Treatment, and Diagnosis-related terms."
)

intent_demo = gr.Interface(
    fn=classify_intent,
    inputs=gr.Textbox(lines=2, placeholder="Enter text to classify its intent & sentiment"),
    outputs=gr.JSON(),
    title="Intent Classifier",
    description="Detects the intent behind the entered text, such as Seeking Reassurance, Reporting Symptoms, or Expressing Concern."
)

demo = gr.TabbedInterface([ner_demo, intent_demo], ["NER Extraction", "Intent Classification"])

demo.launch()
