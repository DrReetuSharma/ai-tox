import gradio as gr
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import shutil
import uuid

# Load model and tokenizer
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=12)
model.eval()

toxicity_labels = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-γ", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def predict_toxicity(file):
    unique_id = uuid.uuid4().hex
    filename = f"toxicity_results_{unique_id}.csv"

    filepath = shutil.copy(file.name, os.path.join(UPLOAD_DIR, os.path.basename(file.name)))
    df = pd.read_csv(filepath)
    results = []

    for _, row in df.iterrows():
        smiles_id = row[0]
        smiles = row[1]

        inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().tolist()

        result_row = {"ID": smiles_id, "SMILES": smiles}
        result_row.update({tox: round(p, 3) for tox, p in zip(toxicity_labels, probs)})
        results.append(result_row)

    result_df = pd.DataFrame(results)
    output_path = os.path.join(RESULT_DIR, filename)
    result_df.to_csv(output_path, index=False)

    return result_df, output_path

# Interface with downloadable link
def app_with_download(file):
    df, download_path = predict_toxicity(file)
    return df, download_path

iface = gr.Interface(
    fn=app_with_download,
    inputs=gr.File(label="Upload input_smiles.csv"),
    outputs=[
        gr.Dataframe(label="Toxicity Predictions"),
        gr.File(label="Download CSV")
    ],
    title="AI-Tox: AI-powered Toxicity Classifier",
    description="Upload a CSV with columns: id,smiles. Get predictions for 12 toxicity endpoints. Results downloadable. Contact: sharmar@aspire10x.com"
)

iface.launch(share=True)
