import transformers
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load the judgments and summaries from the CSV file
def load_judgments_and_summaries(csv_file):
    judgments = []
    summaries = []
    with open(csv_file, "r", encoding="utf-8") as file:
        for line in file:
            data = line.strip().split(",", 1)  # Limit split to 1 to avoid further splitting
            if len(data) == 2:  # Check if there are exactly two values
                judgment, summary = data
                judgments.append(judgment)
                summaries.append(summary)
    return judgments, summaries

# Function to generate summary for a given text
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load judgments and summaries from CSV
judgments, summaries = load_judgments_and_summaries("summary.csv")

# Example usage
input_text = "In 1990, Stephen Kimble obtained a patent for a Spider - Man toy that was set to expire in May 2010. Kimble claimed that he discussed the idea with the president of Marvel Enterprises Inc., and that he would be compensated for use of his ideas. Although no agreement was reached, Marvel produced a model that was similar to Kimble's design. In 1997, Kimble sued under patent protection, and the parties settled in 2001, with Marvel agreeing to purchase the patent and pay royalties to the petitioner without an expiration date. The case was subsequently dismissed. In 2006, Marvel entered a licensing agreement with Hasbro Inc. that gave it the right to produce the toy. Disagreements arose between Kimble and Marvel concerning the royalty payments, and Kimble claimed that the original patent would be infringed if royalties were not paid. Kimble sued Marvel in Arizona state court, and the case was then removed to the federal district court. The magistrate judge determined that settlement agreements was a hybrid  agreement, in which patent and non - patent rights were inseparable, and that the Supreme Court decision in Brulotte v. Thys Co. applied. In that case, the Court ruled that, when patents are sold in return for a royalty payment, the purchaser was not obligated to continue these payments beyond the expiration date of the patents because doing so would over - compensate the seller of the patent and improperly extend the patent monopoly beyond their intended time limit. On recommendation of the magistrate, the district court granted summary judgment in favor of Marvel and ruled that the settlement agreement transferred patent rights, but that it was unclear if non - patent rights were transferred. Kimble sued and argued that the settlement agreement transferred both patent and non - patent rights and that, while royalty payments ended in the patent, they did not end for the toy itself. The you. S. Court of Appeals for the Ninth Circuit affirmed the decision of the district,"
generated_summary = generate_summary(input_text)
print("Generated Summary:", generated_summary)
