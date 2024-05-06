from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import inflect
from graphviz import Digraph

tokenizer = AutoTokenizer.from_pretrained("./NL2ERD")
model = AutoModelForSeq2SeqLM.from_pretrained("./NL2ERD")
nltk.download('punkt')

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': object_.strip(), 'type': relation.strip(),'tail': subject.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': object_.strip(), 'type': relation.strip(),'tail': subject.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': object_.strip(), 'type': relation.strip(),'tail': subject.strip()})
    return triplets

def process_text(text):
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3,
    }
    # Tokenizer text
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
    # Generate
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        **gen_kwargs,
    )
    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    return decoded_preds

def sentence_segmentation(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    decoded_preds = []
    for sentence in sentences:
        decoded_preds.extend(process_text(sentence))
    return decoded_preds

def process_relation(paragraph):
    p = inflect.engine()
    ban_word = ['attributes','attribute','relations','relations','relationship','many-to-many','one-to-may','many-to-many relationships','one-to-many relationships','many-to-many relationship','one-to-many relationship']
    all_relation = []
    final_relation = []
    decoded_preds = sentence_segmentation(paragraph)
    for sentence in decoded_preds:
        sentence = sentence.replace("   "," <subj> ").replace("  "," <obj> ").replace("<s>","").replace("</s>","").replace("<pad>","")
        list_relation = extract_triplets(sentence)
        all_relation.extend(list_relation)
    for value in all_relation:
        status = True
        if value['head'] in ban_word or value['tail'] in ban_word:
            status = False
        if value['head'] == value['tail']:
            status = False
        for item in final_relation:
            if value['head'] == item['head'] and value['tail'] == item['tail'] and value['type'] == item['type']:
                status = False
        if status:
            # Convert 'head' to lowercase and singular form before appending
            singular_head = p.singular_noun(value['head'].lower())
            value['head'] = singular_head if singular_head else value['head'].lower()
            if value not in final_relation:
                final_relation.append(value)
    return final_relation

def generate_erd(final_relation):
    # Create a new Digraph
    dot = Digraph()

    # Extract unique entities
    entities = set()
    for item in final_relation:
        entities.add(item['head'])
        entities.add(item['tail'])

    # Add entities and attributes to the graph
    for entity in entities:
        dot.node(entity)

    for item in final_relation:
        dot.edge(item['head'], item['tail'])

    # Set Graph attributes
    dot.attr(rankdir='LR')

    output_path = './static/picture/erd'
    dot.render(output_path, format='png', cleanup=True)