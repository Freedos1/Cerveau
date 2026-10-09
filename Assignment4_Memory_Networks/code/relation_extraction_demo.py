"""Part B demo: pull (subject, relation, object) triplets from a question and its
answer choices with the pre-trained REBEL model (Huguet Cabot & Navigli, 2021).

Requires:  pip install transformers torch sentencepiece
The model (Babelscape/rebel-large) is downloaded on first use (~1.6 GB).
"""


def parse_rebel(text):
    """REBEL emits '<triplet> subject <subj> object <obj> relation ...'."""
    triplets, subj, obj, rel, state = [], "", "", "", None
    text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for tok in text.split():
        if tok == "<triplet>":
            if rel:
                triplets.append((subj.strip(), rel.strip(), obj.strip()))
            subj, obj, rel, state = "", "", "", "subj"
        elif tok == "<subj>":
            if rel:                     # same subject, new object/relation
                triplets.append((subj.strip(), rel.strip(), obj.strip()))
            obj, rel, state = "", "", "obj"
        elif tok == "<obj>":
            state = "rel"
        elif state == "subj":
            subj += " " + tok
        elif state == "obj":
            obj += " " + tok
        elif state == "rel":
            rel += " " + tok
    if subj and rel and obj:
        triplets.append((subj.strip(), rel.strip(), obj.strip()))
    return triplets


def to_statement(question, choice):
    """Turn a 'why' question + choice into one declarative hypothesis."""
    body = question.rstrip("?").split(" ", 1)[1]           # "did the engine stop"
    body = body.replace("did ", "").replace(" stop", " stopped")
    return f"{body[0].upper()}{body[1:]} because {choice[0].lower()}{choice[1:]}"


if __name__ == "__main__":
    from transformers import pipeline

    extractor = pipeline("text2text-generation", model="Babelscape/rebel-large",
                         tokenizer="Babelscape/rebel-large")
    question = "Why did the engine stop?"
    choices = ["The fuel tank was empty.", "The driver turned on the radio.",
               "The tires were new."]
    texts = [question] + choices + [to_statement(question, c) for c in choices]
    for t in texts:
        out = extractor(t, return_tensors=True, return_text=False)
        ids = out[0]["generated_token_ids"]
        decoded = extractor.tokenizer.batch_decode([ids])[0]
        print(f"{t!r:70} -> {parse_rebel(decoded)}")
