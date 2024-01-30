from docquery import document, pipeline
p = pipeline('document-question-answering')
doc = document.load_document("./Insurance_ID_Card.pdf")
for q in ["What is the policy number?", "What is the Vehicle Identification Number?"]:
    print(q, p(question=q, **doc.context))