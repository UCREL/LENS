import re
from functions import *

class Extractor:
  def __init__(self, text):
    self.text = text
    self.doc = lens(self.text)
    self.entities = self.extract_entities()

# Extract entities
  def extract_entities(self):
    entities = {}
    for ent in self.doc.ents:
      if ent.label_ in ["TRT","SYM","MET","CANC_T","SIZE","EMO","PPL","MED","MHD","ORG","ADV_EFF","INV","POB","EGY","DUR","AGE","GENDER","STG","EXP","A/G","RES","DIAG","GPE","NUM"]:
          tag = ent.label_
          entities[ent.start_char] = (len(ent.text), ent.text, tag)
    return collections.OrderedDict(sorted(entities.items()))

# Generate html formatted text 
  def visualize(self, ents):
    html, end_div = f'<div class="entities" style="line-height: 2.3; direction: ltr">', '\n</div>'
    for token, tag in get_token_tags(self.text, ents):
      html += format_entity(token,tag)
    html += end_div
    return html