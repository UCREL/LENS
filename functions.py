import collections
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import spacy
from spacy import displacy

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'lens_model')
lens = spacy.load(model_path)

BG_COLOR = {"TRT": "#997a8d", "SYM": "#db7093", "MET": "#ffb6c1", "CANC_T": "#e68fac", "SIZE": "#fc89ac", "EMO": "#f78fa7", "PPL": "#dea5a4", "MED": "#e18e96",
                          "MHD": "#ff91af", "ORG": "#ff91a4", "ADV_EFF": "#f19cbb", "INV": "#efbbcc", "POB": "#F9CBCB", "EGY": "#e8ccd7", "DUR": "#f7bfbe", "AGE": "#c4c3d0",
                          "GENDER": "#ffc1cc", "STG": "#aa98a9", "EXP": "#d98695", "A/G": "#dea5a4", "RES": "#cc8899", "DIAG": "#fc6c85", "GPE": "#c9c0bb", "NUM": "#e5ccc9"}

# format a typical entity for display 
def format_entity(token, tag):
  if tag:
    start_mark = f'<mark class="entity" style="background: {BG_COLOR[tag]}; padding: 0.4em 0.4em; margin: 0 0.25em; line-height: 0.8; border-radius: 0.25em;">'
    end_mark = '\n</mark>'
    start_span = '<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">'
    end_span = '\n</span>'
    return f"\n{start_mark}{token}{start_span}{tag}{end_span}{end_mark}"
  return f"{token}"

# extract all known entities in a lists
def get_token_tags(txtstr, entities):
  begin, tokens_tags = 0, []
  for start, vals in entities.items():
    length, ent, tag = vals
    if begin <= start:
      tokens_tags.append((txtstr[begin:start], None))
      tokens_tags.append((txtstr[start:start+length], tag))
      begin = start+length
  tokens_tags.append((txtstr[begin:], None)) #add the last untagged chunk
  return tokens_tags

# Expand list with inflections and lemmas
def get_inflections(names_list):
    gf_names_inflected = []
    for w in names_list:
      w = w.strip()
      gf_names_inflected.append(w)
      gf_names_inflected.extend(list(getInflection(w.strip(), tag='NNS', inflect_oov=False)))
      gf_names_inflected.extend(list(getLemma(w.strip(), 'NOUN', lemmatize_oov=False)))
    return list(set(gf_names_inflected))

combine = lambda x, y: (x[0], x[1]+' '+y[1], x[2])

def combine_multi_tokens(a_list):
  new_list = [a_list.pop()]
  while a_list:
    last = a_list.pop()
    if new_list[-1][0] - last[0] == 1:
      new_list.append(combine(last, new_list.pop()))
    else:
      new_list.append(last)
  return sorted(new_list)

# merge two entities
def merge_entities(first_ents, second_ents):
  return collections.OrderedDict(
      sorted({** second_ents, **first_ents}.items()))

# show text unformated text
def show_plain_text(txtstr):
  'Original text:'
  start_mark = f'<mark class="entity" style="background: #FFFFFF; line-height: 2; border-radius: 0.35em;">'
  end_mark = '\n</mark>'
  return f"{start_mark}{txtstr}{end_mark}"

#=============================  Building the Spacy model  ===================================

def display_entities(text, tag_list=None):
    doc = lens(text)
    tag_list = tag_list
    colors = {"TRT": "#997a8d", "SYM": "#db7093", "MET": "#ffb6c1", "CANC_T": "#e68fac", "SIZE": "#fc89ac", "EMO": "#f78fa7", "PPL": "#dea5a4", "MED": "#e18e96",
                          "MHD": "#ff91af", "ORG": "#ff91a4", "ADV_EFF": "#f19cbb", "INV": "#efbbcc", "POB": "#F9CBCB", "EGY": "#e8ccd7", "DUR": "#f7bfbe", "AGE": "#c4c3d0",
                          "GENDER": "#ffc1cc", "STG": "#aa98a9", "EXP": "#d98695", "A/G": "#dea5a4", "RES": "#cc8899", "DIAG": "#fc6c85", "GPE": "#c9c0bb", "NUM": "#e5ccc9"}
    if tag_list:
        options = {"ents": tag_list, "colors": colors}
        displacy.render(doc, style='ent',options=options)
    else:
        options = {"ents": ["TRT","SYM","MET","CANC_T","SIZE","EMO","PPL","MED","MHD","ORG","ADV_EFF","INV","POB","EGY","DUR","AGE","GENDER",
                              "STG","EXP","A/G","RES","DIAG","GPE","NUM"], "colors":colors}
        displacy.render(doc, style='ent',options=options)

def get_entities(text, tag_list=None):
    doc = lens(text)
    entities = []
    tag_list = tag_list

    for ent in doc.ents:
        if tag_list:
            for tag in tag_list:
                if ent.label_ == tag.upper():
                    entities.append({'entity': ent.text, 'label': ent.label_, 'start_index': ent.start_char, 'end_index': ent.end_char})
        else:
            entities.append({'entity': ent.text, 'label': ent.label_, 'start_index': ent.start_char, 'end_index': ent.end_char})

    return entities

def lens2medcat(text):
    mapper = {"TRT": {'type_desc': ['Therapeutic or Preventive Procedure'], 'type_id': ['T061']}, "SYM": {'type_desc': ['Sign or Symptom'], 'type_id': ['T184']}, "CANC_T": {'type_desc': ['Disease or Syndrome'], 'type_id': ['T047']}, "STG": {'type_desc': ['Clinical Attribute'], 'type_id': ['T201']},
                              "SIZE": {'type_desc': ['Quantitative Concept'], 'type_id': ['T081']}, "PPL": {'type_desc': ['Professional or Occupational Group'], 'type_id': ['T097']}, "MED": {'type_desc': ['Pharmacologic Substance'], 'type_id': ['T121']}, "MHD": {'type_desc': ['Mental or Behavioral Dysfunction'], 'type_id': ['T048']},
                              "ORG": {'type_desc': ['Health Care Related Organization'], 'type_id': ['T093']}, "ADV_EFF": {'type_desc': ['Sign or Symptom'], 'type_id': ['T184']}, "INV": {'type_desc': ['Diagnostic Procedure', 'Finding', 'Laboratory or Test Result'], 'type_id': ['T060', 'T033', 'T059']}, "POB": {'type_desc': ['Body Part, Organ, or Organ Component',	'Body Location or Region'], 'type_id': ['T023', 'T029']},
                              "EGY": {'type_desc': ['Cell or Molecular Dysfunction', 'Genetic Function'], 'type_id': ['T049', 'T045']}, "DUR": {'type_desc': ['Temporal Concept'], 'type_id': ['T079']}, "AGE": {'type_desc': ['Temporal Concept'], 'type_id': ['T079']}, "GENDER": {'type_desc': ['Organism Attribute'], 'type_id': ['T032']}, "A/G": {'type_desc': ['Temporal Concept', 'Organism Attribute'], 'type_id': ['T079', 'T032']}, "RES": {'type_desc': ['Finding'], 'type_id': ['T033']},
                              "DIAG": {'type_desc': ['Disease or Syndrome'], 'type_id': ['T047']}, "GPE": {'type_desc': ['Geographic Area'], 'type_id': ['T083']}, "NUM": {'type_desc': ['Quantitative Concept'], 'type_id': ['T081']},
                              "EMO": {'type_desc': ['Mental Process'], 'type_id': ['T041']}, "MET": {'type_desc': [], 'type_id': []},  "EXP": {'type_desc': [], 'type_id': []} }

    lens2medcat_list = []
    doc = lens(text)

    for ent in doc.ents:
        medcat_type_desc = mapper[ent.label_]['type_desc']
        medcat_type_id = mapper[ent.label_]['type_id']
        lens2medcat_list.append({'entity': ent.text, 'label': ent.label_, 'start_index': ent.start_char, 'end_index': ent.end_char, 'medcat_type_desc': medcat_type_desc, 'medcat_type_id': medcat_type_id})

    return lens2medcat_list

def lens2snomedct(text):
    mapper = {"TRT": {"CUI": "71388002", "Type_ID": "28321150",  "Type_Description": "procedure", "Full_Name": "Procedure"}, "SYM": {"CUI": "72670004", "Type_ID": "67667581", "Type_Description": "finding", "Full_Name": "Sign"},
                              "INV": {"CUI": "386053000", "Type_ID": "363679005", "Type_Description": "procedure", "Full_Name": "Investigations"},  "CANC_T": {"CUI": "363346000", "Type_ID": "9090192",
                              "Type_Description": "disorder", "Full_Name": "Malignant neoplastic disease (disorder)" }, "STG": {"CUI": "254292007", "Type_ID": "30703196", "Type_Description": "tumor staging	",
                              "Full_Name": "Tumor staging"}, "POB": {"CUI": "38866009", "Type_ID": "37552161", "Type_Description": "body structure", "Full_Name": "Body part"},  "ADV_EFF": {"CUI": "281647001",
                              "Type_ID": "9090192", "Type_Description": "disorder", "Full_Name": "Adverse reactions"}, "MED": {"CUI": "410942007", "Type_ID": "91187746", "Type_Description": "substance",
                              "Full_Name": "Drug or medicament"},  "AGE": {"CUI": "424144002", "Type_ID": "2680757", "Type_Description": "observable entity", "Full_Name": "Age"}, "GENDER": {"CUI": "365873007",
                              "Type_ID": "67667581", "Type_Description": "finding", "Full_Name": "Gender finding"},   "A/G": {"CUI": ['424144002', '365873007'], "Type_ID": ['2680757', '67667581'], "Type_Description": ['observable entity', 'finding'], "Full_Name": ['Age', 'Gender finding']},  "DIAG": {"CUI": "439401001", "Type_ID": "2680757", "Type_Description": "observable entity",
                              "Full_Name": "Diagnosis"}, "SIZE": {"CUI": "246115007", "Type_ID": "43039974", "Type_Description": "attribute", "Full_Name": "Size"}, "NUM": {"CUI": "246205007",
                              "Type_ID": "43039974", "Type_Description": "attribute", "Full_Name": "Quantity"}, "MHD": {"CUI": "74732009",  "Type_ID": "9090192", "Type_Description": "disorder", "Full_Name": "Mental disorder"},
                              "PPL": { "CUI": "14679004",  "Type_ID": "16939031",  "Type_Description": "occupation", "Full_Name": "Occupation"},  "EMO": {"CUI": "285854004", "Type_ID": "2680757", "Type_Description": "observable entity", "Full_Name": "Emotions"},
                              "DUR": {"CUI": "282032007", "Type_ID": "7882689", "Type_Description": "qualifier value", "Full_Name": "Periods of life"},  "EGY": {"CUI": "134198009", "Type_ID": "43039974", "Type_Description": "attribute", "Full_Name": "Etiology"}, "GPE": {
                              "CUI": "758638001", "Type_ID": "7882689", "Type_Description": "qualifier value", "Full_Name": "Geographical location"}, "EXP": {"CUI": [],"Type_ID": [], "Type_Description": [],
                              "Full_Name": []}, "MET": {"CUI": [], "Type_ID": [], "Type_Description": [], "Full_Name": []},  "ORG": {"CUI": "568291000005106",  "Type_ID": "75168589",  "Type_Description": "environment",
                              "Full_Name": "Hospital unit"},  "RES": {"CUI": "102468005", "Type_ID": "2680757", "Type_Description": "observable entity", "Full_Name": "Therapeutic response"}}

    lens2snomed_list = []
    doc = lens(text)

    for ent in doc.ents:
        snomed_type_desc = mapper[ent.label_]['Type_Description']
        snomed_type_id = mapper[ent.label_]['Type_ID']
        snomed_cui = mapper[ent.label_]['CUI']
        snomed_name = mapper[ent.label_]['Full_Name']
        lens2snomed_list.append({'entity': ent.text, 'label': ent.label_, 'start_index': ent.start_char, 'end_index': ent.end_char,
                                                          'snomed_ct_cui': snomed_cui, 'snomed_ct_name': snomed_name,
                                                          'snomed_ct_type_desc': snomed_type_desc, 'snomed_ct_type_id': snomed_type_id})

    return lens2snomed_list
# -------------------------------------------------------------------------------

EXAMPLES_DIR = os.path.join(current_dir, 'resources', 'example_texts')

example_files = sorted([f for f in os.listdir(EXAMPLES_DIR) if f.endswith('.txt')])