import spacy
from ..utils import config

spacy_nlp = spacy.load(config.get_("process.spacy_model"))
