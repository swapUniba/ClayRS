from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.information_processor.spacy import Spacy

nltkTest=NLTK(True, False, True,False,True, 'english')
engList=["when", "i", "go", "to", "the", "beacheees", "i", "has", "to", "uses", "a", "lot", "of", "sunscrrems"]
eng = "When I go to the beaches I have to uses a lot of sunscreen" \
      " because http://www.google.com/ htp://www.google.com/ I don't want to burn myself like lorenzoo https://www.google.com/"
it="Quando vado al mare devo usare molta crema solare" \
   " perchè non voglio bruciarmi come lorenzo https://ciao.com"
#print("Frase intera: "+eng)
#print(" token nltk:  " + repr(nltkTest.tokenization_operation(eng)))
#nltkTest.stopwords_removal
#print(nltkTest.stemming)

spacyTest=Spacy(False, False, True, False, False, 'en_core_web_lg')
print(spacyTest.process("gone"))
#spacy2=Spacy('it_core_news_lg',True, False, False, False, True)
