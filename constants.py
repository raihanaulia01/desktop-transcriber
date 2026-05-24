VALID_MODELS = ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large-v3","large","distil-large-v2","distil-medium.en","distil-small.en","distil-large-v3","distil-large-v3.5","large-v3-turbo","turbo"]
VALID_LANGUAGE_CODES = ["af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh","yue",]
VALID_DEVICES = ["cuda", "cpu", "auto"]
HALLUCINATION_DICT = {
  "thank you for your viewing":2, 
  "thank you for watching until the end":2, 
  "thank you so much for watching until the end":2,
  "thank you for watching":2, 
  "thanks for watching":2, 
  "see you next time":2,
  "translated by releska":None,
  "please subscribe to my channel and give it a high rating":None
}