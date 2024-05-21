from langdetect import detect
messages = ["i can not find anything"]



def listtostring(s):  
    str1 = " " 
    return (str1.join(s))

messages = listtostring(messages)
det = detect(messages)
print(det)
#det = detect("War doesn't show who's right, just who's left.")