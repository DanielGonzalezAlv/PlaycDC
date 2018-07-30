import os
folder = 'cards_data/JPEGImages'
path="./" + folder  
dirList=os.listdir(path)
with open("cards_data/cardstrain.txt", "w") as f:
    # create train images
    for filename in dirList[:-2000]:
        f.write(os.path.abspath(os.path.join(folder, filename))+'\n') 
with open("cards_data/cardsval.txt", "w") as f:
    # create test images
    for filename in dirList[-2000:]:
        f.write(os.path.abspath(os.path.join(folder, filename))+'\n') 

