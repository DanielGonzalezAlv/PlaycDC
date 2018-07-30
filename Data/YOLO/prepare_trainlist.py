import os
folder = 'JPEGImages'
path="./" + folder  
dirList=os.listdir(path)
with open("cardstrain.txt", "w") as f:
    for filename in dirList:
        f.write(os.path.abspath(os.path.join(folder, filename))+'\n') 

