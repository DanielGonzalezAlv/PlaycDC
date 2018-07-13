import os
for file in os.listdir('./labels'):
    with open('./labels/' + file, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        if len(lines) == 0:
            os.remove('./labels/' + file)
            os.remove('./JPEGImages/' + file[:-4] + '.jpg')
