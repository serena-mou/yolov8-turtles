import shutil
import os
FILE_IN = './autosplit_test.txt'
FOLDER_OUT = './classifier/test/'

with open(FILE_IN, 'r') as file:
    data = file.read().splitlines()

    print(len(data))
    print(data[0].split('/'))
    t=0
    p=0
    for d in data:
        cls = d.split('/')[2]
        fn = d.split('/')[-1]
        if cls == 'unpainted':
            IN = d
            OUT = os.path.join(FOLDER_OUT,'unpainted',fn)
            shutil.move(IN,OUT)
        elif cls == 'painted':
            IN = d
            OUT = os.path.join(FOLDER_OUT,'painted',fn)
            shutil.move(IN,OUT)

