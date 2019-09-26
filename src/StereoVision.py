from argparse import ArgumentParser
import numpy as np

def ReadMiddleburyTxt(path):
    f = open(path, 'r')
    text = f.readlines()
    f.close()
    newtext = []
    for line in text:
        newtext.append(line.replace('[', ' ').replace(';', '').replace(']', ' ').replace('=', ' '))

    all_words = list(map(lambda l: l.split(), newtext))
    intrinsics0 = np.array(all_words[0][1:], dtype='float').reshape((3, 3))
    intrinsics1 = np.array(all_words[1][1:], dtype='float').reshape((3, 3))
    doffs = float(all_words[2][1])
    baseline = float(all_words[3][1])
    ndisp = int(all_words[6][1])

    return intrinsics0, intrinsics1, doffs, baseline, ndisp

def Req1():
    while(True):
        print('Choose an image set to work with:')
        print('\t1) Piano')
        print('\t2) Playroom')
        print('\t3) Quit')
        option = input('-> ')
        if option == '1':    
            intrinsics0, intrinsics1, doffs, baseline, ndisp = ReadMiddleburyTxt('../data/Middlebury/Piano-perfect/calib.txt')
            print('intrinsics cam0 = \n', intrinsics0)
            print('intrinsics cam1 = \n', intrinsics1)
            print('doffs =', doffs)
            print('baseline =', baseline)
            print('ndisp =', ndisp)
        elif option == '2':
            intrinsics0, intrinsics1, doffs, baseline, ndisp = ReadMiddleburyTxt('../data/Middlebury/Playroom-perfect/calib.txt')
            print('intrinsics cam0 = \n', intrinsics0)
            print('intrinsics cam1 = \n', intrinsics1)
            print('doffs =', doffs)
            print('baseline =', baseline)
            print('ndisp =', ndisp)
        elif option == '3':
            quit()
        else:
            print('Invalid choice!')

def Req2():
    pass

def Req3():
    pass

def Req4():
    pass

# main()
parser = ArgumentParser(description='Choice of requirement')
parser.add_argument('--requirement', '-r', action='store', dest='req', default=1, help='Choose a requirement to see in action (1, 2, 3 or 4)')

arguments = parser.parse_args()

if arguments.req == '1':
    Req1()

elif arguments.req == '2':
    Req2()

elif arguments.req == '3':
    Req3()

elif arguments.req == '4':
    Req4()

else:
    print('Error: invalid requirement! Value must be 1, 2, 3, or 4')