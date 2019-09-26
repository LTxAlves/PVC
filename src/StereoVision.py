from argparse import ArgumentParser

def Req1():
    while(True):
        print('Choose an image set to work with:')
        print('\t1) Piano')
        print('\t2) Playroom')
        print('\t3) Quit')
        option = input('-> ')
        if option == '1':
            pass
        elif option == '2':
            pass
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