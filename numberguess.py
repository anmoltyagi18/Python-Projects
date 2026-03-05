import random
num=random.randint(1,8)
att=0
while True:
    att+=1
    print("Enter the guess number")
    a=int(input())
    if num>a:
        print("too LOW guess")
    elif a>num:
        print("too HIGH guess")
    else:
        print("Right guess hurrayyyy!")
        break
    
