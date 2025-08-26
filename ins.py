a = int(input("How many numbers? "))
arr = []

for i in range(a):
    num = int(input())
    arr.append(num)

b = int(input("Enter one more number to add: "))
arr.append(b)

arr.sort()
print(arr)
