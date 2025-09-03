import statistics
data = list(map(int, input("Enter the data (separated by spaces): ").split()))
mean = sum(data) / len(data) 
sd = sorted(data)
n = len(sd)
if n % 2 == 1:
    median = sd[n // 2]
else:
    median = (sd[n // 2 - 1] + sd[n // 2]) / 2
frequency = {}
for num in data:
    frequency[num] = frequency.get(num, 0) + 1
max_freq = max(frequency.values())
mode = [num for num, freq in frequency.items() if freq == max_freq]

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
