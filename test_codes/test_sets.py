x = {1:"a", 2:"b", 4:"c"}
print(x)
y = [1,2]
z = x.keys
w = z - y
for i in y:
    del x[i]
print(x)