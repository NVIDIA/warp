import warp as wp

wp.init()

# at least for GPU arrays this should throw a runtime exception stating that you can't deference 
# an individual array item, for CPU we could actually return the item directly
a = wp.zeros(100, dtype=float)
print(a[0])

# should return the first row of the array
s = wp.zeros((100,100), dtype=float)
print(s[0])