import numpy

with numpy.load('gesture_commands/data/0aa1389eaad149708fb3f1894f697b0b.npz') as data:
    print(data.files[1])  # This will print the names of the arrays stored in the .npz file
    for key in data.files:
        print(f"{key}: {data[key]}")