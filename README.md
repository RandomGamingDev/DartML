# DartML
A from scratch ML library that uses a custom architecture so that neurons can be easily inserted & uses a linear activation function I made for fun

Basically, instead of implementing a neural algorithm with layers, my idea was to implement it using nodes that instead use a coordinate system, with an x, which allows for practically infinite layers, while making sure that the neurons don't go into loops. I also added an option to evaluate stuff from the neural network per tick, which would allow for continous input, output, and for there to theoretically have some sort of memory. As such it more closely imitates the brain, while being feedforward.

(This library isn't fully tested and might have bugs, for which you can contact me)

- Saving and loading should work, but hasn't been tested
