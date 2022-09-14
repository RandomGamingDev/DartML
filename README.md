# DartML
An ML algorithm implementation that uses a custom algorithm (might've been invented before I thought up of it idk) I call DartML

Basically, instead of implementing a neural algorithm with layers, my idea was ti implement it using nodes that instead use a coordinate system, with an x, which allows for practically infinite layers, while making sure that the neurons don't go into loops. I also added an option to evaluate stuff from the neural network per tick, which would allow for continous input, output, and for there to theoretically be some sort of memory, which isn't possible in a lot of other neural networks. As such it more closely imitates the brain, while being efficient, fast, and easy to use and understand.

(This library isn't fully tested and might have bugs, for which you can contact me)

- Saving and loading should work, but hasn't been tested
- C bindings aren't fully supported as saving and loading isn't implemented yet, and there's a test function at the bottom named test_test_test lol
