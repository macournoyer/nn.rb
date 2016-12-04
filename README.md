# nn.rb

A toy Neural Network, from scratch, in Ruby. For learning purposes.

The Neural Network framework is in `nn.rb`.

## XOR

See `xor.rb` for the canonical XOR example:

```
$ ruby xor.rb
----------- epoch 100 -----------
 0 XOR 0 = 0.50    error = +0.50
 0 XOR 1 = 0.38    error = -0.62
 1 XOR 0 = 0.54    error = -0.46
 1 XOR 1 = 0.78    error = +0.78
...
----------- epoch 1000 -----------
 0 XOR 0 = 0.09    error = +0.09
 0 XOR 1 = 0.91    error = -0.09
 1 XOR 0 = 0.91    error = -0.09
 1 XOR 1 = 0.08    error = +0.08
```

## Predicting letters from images

See `letters.rb` for a more interesting example, predicting letters from 4x5 "images" of letters:

```
$ ruby letters.rb

 * * *
 *
 *
 *
 * * * *

A:  2.8%
B:  1.3%
C: 94.2%
D:  3.9%

$ ruby letters.rb

 * * *
 *     *
 *     *
 *
 * * * *

A:  2.1%
B:  0.9%
C: 34.1%
D: 73.4%

$ ruby letters.rb

 * * *
 *     *
 *     *
 *     *
 * * * *

A:  2.9%
B:  1.0%
C:  7.1%
D: 92.9%
```

