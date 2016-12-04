require_relative "nn"

# 4x5 "images" of letters

letters = {
  A: [
    0, 1, 1, 0,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 1, 1, 1,
    1, 0, 0, 1
  ].to_v,

  B: [
    1, 1, 1, 0,
    1, 0, 0, 1,
    1, 1, 1, 0,
    1, 0, 0, 1,
    1, 1, 1, 0
  ].to_v,

  C: [
    1, 1, 1, 1,
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 1, 1, 1
  ].to_v,

  D: [
    1, 1, 1, 0,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 1, 1, 0
  ].to_v,
}

# Make rand deterministic, for testing purposes
srand 0


## Create the network

inputs_count = 20 # Image size: 4x5
hiddens_count = 10
outputs_count = 4

network = Network.new(
  Layer.new(inputs_count, hiddens_count),
  Layer.new(hiddens_count, outputs_count)
)


## Training

examples = [
  #                   outputs
  #    inputs        A  B  C  D
  [  letters[:A],  [ 1, 0, 0, 0 ].to_v  ],
  [  letters[:B],  [ 0, 1, 0, 0 ].to_v  ],
  [  letters[:C],  [ 0, 0, 1, 0 ].to_v  ],
  [  letters[:D],  [ 0, 0, 0, 1 ].to_v  ],
]

(1..1000).each do |epoch|

  examples.each do |inputs, expected_outputs|
    # Forward pass (evaluate / predict)
    actual_outputs = network.forward(inputs)

    # Compute the errors. By how much did we miss?
    errors = actual_outputs - expected_outputs

    # Back-propagate the errors
    network.back_propagate errors

    # Update the weights
    network.update_weights
  end

end


## Evaluating the network

inputs = [
  1, 1, 1, 0,
  1, 0, 0, 0,
  1, 0, 0, 0,
  1, 0, 0, 0,
  1, 1, 1, 1
].to_v
outputs = network.forward(inputs)

# Draw the inputs
puts
inputs.each_slice(4) do |line|
  puts line.map { |pixel| pixel == 1 ? ' *' : '  ' }.join
end
puts

# Print the output (prediction) for each letter
%w( A B C D ).zip(outputs).each do |letter, output|
  puts "%s: %4.1f%%" % [letter, output * 100]
end
