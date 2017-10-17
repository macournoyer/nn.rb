# Make rand deterministic, for testing purposes
srand 0

################ 1. Create the network ################
require_relative "nn"

network = Network.new(
  Layer.new(20, 10),
  Layer.new(10, 4)
)

################ 2. Train the network ################
require_relative "data"

examples = [
  #                            OUTPUTS
  #        INPUTS             A  B  C  D
  [  LETTER_TO_PIXELS[:A],  [ 1, 0, 0, 0 ].to_v  ],
  [  LETTER_TO_PIXELS[:B],  [ 0, 1, 0, 0 ].to_v  ],
  [  LETTER_TO_PIXELS[:C],  [ 0, 0, 1, 0 ].to_v  ],
  [  LETTER_TO_PIXELS[:D],  [ 0, 0, 0, 1 ].to_v  ],
]

1000.times do
  examples.each do |inputs, expected_outputs|
    actual_outputs = network.forward(inputs)

    errors = actual_outputs - expected_outputs

    network.back_propagate errors
    network.update_weights
  end
end

################ 3. Test the network ################

inputs = [
  1, 1, 1, 0,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 1, 1, 1
].to_v

# Draw the inputs
puts
inputs.each_slice(4) do |line|
  puts line.map { |pixel| pixel == 1 ? ' *' : '  ' }.join
end
puts

outputs = network.forward(inputs)

# Print the output (prediction) for each letter
%w( A B C D ).zip(outputs).each do |letter, output|
  puts "%s: %4.1f%%" % [letter, output * 100]
end
