require_relative "nn"

# Make rand deterministic, for testing purposes
srand 0

## Create the network

inputs_count = 2
hiddens_count = 20
outputs_count = 1

network = Network.new(
  Layer.new(inputs_count, hiddens_count),
  Layer.new(hiddens_count, outputs_count)
)


## Training

examples = [
  #    _ XOR _  =        _
  [  [ 0,    0 ].to_v, [ 0 ].to_v  ],
  [  [ 0,    1 ].to_v, [ 1 ].to_v  ],
  [  [ 1,    0 ].to_v, [ 1 ].to_v  ],
  [  [ 1,    1 ].to_v, [ 0 ].to_v  ]
]

(1..1000).each do |epoch|

  if epoch % 100 == 0
    puts "----------- epoch #{epoch} -----------"
  end

  examples.each do |inputs, expected_outputs|
    # Forward pass (evaluate / predict)
    actual_outputs = network.forward(inputs)

    # Compute the errors. By how much did we miss?
    errors = actual_outputs - expected_outputs

    if epoch % 100 == 0
      puts " %d XOR %d = %.2f    error = %+.2f" % [
        inputs[0], inputs[1],
        actual_outputs[0],
        errors[0]
      ]
    end

    # Back-propagate the errors
    network.back_propagate errors

    # Update the weights
    network.update_weights
  end

end