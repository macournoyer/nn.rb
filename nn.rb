require_relative "ext"

class Neuron
  def initialize(inputs_count)
    # Initialize the weights at random
    @weights = Array.new(inputs_count) { rand }.to_v
  end

  def forward(inputs)
    @inputs = inputs
    
    # (0...inputs.size).map { |i| inputs[i] * @weights[i] }.sum
    # inputs.zip(@weights).map { |input, weight| input * weight }.sum

    # Calculate the output. The prediction of that neuron
    @output = Math.sigmoid(inputs.dot(@weights))
  end

  def back_propagate(error)
    # We multiply the error by the derivative (slope).
    #
    # The derivative increases as we approach 0.5. Where a neuron is less certain.
    # We want to avoid neurons being less certain. Because a bunch of neurons that are not certain
    # about anything is not worth much. So we penalize this.
    #
    # Predictions around 0.5 are corrected the most.
    # Predictions close to 0 or 1 are corrected less.
    @delta = error * Math.sigmoid_derivative_from_output(@output)

    # Return the error for the previous layer by filtering our delta backward,
    # through the weights.
    @weights * @delta
  end

  def update_weights
    # Update the weights by applying the delta to the input.
    # Since an error of +0.1 means we need to subtract 0.1, we use a subtraction to update the
    # weights.
    @weights -= @inputs * @delta
  end
end

class Layer
  def initialize(inputs_count, outputs_count)
    @neurons = Array.new(outputs_count) { Neuron.new(inputs_count) }
  end

  def forward(inputs)
    @neurons.map { |neuron| neuron.forward(inputs) }.to_v
  end

  def back_propagate(errors)
    @neurons.zip(errors).map { |neuron, error| neuron.back_propagate(error) }.sum
  end

  def update_weights
    @neurons.each &:update_weights
  end
end

class Network
  def initialize(*layers)
    @layers = layers
  end

  def forward(inputs)
    outputs = nil

    @layers.each do |layer|
      outputs = layer.forward(inputs)
      # Output will be the input of the next layer
      inputs = outputs
    end

    outputs
  end

  # Propagate the errors at each layer from the back (last layer).
  def back_propagate(errors)
    @layers.reverse_each do |layer|
      errors = layer.back_propagate(errors)
    end
  end

  def update_weights
    @layers.each &:update_weights
  end
end
