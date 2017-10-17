require_relative "ext"

class Neuron
  def initialize(inputs_count)
    @weights = Array.new(inputs_count) { rand }.to_v
  end

  def forward(inputs)
    @inputs = inputs
    Math.sigmoid(inputs.dot(@weights))
  end

  def back_propagate(error)
    @delta = error * 0.1 # learning rate
    @weights * @delta
  end

  def update_weights
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
    @neurons.zip(errors).map { |neuron, error|
      neuron.back_propagate(error) }.sum
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
    @layers.inject(inputs) do |inputs, layer|
      layer.forward(inputs)
    end
  end

  def back_propagate(errors)
    @layers.reverse_each do |layer|
      errors = layer.back_propagate(errors)
    end
  end

  def update_weights
    @layers.each &:update_weights
  end
end



