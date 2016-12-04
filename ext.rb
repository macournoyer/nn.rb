require "matrix"

class Array
  def to_v
    Vector.elements self
  end

  def sum
    inject(:+)
  end
end

module Math
  def self.sigmoid(t)
    1.0 / (1.0 + Math.exp(-t))
  end

  def self.sigmoid_derivative_from_output(sigmoid_t)
    sigmoid_t * (1 - sigmoid_t)
  end
end