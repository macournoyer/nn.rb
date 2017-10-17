require "matrix"

class Array
  def to_v
    Vector.elements self
  end

  # Part of Ruby 2.4
  def sum
    inject(:+)
  end
end

module Math
  def self.sigmoid(t)
    1.0 / (1.0 + Math.exp(-t))
  end
end