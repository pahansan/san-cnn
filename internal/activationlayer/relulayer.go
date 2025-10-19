package activationlayer

import "san-cnn/internal/tensor"

type ReLULayer struct {
	size tensor.TensorSize
}

func NewReLULayer(size tensor.TensorSize) ReLULayer {
	return ReLULayer{size: size}
}

func (l *ReLULayer) Forward(X tensor.Tensor) tensor.Tensor {
	output := tensor.NewTensor(l.size)

	for i := 0; i < l.size.Height; i++ {
		for j := 0; j < l.size.Width; j++ {
			for k := 0; k < l.size.Depth; k++ {
				value := X.GetValue(k, i, j)
				if value > 0 {
					output.SetValue(k, i, j, value)
				} else {
					output.SetValue(k, i, j, 0)
				}
			}
		}
	}
	return output
}

func (l *ReLULayer) Backward(dout, X tensor.Tensor) tensor.Tensor {
	dX := tensor.NewTensor(l.size)

	for i := 0; i < l.size.Height; i++ {
		for j := 0; j < l.size.Width; j++ {
			for k := 0; k < l.size.Depth; k++ {
				if X.GetValue(k, i, j) > 0 {
					dX.SetValue(k, i, j, dout.GetValue(k, i, j))
				} else {
					dX.SetValue(k, i, j, 0)
				}
			}
		}
	}
	return dX
}
