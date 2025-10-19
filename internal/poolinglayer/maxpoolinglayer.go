package poolinglayer

import "san-cnn/internal/tensor"

type MaxPoolingLayer struct {
	scale      int
	inputSize  tensor.TensorSize
	OutputSize tensor.TensorSize
	mask       tensor.Tensor
}

func NewMaxPoolingLayer(size tensor.TensorSize, scale int) MaxPoolingLayer {
	return MaxPoolingLayer{
		scale:     scale,
		inputSize: size,
		OutputSize: tensor.TensorSize{
			Width:  size.Width / scale,
			Height: size.Height / scale,
			Depth:  size.Depth,
		},
		mask: tensor.NewTensor(size),
	}
}

func (l *MaxPoolingLayer) Forward(X tensor.Tensor) tensor.Tensor {
	output := tensor.NewTensor(l.OutputSize)

	for d := 0; d < l.inputSize.Depth; d++ {
		for i := 0; i < l.inputSize.Height; i += l.scale {
			for j := 0; j < l.inputSize.Width; j += l.scale {
				imax := i
				jmax := j
				max := X.GetValue(d, i, j)

				for y := i; y < i+l.scale; y++ {
					for x := j; x < j+l.scale; x++ {
						value := X.GetValue(d, y, x)
						l.mask.SetValue(d, y, x, 0)

						if value > max {
							max = value
							imax = y
							jmax = x
						}
					}
				}

				output.SetValue(d, i/l.scale, j/l.scale, max)
				l.mask.SetValue(d, imax, jmax, 1)
			}
		}
	}

	return output
}

func (l *MaxPoolingLayer) Backward(dout, X tensor.Tensor) tensor.Tensor {
	dX := tensor.NewTensor(l.inputSize)

	for d := 0; d < l.inputSize.Depth; d++ {
		for i := 0; i < l.inputSize.Height; i++ {
			for j := 0; j < l.inputSize.Width; j++ {
				dX.SetValue(d, i, j, dout.GetValue(d, i/l.scale, j/l.scale)*l.mask.GetValue(d, i, j))
			}
		}
	}
	return dX
}
