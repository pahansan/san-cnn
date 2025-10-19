package convlayer

import (
	"math"
	"math/rand"
	"san-cnn/internal/tensor"
)

type ConvLayer struct {
	InputSize  tensor.TensorSize
	OutputSize tensor.TensorSize
	W          []tensor.Tensor
	B          []float64
	DW         []tensor.Tensor
	DB         []float64
	P          int
	S          int
	Fc         int
	Fs         int
	Fd         int
}

func (l *ConvLayer) InitWeightsRand() {
	sigma := math.Sqrt(2.0 / (float64(l.Fs * l.Fs * l.Fd)))
	for index := 0; index < l.Fc; index++ {
		for i := 0; i < l.Fs; i++ {
			for j := 0; j < l.Fs; j++ {
				for k := 0; k < l.Fd; k++ {
					*l.W[index].GetValuePtr(k, i, j) = rand.NormFloat64() * sigma
				}
				l.B[index] = 0.01
			}
		}
	}
}

func NewConvLayer(size tensor.TensorSize, fc, fs, p, s int) ConvLayer {
	newLayer := ConvLayer{
		InputSize: size,
		OutputSize: tensor.TensorSize{
			Width:  (size.Width-fs+2*p)/s + 1,
			Height: (size.Height-fs+2*p)/s + 1,
			Depth:  fc,
		},
		P:  p,
		S:  s,
		Fc: fc,
		Fs: fs,
		Fd: size.Depth,
		W:  make([]tensor.Tensor, fc),
		DW: make([]tensor.Tensor, fc),
		B:  make([]float64, fc),
		DB: make([]float64, fc),
	}

	for i := range newLayer.W {
		newLayer.W[i] = tensor.NewTensor(tensor.TensorSize{Width: fs, Height: fs, Depth: size.Depth})
		newLayer.DW[i] = tensor.NewTensor(tensor.TensorSize{Width: fs, Height: fs, Depth: size.Depth})
	}

	newLayer.InitWeightsRand()
	return newLayer
}

func (l *ConvLayer) Forward(X tensor.Tensor) tensor.Tensor {
	output := tensor.NewTensor(l.OutputSize)

	for f := 0; f < l.Fc; f++ {
		for y := 0; y < l.OutputSize.Height; y++ {
			for x := 0; x < l.InputSize.Width; x++ {
				sum := l.B[f]

				for i := 0; i < l.Fs; i++ {
					for j := 0; j < l.Fs; j++ {
						i0 := l.S*y + i - l.P
						j0 := l.S*y + j - l.P

						if i0 < 0 || i0 >= l.InputSize.Height || j0 < 0 || j0 >= l.InputSize.Width {
							continue
						}

						for c := 0; c < l.Fd; c++ {
							sum += X.GetValue(c, i0, j0) * l.W[f].GetValue(c, i, j)
						}
					}
				}

				output.SetValue(f, y, x, sum)
			}
		}
	}
	return output
}

func (l *ConvLayer) Backward(dout tensor.Tensor, X tensor.Tensor) tensor.Tensor {
	deltas := tensor.NewTensor(tensor.TensorSize{
		Height: l.S*(l.OutputSize.Height-1) + 1,
		Width:  l.S*(l.OutputSize.Width-1) + 1,
		Depth:  l.OutputSize.Depth,
	})

	for d := 0; d < deltas.GetSize().Depth; d++ {
		for i := 0; i < l.OutputSize.Height; i++ {
			for j := 0; j < l.OutputSize.Width; j++ {
				deltas.SetValue(d, i*l.S, j*l.S, dout.GetValue(d, i, j))
			}
		}
	}

	for f := 0; f < l.Fc; f++ {
		for y := 0; y < deltas.GetSize().Height; y++ {
			for x := 0; x < deltas.GetSize().Width; x++ {
				delta := deltas.GetValue(f, y, x)

				for i := 0; i < l.Fs; i++ {
					for j := 0; j < l.Fs; j++ {
						i0 := i + y - l.P
						j0 := j + x - l.P

						if i0 < 0 || i0 >= l.InputSize.Height || j0 < 0 || j0 >= l.InputSize.Width {
							continue
						}

						for c := 0; c < l.Fd; c++ {
							*l.DW[f].GetValuePtr(c, i, j) += delta * X.GetValue(c, i0, j0)
						}
					}
				}

				l.DB[f] += delta
			}
		}
	}

	pad := l.Fs - 1 + l.P
	dX := tensor.NewTensor(l.InputSize)

	for y := 0; y < l.InputSize.Height; y++ {
		for x := 0; x < l.InputSize.Width; x++ {
			for c := 0; c < l.Fd; c++ {
				sum := 0.0

				for i := 0; i < l.Fs; i++ {
					for j := 0; j < l.Fs; j++ {
						i0 := y + i - pad
						j0 := x + j - pad

						if i0 < 0 || i0 >= deltas.GetSize().Height || j0 < 0 || j0 >= deltas.GetSize().Width {
							continue
						}

						for f := 0; f < l.Fc; f++ {
							sum += l.W[f].GetValue(c, l.Fs-1-i, l.Fs-1-j) * deltas.GetValue(f, i0, j0)
						}
					}
				}

				dX.SetValue(c, y, x, sum)
			}
		}
	}

	return dX
}

func (l *ConvLayer) UpdateWeights(learningRate float64) {
	for index := 0; index < l.Fc; index++ {
		for i := 0; i < l.Fs; i++ {
			for j := 0; j < l.Fs; j++ {
				for d := 0; d < l.Fd; d++ {
					*l.W[index].GetValuePtr(d, i, j) -= learningRate * l.DW[index].GetValue(d, i, j)
					l.DW[index].SetValue(d, i, j, 0)
				}
			}
		}

		l.B[index] -= learningRate * l.DB[index]
		l.DB[index] = 0
	}
}
