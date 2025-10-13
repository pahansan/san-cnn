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
