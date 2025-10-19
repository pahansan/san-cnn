package fullyconnectedlayer

import (
	"math"
	"math/rand/v2"
	"san-cnn/internal/matrix"
	"san-cnn/internal/tensor"
)

type ActivationType int

const (
	None ActivationType = iota
	Sigmoid
	Tanh
	ReLU
	LeakyReLU
	ELU
)

type FullyConnectedLayer struct {
	inputSize      tensor.TensorSize
	outputSize     tensor.TensorSize
	inputs         int
	outputs        int
	activationType ActivationType
	df             tensor.Tensor
	w              matrix.Matrix
	dw             matrix.Matrix
	b              []float64
	db             []float64
}

func (l *FullyConnectedLayer) InitWeightsRand() {
	sigma := math.Sqrt(2.0 / float64(l.inputSize.Depth*l.inputSize.Height*l.inputSize.Width))
	for i := 0; i < l.outputs; i++ {
		for j := 0; j < l.inputs; j++ {
			l.w.SetValue(i, j, rand.NormFloat64()*sigma)
		}
		l.b[i] = 0.01
	}
}

func NewFullyConnectedLayer(size tensor.TensorSize, outputs int, activationType ActivationType) FullyConnectedLayer {
	l := FullyConnectedLayer{
		w:  matrix.NewMatrix(outputs, size.Height*size.Width*size.Depth),
		dw: matrix.NewMatrix(outputs, size.Height*size.Width*size.Depth),
		df: tensor.NewTensor(tensor.TensorSize{
			Width:  1,
			Height: 1,
			Depth:  outputs,
		}),
		inputSize: size,
		outputSize: tensor.TensorSize{
			Width:  1,
			Height: 1,
			Depth:  outputs,
		},
		inputs:         size.Depth * size.Height * size.Width,
		outputs:        outputs,
		activationType: activationType,
		b:              make([]float64, outputs),
		db:             make([]float64, outputs),
	}
	l.InitWeightsRand()
	return l
}

func (l *FullyConnectedLayer) Activate(output *tensor.Tensor) {
	if l.activationType == ReLU {
		for i := 0; i < l.outputs; i++ {
			if output.GetValue(i, 0, 0) > 0 {
				l.df.SetValue(i, 0, 0, 1)
			} else {
				output.SetValue(i, 0, 0, 0)
				l.df.SetValue(i, 0, 0, 0)
			}
		}
	}
}

func (l *FullyConnectedLayer) Forward(X tensor.Tensor) tensor.Tensor {
	output := tensor.NewTensor(l.outputSize)

	for i := 0; i < l.outputs; i++ {
		sum := l.b[i]

		for j := 0; j < l.inputs; j++ {
			sum += l.w.GetValue(i, j) * X.GetValue(j, 0, 0)
		}
		output.SetValue(i, 0, 0, sum)
	}

	l.Activate(&output)

	return output
}

func (l *FullyConnectedLayer) Backward(dout, X tensor.Tensor) tensor.Tensor {
	for i := 0; i < l.outputs; i++ {
		*l.df.GetValuePtr(i, 0, 0) *= dout.GetValue(i, 0, 0)
	}

	for i := 0; i < l.outputs; i++ {
		for j := 0; j < l.inputs; j++ {
			l.dw.SetValue(i, j, l.df.GetValue(i, 0, 0)*X.GetValue(j, 0, 0))
		}
		l.db[i] = l.df.GetValue(i, 0, 0)
	}

	dX := tensor.NewTensor(l.inputSize)

	for j := 0; j < l.inputs; j++ {
		sum := 0.0
		for i := 0; i < l.outputs; i++ {
			sum += l.w.GetValue(i, j) * l.df.GetValue(i, 0, 0)
		}
		dX.SetValue(j, 0, 0, sum)
	}

	return dX
}

func (l *FullyConnectedLayer) UpdateWeights(learningRate float64) {
	for i := 0; i < l.outputs; i++ {
		for j := 0; j < l.inputs; j++ {
			*l.w.GetValuePtr(i, j) -= learningRate * l.dw.GetValue(i, j)
		}
		l.b[i] -= learningRate * l.db[i]
	}
}
