package nn

import (
	"san-cnn/internal/activationlayer"
	"san-cnn/internal/convlayer"
	"san-cnn/internal/fullyconnectedlayer"
	"san-cnn/internal/poolinglayer"
	"san-cnn/internal/tensor"
)

type LeNet5 struct {
	// --- Слои ---
	c1    convlayer.ConvLayer
	a1    activationlayer.ReLULayer
	s2    poolinglayer.MaxPoolingLayer
	a2    activationlayer.ReLULayer
	c3    convlayer.ConvLayer
	a3    activationlayer.ReLULayer
	s4    poolinglayer.MaxPoolingLayer
	a4    activationlayer.ReLULayer
	c5    convlayer.ConvLayer
	a5    activationlayer.ReLULayer
	f6    fullyconnectedlayer.FullyConnectedLayer
	f7    fullyconnectedlayer.FullyConnectedLayer
	cache []tensor.Tensor
}

func NewLeNet5(inputSize tensor.TensorSize) LeNet5 {
	c1 := convlayer.NewConvLayer(inputSize, 6, 5, 2, 1)
	a1 := activationlayer.NewReLULayer(c1.OutputSize)
	s2 := poolinglayer.NewMaxPoolingLayer(c1.OutputSize, 2)
	a2 := activationlayer.NewReLULayer(s2.OutputSize)
	c3 := convlayer.NewConvLayer(s2.OutputSize, 16, 5, 0, 1)
	a3 := activationlayer.NewReLULayer(c3.OutputSize)
	s4 := poolinglayer.NewMaxPoolingLayer(c3.OutputSize, 2)
	a4 := activationlayer.NewReLULayer(s4.OutputSize)
	c5 := convlayer.NewConvLayer(s4.OutputSize, 120, 5, 0, 1)
	a5 := activationlayer.NewReLULayer(c5.OutputSize)
	f6 := fullyconnectedlayer.NewFullyConnectedLayer(c5.OutputSize, 84, fullyconnectedlayer.ReLU)
	f7 := fullyconnectedlayer.NewFullyConnectedLayer(f6.OutputSize, 10, fullyconnectedlayer.ReLU)

	return LeNet5{
		c1: c1, a1: a1,
		s2: s2, a2: a2,
		c3: c3, a3: a3,
		s4: s4, a4: a4,
		c5: c5, a5: a5,
		f6: f6, f7: f7,
	}
}

func (nn *LeNet5) Forward(input tensor.Tensor) tensor.Tensor {
	nn.cache = nil

	o1 := nn.c1.Forward(input)
	o2 := nn.a1.Forward(o1)
	o3 := nn.s2.Forward(o2)
	o4 := nn.a2.Forward(o3)
	o5 := nn.c3.Forward(o4)
	o6 := nn.a3.Forward(o5)
	o7 := nn.s4.Forward(o6)
	o8 := nn.a4.Forward(o7)
	o9 := nn.c5.Forward(o8)
	o10 := nn.a5.Forward(o9)
	o11 := nn.f6.Forward(o10)
	o12 := nn.f7.Forward(o11)

	nn.cache = []tensor.Tensor{input, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11}

	return o12
}

func (nn *LeNet5) Backward(dout tensor.Tensor) {
	d11 := nn.f7.Backward(dout, nn.cache[11])
	d10 := nn.f6.Backward(d11, nn.cache[10])
	d9 := nn.a5.Backward(d10, nn.cache[9])
	d8 := nn.c5.Backward(d9, nn.cache[8])
	d7 := nn.a4.Backward(d8, nn.cache[7])
	d6 := nn.s4.Backward(d7, nn.cache[6])
	d5 := nn.a3.Backward(d6, nn.cache[5])
	d4 := nn.c3.Backward(d5, nn.cache[4])
	d3 := nn.a2.Backward(d4, nn.cache[3])
	d2 := nn.s2.Backward(d3, nn.cache[2])
	d1 := nn.a1.Backward(d2, nn.cache[1])
	_ = nn.c1.Backward(d1, nn.cache[0])
}

func (nn *LeNet5) UpdateWeights(lr float64) {
	nn.c1.UpdateWeights(lr)
	nn.c3.UpdateWeights(lr)
	nn.c5.UpdateWeights(lr)
	nn.f6.UpdateWeights(lr)
	nn.f7.UpdateWeights(lr)
}
