package main

import (
	"fmt"
	"math/rand"
	"san-cnn/internal/activationlayer"
	"san-cnn/internal/convlayer"
	"san-cnn/internal/fullyconnectedlayer"
	"san-cnn/internal/poolinglayer"
	"san-cnn/internal/tensor"
)

// Простой лосс: MSE
func mseLoss(pred, target tensor.Tensor) (float64, tensor.Tensor) {
	loss := 0.0
	grad := tensor.NewTensor(pred.GetSize())

	for d := 0; d < pred.GetSize().Depth; d++ {
		diff := pred.GetValue(d, 0, 0) - target.GetValue(d, 0, 0)
		loss += diff * diff
		grad.SetValue(d, 0, 0, 2*diff)
	}
	return loss / float64(pred.GetSize().Depth), grad
}

func main() {
	inputSize := tensor.TensorSize{Depth: 1, Height: 28, Width: 28}

	// === Архитектура сети ===
	conv1 := convlayer.NewConvLayer(inputSize, 4, 3, 1, 1)
	relu1 := activationlayer.NewReLULayer(conv1.OutputSize)
	pool1 := poolinglayer.NewMaxPoolingLayer(conv1.OutputSize, 2)

	fc1 := fullyconnectedlayer.NewFullyConnectedLayer(pool1.OutputSize, 10, fullyconnectedlayer.ReLU)

	// === Пример входа ===
	X := tensor.NewTensor(inputSize)

	for i := 0; i < X.GetSize().Height; i++ {
		for j := 0; j < X.GetSize().Width; j++ {
			for d := 0; d < X.GetSize().Depth; d++ {
				X.SetValue(d, i, j, rand.Float64())
			}
		}
	}

	// === Пример целевого выхода === (one-hot на класс 3)
	Y := tensor.NewTensor(tensor.TensorSize{Depth: 10, Height: 1, Width: 1})
	Y.SetValue(3, 0, 0, 1.0)

	learningRate := 0.01

	// === Прямое распространение ===
	out1 := conv1.Forward(X)
	out2 := relu1.Forward(out1)
	out3 := pool1.Forward(out2)
	out4 := fc1.Forward(out3)

	loss, dLoss := mseLoss(out4, Y)
	fmt.Printf("Loss: %.4f\n", loss)

	// === Обратное распространение ===
	d4 := fc1.Backward(dLoss, out3)
	d3 := pool1.Backward(d4, out2)
	d2 := relu1.Backward(d3, out1)
	_ = conv1.Backward(d2, X)

	// === Обновление весов ===
	conv1.UpdateWeights(learningRate)
	fc1.UpdateWeights(learningRate)
}
