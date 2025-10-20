package main

import (
	"fmt"
	"math/rand"
	"san-cnn/internal/nn"
	"san-cnn/internal/tensor"
)

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
	net := nn.NewLeNet5(tensor.TensorSize{Depth: 1, Height: 28, Width: 28})

	X := tensor.NewTensor(tensor.TensorSize{Depth: 1, Height: 28, Width: 28})
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			X.SetValue(0, i, j, rand.Float64())
		}
	}

	// Простейшая цель (one-hot, класс 3)
	Y := tensor.NewTensor(tensor.TensorSize{Depth: 10, Height: 1, Width: 1})
	Y.SetValue(3, 0, 0, 1.0)

	// Forward
	out := net.Forward(X)
	loss, dLoss := mseLoss(out, Y)
	fmt.Println("Loss:", loss)

	// Backward
	net.Backward(dLoss)

	// Update
	net.UpdateWeights(0.01)
}
