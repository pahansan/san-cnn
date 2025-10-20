package main

import (
	"fmt"
	"math/rand"
	"san-cnn/internal/nn"
	"san-cnn/internal/parser"
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

func maxIndex(t tensor.Tensor) int {
	max := 0.0
	var idx int
	for i := 0; i < t.GetSize().Depth; i++ {
		value := t.GetValue(i, 0, 0)
		if value > max {
			max = value
			idx = i
		}
	}
	return idx
}

func shuffle(slice []parser.Sample) {
	for i := range slice {
		j := rand.Intn(len(slice))
		slice[i], slice[j] = slice[j], slice[i]
	}
}

func prepareData(data [][]float64) {
	for _, ex := range data {
		input := ex[1:]
		for j := range input {
			input[j] = input[j] / 255
		}
	}
}

func countAccuracy(data []parser.Sample, model nn.LeNet5) float64 {
	correctCount := 0
	for _, ex := range data {
		out := model.Forward(ex.Input)
		ans := maxIndex(out)
		if ans == ex.Answer {
			correctCount++
		}
	}
	return float64(correctCount) / 10000 * 100
}

func main() {
	fmt.Println("Parsing...")
	strs, _ := parser.ReadCSV("mnist_train.csv")
	train := parser.ParseLines(strs)
	strs, _ = parser.ReadCSV("mnist_test.csv")
	test := parser.ParseLines(strs)

	fmt.Println("Train...")
	net := nn.NewLeNet5(tensor.TensorSize{Depth: 1, Height: 28, Width: 28})
	accuracy := 0.0
	targetAccuracy := 95.0
	for j := 0; accuracy <= targetAccuracy; j++ {
		shuffle(train)
		for i, ex := range train {
			out := net.Forward(ex.Input)
			_, dLoss := mseLoss(out, ex.Target)
			net.Backward(dLoss)
			net.UpdateWeights(0.1)
			if i%10000 == 0 {
				accuracy = countAccuracy(test, net)
				fmt.Println("Iteration:", i+j*60000, "Accuracy:", accuracy, "%")
				if accuracy >= targetAccuracy {
					break
				}
			}
		}
	}
}
