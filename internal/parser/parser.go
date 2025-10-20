package parser

import (
	"encoding/csv"
	"os"
	"san-cnn/internal/tensor"
	"strconv"
)

type Sample struct {
	Input  tensor.Tensor
	Target tensor.Tensor
	Answer int
}

func formatTarget(t int) tensor.Tensor {
	tmp := tensor.NewTensor(tensor.TensorSize{Width: 1, Height: 1, Depth: 10})
	tmp.SetValue(t, 0, 0, 1)
	return tmp
}

func ReadCSV(path string) ([][]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	return records, err
}

func ParseLines(lines [][]string) []Sample {
	data := make([]Sample, len(lines))
	for i, line := range lines {
		data[i] = Sample{
			Input: tensor.NewTensor(tensor.TensorSize{
				Width:  28,
				Height: 28,
				Depth:  1,
			}),
		}

		for j, strNum := range line {
			if j == 0 {
				ans, _ := strconv.ParseInt(strNum, 10, 32)
				data[i].Answer = int(ans)
				data[i].Target = formatTarget(int(ans))
			} else {
				floatNum, _ := strconv.ParseFloat(strNum, 64)
				data[i].Input.SetValue(0, (j-1)/28, (j-1)%28, floatNum)
			}
		}
	}
	return data
}
