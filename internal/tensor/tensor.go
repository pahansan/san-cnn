package tensor

type TensorSize struct {
	Depth  int
	Height int
	Width  int
}

type Tensor struct {
	size   TensorSize
	dw     int
	Values []float64
}

func NewTensor(size TensorSize) Tensor {
	return Tensor{
		size:   size,
		dw:     size.Depth * size.Width,
		Values: make([]float64, size.Depth*size.Height*size.Width),
	}
}

func (t *Tensor) GetValue(d, i, j int) float64        { return t.Values[i*t.dw+j*t.size.Depth+d] }
func (t *Tensor) SetValue(d, i, j int, value float64) { t.Values[i*t.dw+j*t.size.Depth+d] = value }
func (t *Tensor) GetValuePtr(d, i, j int) *float64    { return &t.Values[i*t.dw+j*t.size.Depth+d] }
func (t *Tensor) GetSize() TensorSize                 { return t.size }
