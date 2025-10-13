package tensor

type Tensor struct {
	depth  int
	height int
	width  int
	dw     int
	Values []float64
}

func NewTensor(depth, height, width int) Tensor {
	return Tensor{
		depth:  depth,
		height: height,
		width:  width,
		dw:     depth * width,
		Values: make([]float64, depth*height*width),
	}
}

func (t *Tensor) GetValue(d, i, j int) float64        { return t.Values[i*t.dw+j*t.depth+d] }
func (t *Tensor) SetValue(d, i, j int, value float64) { t.Values[i*t.dw+j*t.depth+d] = value }
func (t *Tensor) GetValuePtr(d, i, j int) *float64    { return &t.Values[i*t.dw+j*t.depth+d] }
func (t *Tensor) GetDepth() int                       { return t.depth }
func (t *Tensor) GetHeight() int                      { return t.height }
func (t *Tensor) GetWidth() int                       { return t.width }
