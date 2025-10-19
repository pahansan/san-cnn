package matrix

type Matrix struct {
	rows   int
	cols   int
	values [][]float64
}

func NewMatrix(rows, cols int) Matrix {
	m := Matrix{
		rows:   rows,
		cols:   cols,
		values: make([][]float64, rows),
	}

	for i := 0; i < rows; i++ {
		m.values[i] = make([]float64, cols)
	}

	return m
}

func (m *Matrix) GetValue(i, j int) float64        { return m.values[i][j] }
func (m *Matrix) SetValue(i, j int, value float64) { m.values[i][j] = value }
func (m *Matrix) GetValuePtr(i, j int) *float64    { return &m.values[i][j] }
