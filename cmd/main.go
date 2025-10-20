package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

type Tensor struct {
	C, H, W int
	Data    []float64
}

func NewTensor(c, h, w int) *Tensor {
	return &Tensor{C: c, H: h, W: w, Data: make([]float64, c*h*w)}
}

func (t *Tensor) At(c, y, x int) float64     { return t.Data[(c*t.H+y)*t.W+x] }
func (t *Tensor) Set(c, y, x int, v float64) { t.Data[(c*t.H+y)*t.W+x] = v }

func RandTensor(c, h, w int, scale float64) *Tensor {
	t := NewTensor(c, h, w)
	for i := range t.Data {
		t.Data[i] = (rand.NormFloat64()) * scale
	}
	return t
}

func ZerosLike(t *Tensor) *Tensor { return NewTensor(t.C, t.H, t.W) }

func (t *Tensor) Flatten() []float64 {
	out := make([]float64, len(t.Data))
	copy(out, t.Data)
	return out
}

type Layer interface {
	Forward(input *Tensor) (*Tensor, error)
	Backward(dout *Tensor) (*Tensor, error)
	Params() []*Param
}

type Param struct {
	Val  []float64
	Grad []float64
}

func NewParam(n int) *Param {
	p := &Param{Val: make([]float64, n), Grad: make([]float64, n)}
	return p
}

type Conv2D struct {
	InC, OutC, K int
	W            *Param
	B            *Param

	lastInput *Tensor
}

func NewConv2D(inC, outC, k int) *Conv2D {
	n := outC * inC * k * k
	w := NewParam(n)
	scale := math.Sqrt(2.0 / float64(inC*k*k))
	for i := range w.Val {
		w.Val[i] = rand.NormFloat64() * scale
	}
	b := NewParam(outC)
	return &Conv2D{InC: inC, OutC: outC, K: k, W: w, B: b}
}

func (c *Conv2D) Params() []*Param { return []*Param{c.W, c.B} }

func (c *Conv2D) Forward(input *Tensor) (*Tensor, error) {
	if input.C != c.InC {
		return nil, errors.New("conv in channels mismatch")
	}
	c.lastInput = input
	h := input.H - c.K + 1
	w := input.W - c.K + 1
	out := NewTensor(c.OutC, h, w)
	for oc := 0; oc < c.OutC; oc++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var sum float64 = 0
				for ic := 0; ic < c.InC; ic++ {
					for ky := 0; ky < c.K; ky++ {
						for kx := 0; kx < c.K; kx++ {
							wIdx := (((oc*c.InC+ic)*c.K+ky)*c.K + kx)
							inVal := input.At(ic, y+ky, x+kx)
							sum += c.W.Val[wIdx] * inVal
						}
					}
				}
				sum += c.B.Val[oc]
				out.Set(oc, y, x, sum)
			}
		}
	}
	return out, nil
}

func (c *Conv2D) Backward(dout *Tensor) (*Tensor, error) {
	in := c.lastInput
	if in == nil {
		return nil, errors.New("conv backward called before forward")
	}
	hout := dout.H
	wout := dout.W
	for i := range c.W.Grad {
		c.W.Grad[i] = 0
	}
	for i := range c.B.Grad {
		c.B.Grad[i] = 0
	}
	dx := ZerosLike(in)
	for oc := 0; oc < c.OutC; oc++ {
		for y := 0; y < hout; y++ {
			for x := 0; x < wout; x++ {
				d := dout.At(oc, y, x)
				c.B.Grad[oc] += d
				for ic := 0; ic < c.InC; ic++ {
					for ky := 0; ky < c.K; ky++ {
						for kx := 0; kx < c.K; kx++ {
							wIdx := (((oc*c.InC+ic)*c.K+ky)*c.K + kx)
							inVal := in.At(ic, y+ky, x+kx)
							c.W.Grad[wIdx] += d * inVal
							dx.Set(ic, y+ky, x+kx, dx.At(ic, y+ky, x+kx)+d*c.W.Val[wIdx])
						}
					}
				}
			}
		}
	}
	return dx, nil
}

type AvgPool2 struct {
	Kernel    int
	lastInput *Tensor
}

func NewAvgPool2() *AvgPool2 { return &AvgPool2{Kernel: 2} }

func (p *AvgPool2) Params() []*Param { return nil }

func (p *AvgPool2) Forward(input *Tensor) (*Tensor, error) {
	p.lastInput = input
	h := input.H / p.Kernel
	w := input.W / p.Kernel
	out := NewTensor(input.C, h, w)
	k := p.Kernel
	for c0 := 0; c0 < input.C; c0++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var s float64
				for ky := 0; ky < k; ky++ {
					for kx := 0; kx < k; kx++ {
						s += input.At(c0, y*k+ky, x*k+kx)
					}
				}
				out.Set(c0, y, x, s/float64(k*k))
			}
		}
	}
	return out, nil
}

func (p *AvgPool2) Backward(dout *Tensor) (*Tensor, error) {
	in := p.lastInput
	if in == nil {
		return nil, errors.New("pool backward before forward")
	}
	dx := ZerosLike(in)
	hout := dout.H
	wout := dout.W
	k := p.Kernel
	for c0 := 0; c0 < in.C; c0++ {
		for y := 0; y < hout; y++ {
			for x := 0; x < wout; x++ {
				grad := dout.At(c0, y, x) / float64(k*k)
				for ky := 0; ky < k; ky++ {
					for kx := 0; kx < k; kx++ {
						dx.Set(c0, y*k+ky, x*k+kx, dx.At(c0, y*k+ky, x*k+kx)+grad)
					}
				}
			}
		}
	}
	return dx, nil
}

type ReLU struct{ last *Tensor }

func NewReLU() *ReLU             { return &ReLU{} }
func (r *ReLU) Params() []*Param { return nil }

func (r *ReLU) Forward(input *Tensor) (*Tensor, error) {
	r.last = input
	out := NewTensor(input.C, input.H, input.W)
	for i, v := range input.Data {
		if v > 0 {
			out.Data[i] = v
		} else {
			out.Data[i] = 0
		}
	}
	return out, nil
}

func (r *ReLU) Backward(dout *Tensor) (*Tensor, error) {
	in := r.last
	if in == nil {
		return nil, errors.New("relu backward before forward")
	}
	dx := ZerosLike(in)
	for i := range dx.Data {
		if in.Data[i] > 0 {
			dx.Data[i] = dout.Data[i]
		} else {
			dx.Data[i] = 0
		}
	}
	return dx, nil
}

type Flatten struct{ inShape [3]int }

func NewFlatten() *Flatten          { return &Flatten{} }
func (f *Flatten) Params() []*Param { return nil }

func (f *Flatten) Forward(input *Tensor) (*Tensor, error) {
	f.inShape = [3]int{input.C, input.H, input.W}
	out := NewTensor(1, 1, len(input.Data))
	copy(out.Data, input.Data)
	return out, nil
}

func (f *Flatten) Backward(dout *Tensor) (*Tensor, error) {
	c, h, w := f.inShape[0], f.inShape[1], f.inShape[2]
	if dout.W != c*h*w {
		return nil, errors.New("flatten backward size mismatch")
	}
	out := NewTensor(c, h, w)
	copy(out.Data, dout.Data)
	return out, nil
}

type Dense struct {
	In, Out   int
	W         *Param
	B         *Param
	lastInput []float64
}

func NewDense(in, out int) *Dense {
	pW := NewParam(in * out)
	scale := math.Sqrt(2.0 / float64(in))
	for i := range pW.Val {
		pW.Val[i] = rand.NormFloat64() * scale
	}
	pB := NewParam(out)
	return &Dense{In: in, Out: out, W: pW, B: pB}
}

func (d *Dense) Params() []*Param { return []*Param{d.W, d.B} }

func (d *Dense) Forward(input *Tensor) (*Tensor, error) {
	vec := input.Data
	if len(vec) != d.In {
		return nil, errors.New("dense forward size mismatch")
	}
	d.lastInput = make([]float64, d.In)
	copy(d.lastInput, vec)
	out := NewTensor(1, 1, d.Out)
	for j := 0; j < d.Out; j++ {
		var s float64 = d.B.Val[j]
		for i := 0; i < d.In; i++ {
			s += d.W.Val[j*d.In+i] * vec[i]
		}
		out.Data[j] = s
	}
	return out, nil
}

func (d *Dense) Backward(dout *Tensor) (*Tensor, error) {
	if len(dout.Data) != d.Out {
		return nil, errors.New("dense backward size mismatch")
	}
	for i := range d.W.Grad {
		d.W.Grad[i] = 0
	}
	for i := range d.B.Grad {
		d.B.Grad[i] = 0
	}
	for j := 0; j < d.Out; j++ {
		d.B.Grad[j] += dout.Data[j]
	}
	for j := 0; j < d.Out; j++ {
		for i := 0; i < d.In; i++ {
			d.W.Grad[j*d.In+i] += dout.Data[j] * d.lastInput[i]
		}
	}
	dx := NewTensor(1, 1, d.In)
	for i := 0; i < d.In; i++ {
		var s float64
		for j := 0; j < d.Out; j++ {
			s += d.W.Val[j*d.In+i] * dout.Data[j]
		}
		dx.Data[i] = s
	}
	return dx, nil
}

func SoftmaxCrossEntropyLoss(logits *Tensor, label int) (float64, *Tensor, error) {
	N := logits.W
	if label < 0 || label >= N {
		return 0, nil, errors.New("label out of range")
	}
	max := logits.Data[0]
	for i := 1; i < N; i++ {
		if logits.Data[i] > max {
			max = logits.Data[i]
		}
	}
	exps := make([]float64, N)
	var sum float64
	for i := 0; i < N; i++ {
		exps[i] = math.Exp(logits.Data[i] - max)
		sum += exps[i]
	}
	for i := 0; i < N; i++ {
		exps[i] /= sum
	}
	loss := -math.Log(exps[label] + 1e-15)
	dlogits := NewTensor(1, 1, N)
	for i := 0; i < N; i++ {
		dlogits.Data[i] = exps[i]
	}
	dlogits.Data[label] -= 1
	return loss, dlogits, nil
}

type Sequential struct {
	Layers []Layer
}

func NewSequential(layers ...Layer) *Sequential { return &Sequential{Layers: layers} }

func (s *Sequential) Forward(input *Tensor) (*Tensor, error) {
	x := input
	var err error
	for _, l := range s.Layers {
		x, err = l.Forward(x)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}

func (s *Sequential) Backward(dout *Tensor) error {
	d := dout
	var err error
	for i := len(s.Layers) - 1; i >= 0; i-- {
		d, err = s.Layers[i].Backward(d)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Sequential) Params() []*Param {
	var ps []*Param
	for _, l := range s.Layers {
		ps = append(ps, l.Params()...)
	}
	return ps
}

type SGD struct{ LR float64 }

func (opt *SGD) Step(params []*Param) {
	for _, p := range params {
		for i := range p.Val {
			p.Val[i] -= opt.LR * p.Grad[i]
		}
	}
}

func NewLeNet5() *Sequential {
	c1 := NewConv2D(1, 6, 5)
	r1 := NewReLU()
	p2 := NewAvgPool2()
	c3 := NewConv2D(6, 16, 5)
	r3 := NewReLU()
	p4 := NewAvgPool2()
	f := NewFlatten()
	fc5 := NewDense(16*5*5, 120)
	r5 := NewReLU()
	fc6 := NewDense(120, 84)
	r6 := NewReLU()
	out := NewDense(84, 10)
	return NewSequential(c1, r1, p2, c3, r3, p4, f, fc5, r5, fc6, r6, out)
}

func LoadCSVImages(path string) (images []*Tensor, labels []int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	for {
		rec, err := r.Read()
		if err != nil {
			if errors.Is(err, os.ErrClosed) {
				break
			}
			break
		}
		if len(rec) < 2 {
			continue
		}
		lab, _ := strconv.Atoi(rec[0])
		pix := make([]float64, len(rec)-1)
		for i := 1; i < len(rec); i++ {
			v, _ := strconv.ParseFloat(rec[i], 64)
			pix[i-1] = v / 255.0
		}
		sz := int(math.Sqrt(float64(len(pix))))
		img := NewTensor(1, 32, 32)
		for y := 0; y < sz; y++ {
			for x := 0; x < sz; x++ {
				img.Set(0, y+(32-sz)/2, x+(32-sz)/2, pix[y*sz+x])
			}
		}
		images = append(images, img)
		labels = append(labels, lab)
	}
	if len(images) == 0 {
		return nil, nil, errors.New("no images loaded")
	}
	return images, labels, nil
}

func validate(data []*Tensor, ans []int, model *Sequential) float64 {
	correct := 0
	for i := range data {
		out, _ := model.Forward(data[i])
		pred := 0
		max := out.Data[0]
		for j, v := range out.Data {
			if v > max {
				max = v
				pred = j
			}
		}
		if pred == ans[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(data)) * 100
}

func shuffle(data []*Tensor, ans []int) {
	for i := range data {
		j := rand.Intn(len(data))
		data[i], data[j] = data[j], data[i]
		ans[i], ans[j] = ans[j], ans[i]
	}
}

func main() {
	fmt.Println("Loading MNIST train data...")
	trainImages, trainLabels, err := LoadCSVImages("mnist_train.csv")
	if err != nil {
		panic(err)
	}
	fmt.Println("Loading MNIST test data...")
	testImages, testLabels, err := LoadCSVImages("mnist_test.csv")
	if err != nil {
		panic(err)
	}

	net := NewLeNet5()
	opt := &SGD{LR: 0.01}
	targetAcc := 90.0
	accuracy := 0.0
	epoch := 0

	file, _ := os.Create("data.txt")
	defer file.Close()

	fmt.Println("Train...")
	avgLoss := 0.0
	nSteps := 100
	for accuracy < targetAcc {
		shuffle(trainImages, trainLabels)

		for i := range trainImages {
			out, err := net.Forward(trainImages[i])
			if err != nil {
				panic(err)
			}

			loss, dlogits, _ := SoftmaxCrossEntropyLoss(out, trainLabels[i])
			avgLoss += loss
			err = net.Backward(dlogits)
			if err != nil {
				panic(err)
			}

			opt.Step(net.Params())
			if i%nSteps == 0 {
				accuracy = validate(testImages, testLabels, net)
				fmt.Printf("Iteration: %d Cost: %f Accuracy: %.2f %%\n", i+epoch*60000, avgLoss/float64(nSteps), accuracy)
				file.Write([]byte(fmt.Sprintf("%d %1.15f %.2f\n", i+epoch*60000, avgLoss/float64(nSteps), accuracy)))
				avgLoss = 0
				if accuracy >= targetAcc {
					break
				}
			}
		}
		epoch++
	}
	fmt.Println("Training complete. Target accuracy reached.")
}
