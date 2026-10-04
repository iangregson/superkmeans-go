package h5

import (
	"math"
	"os"
	"testing"
)

func TestGloveFixture(t *testing.T) {
	path := "../../../__superkmeansgo/bench/data-src/glove-200-angular.hdf5"
	if _, err := os.Stat(path); err != nil {
		t.Skip("glove fixture not present")
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	train, err := f.Lookup("train")
	if err != nil {
		t.Fatal(err)
	}
	if train.Rows() != 1183514 || train.RowLen() != 200 || train.Kind != F32 {
		t.Fatalf("train = %v", train)
	}
	row := make([]float32, train.RowLen())
	if err := f.ReadF32(train, 0, 1, row); err != nil {
		t.Fatal(err)
	}
	for i, v := range row {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("row[%d] = %v", i, v)
		}
	}

	nb, err := f.Lookup("neighbors")
	if err != nil {
		t.Fatal(err)
	}
	if nb.Rows() != 10000 || nb.RowLen() != 100 || nb.Kind != I32 {
		t.Fatalf("neighbors = %v", nb)
	}
	ids := make([]int32, nb.RowLen())
	if err := f.ReadI32(nb, 0, 1, ids); err != nil {
		t.Fatal(err)
	}
	for _, id := range ids {
		if id < 0 || int(id) >= train.Rows() {
			t.Fatalf("neighbor id %d out of range", id)
		}
	}
}
