package utils

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func RandomMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64() * math.Sqrt(2/float64(rows))
	}
	return mat.NewDense(rows, cols, data)
}

func ZeroMatrix(rows, cols int) *mat.Dense {
	return mat.NewDense(rows, cols, nil)
}

func MeanSquaredError(expected, predicted *mat.Dense) float64 {
	rows, cols := expected.Dims()
	if r, c := predicted.Dims(); r != rows || c != cols {
		panic("matrices must have the same dimensions")
	}

	var sum float64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := expected.At(i, j) - predicted.At(i, j)
			sum += diff * diff
		}
	}

	mean := sum / float64(rows*cols)
	return mean
}

func SumAlongAxis0(matrix *mat.Dense) *mat.Dense {
	rows, cols := matrix.Dims()
	result := mat.NewDense(1, cols, nil)

	for j := 0; j < cols; j++ {
		var sum float64
		for i := 0; i < rows; i++ {
			sum += matrix.At(i, j)
		}
		result.Set(0, j, sum)
	}

	return result
}
