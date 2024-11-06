package activations

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type activation func(*mat.Dense) *mat.Dense

func ReLU(matrix *mat.Dense) *mat.Dense {
	rows, cols := matrix.Dims()
	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := matrix.At(i, j)
			result.Set(i, j, math.Max(0, value))
		}
	}
	return result
}

func ReLUDerivative(matrix *mat.Dense) *mat.Dense {
	rows, cols := matrix.Dims()
	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := matrix.At(i, j)
			if value > 0 {
				result.Set(i, j, 1)
			} else {
				result.Set(i, j, 0)
			}
		}
	}

	return result
}

func Sigmoid(matrix *mat.Dense) *mat.Dense {
	rows, cols := matrix.Dims()
	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := matrix.At(i, j)
			sigmoidValue := 1 / (1 + math.Exp(-value))
			result.Set(i, j, sigmoidValue)
		}
	}
	return result
}
