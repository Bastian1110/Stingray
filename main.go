package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type activation func(*mat.Dense) *mat.Dense

func randomMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64() * math.Sqrt(2/float64(rows))
	}
	return mat.NewDense(rows, cols, data)
}

func zeroMatrix(rows, cols int) *mat.Dense {
	return mat.NewDense(rows, cols, nil)
}

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

func forwardLayer(x *mat.Dense, weights *mat.Dense, bias *mat.Dense, b activation) *mat.Dense {
	var dotProductResult mat.Dense
	dotProductResult.Mul(x, weights)
	var result mat.Dense
	result.Add(&dotProductResult, bias)
	if b != nil {
		return b(&result)
	}
	return &result
}

func main() {
	data := []float64{
		1, 0,
	}

	// Create a 2x2 matrix with the provided data
	input := mat.NewDense(1, 2, data)

	weights_input_to_hidden_one := randomMatrix(2, 3)
	bias_hidden_one := zeroMatrix(1, 3)

	weights_hidden_one_to_hidden_two := randomMatrix(3, 3)
	bias_hidden_two := zeroMatrix(1, 3)

	weights_hidden_two_to_output := randomMatrix(3, 1)
	bias_output := zeroMatrix(1, 1)

	fmt.Printf("Matrix:\n%v\n", mat.Formatted(weights_input_to_hidden_one, mat.Prefix(" "), mat.Excerpt(0)))
	fmt.Printf("Matrix:\n%v\n", mat.Formatted(bias_hidden_one, mat.Prefix(" "), mat.Excerpt(0)))

	fmt.Printf("Matrix:\n%v\n", mat.Formatted(weights_hidden_one_to_hidden_two, mat.Prefix(" "), mat.Excerpt(0)))
	fmt.Printf("Matrix:\n%v\n", mat.Formatted(bias_hidden_two, mat.Prefix(" "), mat.Excerpt(0)))

	fmt.Printf("Matrix:\n%v\n", mat.Formatted(weights_hidden_two_to_output, mat.Prefix(" "), mat.Excerpt(0)))
	fmt.Printf("Matrix:\n%v\n", mat.Formatted(bias_output, mat.Prefix(" "), mat.Excerpt(0)))

	result_hidden_one := forwardLayer(input, weights_input_to_hidden_one, bias_hidden_one, ReLU)
	fmt.Printf("Res:\n%v\n", mat.Formatted(result_hidden_one, mat.Prefix(" "), mat.Excerpt(0)))

}
