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

func meanSquaredError(expected, predicted *mat.Dense) float64 {
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

func sumAlongAxis0(matrix *mat.Dense) *mat.Dense {
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
	iterations := 10
	learning_rate := []float64{100}

	dataX := []float64{
		1, 0,
		0, 1,
		1, 1,
		0, 0,
	}
	input := mat.NewDense(4, 2, dataX)

	dataY := []float64{1, 1, 0, 0}
	expectedOutput := mat.NewDense(4, 1, dataY)

	weights_input_to_hidden_one := randomMatrix(2, 3)
	bias_hidden_one := zeroMatrix(1, 3)

	weights_hidden_one_to_hidden_two := randomMatrix(3, 3)
	bias_hidden_two := zeroMatrix(1, 3)

	weights_hidden_two_to_output := randomMatrix(3, 1)
	bias_output := zeroMatrix(1, 1)

	for i := 0; i <= iterations; i++ {
		fmt.Println("Iteration : ", i)
		for i := 0; i < input.RawMatrix().Rows; i++ {
			sampleInput := mat.NewDense(1, input.RawMatrix().Cols, mat.Row(nil, i, input))
			targetOutput := mat.NewDense(1, 1, []float64{expectedOutput.At(i, 0)})

			fmt.Printf("Sample Input: %v -> Target Output: %v\n", mat.Formatted(sampleInput), mat.Formatted(targetOutput))

			result_hidden_one := forwardLayer(sampleInput, weights_input_to_hidden_one, bias_hidden_one, ReLU)
			result_hidden_two := forwardLayer(result_hidden_one, weights_hidden_one_to_hidden_two, bias_hidden_two, ReLU)
			result_output := forwardLayer(result_hidden_two, weights_hidden_two_to_output, bias_output, Sigmoid)

			fmt.Print("Pediction : ")
			if result_output.At(0, 0) < 0.3 {
				fmt.Println("0")
			} else {
				fmt.Println("1")
			}

			loss := meanSquaredError(targetOutput, result_output)
			fmt.Printf("Loss : %v\n", loss)

			// Backprop
			var gradient_sum_output mat.Dense
			gradient_sum_output.Sub(result_output, targetOutput)
			var gradient_output_to_hidden_two mat.Dense
			gradient_output_to_hidden_two.Mul(result_hidden_two.T(), &gradient_sum_output)
			gradient_bias_output := sumAlongAxis0(&gradient_sum_output)

			var gradient_sum_hidden_two mat.Dense
			gradient_sum_hidden_two.Mul(&gradient_sum_output, weights_hidden_two_to_output.T())
			gradient_sum_hidden_two.MulElem(&gradient_sum_hidden_two, ReLUDerivative(result_hidden_two))
			var gradient_hidden_one_to_hidden_two mat.Dense
			gradient_hidden_one_to_hidden_two.Mul(result_hidden_one.T(), &gradient_sum_hidden_two)
			gradient_bias_hidden_two := sumAlongAxis0(&gradient_sum_hidden_two)

			var gradient_sum_hidden_one mat.Dense
			gradient_sum_hidden_one.Mul(&gradient_sum_hidden_two, weights_hidden_one_to_hidden_two.T())
			gradient_sum_hidden_one.MulElem(&gradient_sum_hidden_one, ReLUDerivative(result_hidden_one))
			var gradient_input_to_hidden_one mat.Dense
			gradient_input_to_hidden_one.Mul(sampleInput.T(), &gradient_sum_hidden_one)
			gradient_bias_hidden_one := sumAlongAxis0(&gradient_sum_hidden_one)

			// Update weights
			var temp_input_to_hidden_one mat.Dense
			temp_input_to_hidden_one.Scale(learning_rate[0], &gradient_input_to_hidden_one)
			weights_input_to_hidden_one.Sub(weights_input_to_hidden_one, &temp_input_to_hidden_one)

			var temp_hidden_one_to_hidden_two mat.Dense
			temp_hidden_one_to_hidden_two.Scale(learning_rate[0], &gradient_hidden_one_to_hidden_two)
			weights_hidden_one_to_hidden_two.Sub(weights_hidden_one_to_hidden_two, &temp_hidden_one_to_hidden_two)

			var temp_hidden_two_to_output mat.Dense
			temp_hidden_two_to_output.Scale(learning_rate[0], &gradient_output_to_hidden_two)
			weights_hidden_two_to_output.Sub(weights_hidden_two_to_output, &temp_hidden_two_to_output)

			// Update biases
			var temp_bias_hidden_one mat.Dense
			temp_bias_hidden_one.Scale(learning_rate[0], gradient_bias_hidden_one)
			bias_hidden_one.Sub(bias_hidden_one, &temp_bias_hidden_one)

			var temp_bias_hidden_two mat.Dense
			temp_bias_hidden_two.Scale(learning_rate[0], gradient_bias_hidden_two)
			bias_hidden_two.Sub(bias_hidden_two, &temp_bias_hidden_two)

			var temp_bias_output mat.Dense
			temp_bias_output.Scale(learning_rate[0], gradient_bias_output)
			bias_output.Sub(bias_output, &temp_bias_output)
		}
	}
}
