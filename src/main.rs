extern crate rand;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Range;
use std::ops::RangeTo;
use std::ops::RangeFrom;
use std::ops::RangeFull;

fn main() {
    let m = Matrix::rand(2, 3);
    let n = Matrix::rand(3, 2);
    let k = m.mult(&n);
    print!("{:?}\n", k.data);
    print!("{:?}\n", m.data);
    print!("{:?}\n", Matrix::from_vec(2, 3, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).data);
    print!("{:?}\n", Matrix::from_vec(2, 3, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).apply(|x: f32| x + 1.0).data);
    let nn = NeuralNetwork::new(3, vec![2,3,1], 0.001);
}


struct Matrix {
   rows: usize,
   cols: usize,
   data: Vec<f32>
}


impl Matrix {
    /* A matrix is in this case a way to represent the connections between
     * layers of a neural network. If you represent the initial input as a
     * row vector (1 row, n columns, which we do), then the each row of the
     * matrix represents the output weights of one neuron on the first layer,
     * and each column represents the input weights of one of the neurons on
     * the second layer.
     */
    
    fn _one() -> f32 {
        // function used to generate ones in build_matrix for Matrix::ones
        return 1.0;
    }

    fn build_matrix(rows: usize, cols: usize, gen: fn() -> f32) -> Matrix {
        let mut d: Vec<f32> = Vec::with_capacity(rows * cols);
        for i in 0..(rows * cols) {
            d.push(gen());
        }

        Matrix {
            rows: rows,
            cols: cols,
            data: d
        }
    }

    fn ones(rows: usize, cols: usize) -> Matrix {
        return Matrix::build_matrix(rows, cols, Matrix::_one);
    }

    fn rand(rows: usize, cols: usize) -> Matrix {
        return Matrix::build_matrix(rows, cols, rand::random).apply(|x: f32| x * 2.0 - 1.0);
    }

    fn from_vec(rows: usize, cols: usize, source: &Vec<f32>) -> Matrix {
        let mut d: Vec<f32> = Vec::with_capacity(rows * cols);

        for i in 0..(rows * cols) {
            d.push(source[i]);
        }

        Matrix {
            rows: rows,
            cols: cols,
            data: d
        }
    }

    fn mult(&self, other: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = Matrix::rand(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                result_matrix.data[i * other.cols + j] = 0.0;
                for k in 0..self.cols {
                    result_matrix.data[i * other.cols + j] += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
            }
        }
        return result_matrix;
    }


    fn apply<F>(&self, f: F) -> Matrix where F: Fn(f32) -> f32{
        // yum, closures and function passing
        let mut result_matrix: Matrix = Matrix::ones(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_matrix[(i,j)] = f(self[(i,j)]);
            }
        }
        return result_matrix;
    }

    fn iadd(&self, other: &Matrix) {
    }
}

impl Index<usize> for Matrix {
    type Output = f32;

    fn index(&self, idx: usize) -> &f32 {
        return &self.data[idx];
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, coords: (usize, usize)) -> &f32 {
        let (i, j) = coords;
        return &self.data[self.cols * i + j];
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, coords: (usize, usize)) -> &mut f32 {
        let (i, j) = coords;
        return &mut self.data[self.cols * i + j];
    }
}


struct NeuralNetwork {
    num_layers: usize,
    layer_sizes: Vec<usize>,
    connections: Vec<Matrix>,
    learning_rate: f32,

}

impl NeuralNetwork {
    fn new(num_layers: usize, layer_sizes: Vec<usize>, learning_rate: f32) -> NeuralNetwork {
        /* While this isn't enforced, for any reasonable network, num_layers
         * should be >= 3, where layer 1 is the input size, and layer 3 is the
         * output size. This isn't enforced because I'm lazy.
         */

        // Each 'layer' is actually an MxN matrix, where M is the number of neurons
        // in layer L, and N is the number of neurons in layer L + 1.
        // That means that connections should actually be a tensor of order 3, of shape
        // (num_layers - 1) * layer_L.len * layer_L+1.len
        let mut connections: Vec<Matrix> = (0..(num_layers-1)).map(|l| 
            Matrix::rand(layer_sizes[l], layer_sizes[l+1])).collect();

        NeuralNetwork { 
            num_layers: num_layers,
            layer_sizes: layer_sizes, 
            connections: connections,
            learning_rate: learning_rate,
        }
    }

    fn train_once(&self, X: &Vec<f32>, Y: &Vec<f32>) {
        let xs = X.len();
        let ys = Y.len();
        if !(xs == self.layer_sizes[0] && ys == self.layer_sizes[self.num_layers - 1]) {
            panic!("Incorrect layer size")
        }
        
        let pred = self._feed_forward(X);
        let ref pred_vec = pred.data;
        self._backpropogate(Y, pred_vec);
    }

    fn _feed_forward(&self, X: &Vec<f32>) -> Matrix {
        //I can probably use fold here, maybe?
        let input: Matrix = Matrix::from_vec(1, X.len(), X);
        return self.connections.iter().fold(input, |i: Matrix, l: &Matrix| i.mult(l).apply(NeuralNetwork::_sigmoid));
    }

    fn _sigmoid(t: f32) -> f32 {
        let e: f32 = 2.71828182846;
        return 1.0 / (1.0 + e.powf(t));
    }

    fn _cost(&self, Y: &Vec<f32>, pred: &Vec<f32>) -> f32 {
        //just rmse for now
        let mut accum = 0.0;
        let outs = Y.len() as f32;
        for i in 0..Y.len() {
            accum += (Y[i] - pred[i]) * (Y[i] - pred[i]) / outs;
        }

        return accum.sqrt();
    }

    fn _d_sigmoid(t: f32) -> f32 {
        return NeuralNetwork::_sigmoid(t) * (1.0 - NeuralNetwork::_sigmoid(t));
    }

    fn _backpropogate(&self, Y: &Vec<f32>, pred: &Vec<f32>) {
        // handle last layer specially
    }

    fn predict(&self, X: &Vec<f32>) -> Vec<f32>{
        let result: Vec<f32> = X.clone();
        return result;
    }

    fn train(&self, samples: &Vec<Vec<f32>>, outs: &Vec<Vec<f32>> ) {
        for i in 0..samples.len() {
            self.train_once(&samples[i], &outs[i]);
        }
    }
}
