extern crate rand;

fn main() {
    let m = Matrix::rand(2, 3);
    let n = Matrix::rand(3, 2);
    let k = m.mult(n);
    print!("{:?}\n", k.data);
    print!("{:?}\n", Matrix::from_vec(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).data);
    let nn = NeuralNetwork::new(3, vec![2,3,1]);
}


struct Matrix {
   rows: usize,
   cols: usize,
   data: Vec<Vec<f32>>
}


impl Matrix {
    
    fn ONE() -> f32 {
        return 1.0;
    }

    fn build_matrix(rows: usize, cols: usize, gen: fn() -> f32) -> Matrix {
        let mut d: Vec<Vec<f32>> = (0..rows).map(|i| Vec::with_capacity(cols)).collect();

        for ref mut row in d.iter_mut() {
            for i in 0..cols {
                row.push(gen());
            }
        }

        Matrix {
            rows: rows,
            cols: cols,
            data: d
        }
    }

    fn ones(rows: usize, cols: usize) -> Matrix {
        return Matrix::build_matrix(rows, cols, Matrix::ONE);
    }

    fn rand(rows: usize, cols: usize) -> Matrix {
        return Matrix::build_matrix(rows, cols, rand::random);
    }

    fn from_vec(rows: usize, cols: usize, source: Vec<f32>) -> Matrix {
        let mut d: Vec<Vec<f32>> = (0..rows).map(|i| Vec::with_capacity(cols)).collect();

        let mut rind: usize = 0;
        for ref mut row in d.iter_mut() {
            for i in 0..cols {
                row.push(source[rind * cols + i]);
            }
            rind += 1;
        }

        Matrix {
            rows: rows,
            cols: cols,
            data: d
        }
    }

    fn mult(&self, other: Matrix) -> Matrix {
        // I'm being bad and won't errorcheck for now
        
        // I'm also being lazy and implementing a naive matmul for now
        
        let mut result_matrix: Matrix = Matrix::rand(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                result_matrix.data[i][j] = 0.0;
                for k in 0..self.cols {
                    result_matrix.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        return result_matrix;
    }
}


struct NeuralNetwork {
    num_layers: usize,
    layer_sizes: Vec<usize>,
    connections: Vec<Matrix>

}

impl NeuralNetwork {
    fn new(num_layers: usize, layer_sizes: Vec<usize>) -> NeuralNetwork {
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
            connections: connections
        }
    }

    fn train_once(&self, X: Vec<f32>, Y: Vec<f32>) {
        let xs = X.len();
        let ys = Y.len();
        if !(xs == self.layer_sizes[0] && ys == self.layer_sizes[self.num_layers - 1]) {
            panic!("Incorrect layer size")
        }
        
        self._feed_forward(X);
    }

    fn _feed_forward(&self, X: Vec<f32>) {

    }

    fn _cost() {
    }

    fn _backpropogate() {
    }

    fn predict() {
    }

    fn train() {
    }
}
