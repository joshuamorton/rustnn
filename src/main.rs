extern crate rand;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::RangeFull;

fn main() {
    let m = Matrix::rand(2, 3);
    let n = Matrix::rand(3, 2);
    let k = m.mult(&n);
    print!("{:?}\n", k._p());
    print!("{:?}\n", m._p());
    print!("{:?}\n", Matrix::from_vec(2, 3, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])._p());
    print!("{:?}\n", Matrix::from_vec(2, 3, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).apply(|x: f32| x + 1.0)._p());

    //let mut nn = NeuralNetwork::new(vec![2,4,1], 0.005);
    let mut nn = NeuralNetwork::new(vec![1,1], 0.005);

    for i in 0..100000 {
        nn.train_once(&vec![0.0], &vec![0.0]);
        nn.train_once(&vec![0.25], &vec![0.5]);
        nn.train_once(&vec![0.5], &vec![1.0]);
        nn.train_once(&vec![0.75], &vec![1.5]);
        nn.train_once(&vec![1.0], &vec![2.0]);
        nn.train_once(&vec![5.0], &vec![10.0]);
        if i % 1000 == 0 {
            print!("{:?}\n", nn.connections[0]._p());
        }
    }

    print!("{:?}\n", nn.predict(&vec![0.0]));
    print!("{:?}\n", nn.predict(&vec![0.25]));
    print!("{:?}\n", nn.predict(&vec![0.5]));
    print!("{:?}\n", nn.predict(&vec![0.75]));
    print!("{:?}\n", nn.predict(&vec![1.0]));


    nn = NeuralNetwork::new(vec![2,3,1], 0.005);

    for i in 0..100000 {
        nn.train_once(&vec![0.0, 0.0], &vec![0.0]);
        nn.train_once(&vec![1.0, 1.0], &vec![0.0]);
        nn.train_once(&vec![1.0, 0.0], &vec![1.0]);
        nn.train_once(&vec![0.0, 1.0], &vec![1.0]);
        if i % 1000 == 0 {
            print!("{:?}\n", nn.connections[0]._p());
        }
    }

    print!("{:?}\n", nn.predict(&vec![0.0, 0.0]));
    print!("{:?}\n", nn.predict(&vec![0.0, 1.0]));
    print!("{:?}\n", nn.predict(&vec![1.0, 0.0]));
    print!("{:?}\n", nn.predict(&vec![1.0, 1.0]));
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
        for _ in 0..(rows * cols) {
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

    fn vec_to_mat(source: &Vec<f32>) -> Matrix {
        let d = source.clone();
        Matrix {
            rows: 1,
            cols: d.len(),
            data: d
        }
    }

    fn to_vec(&self) -> Vec<f32> {
        return self.data.clone();
    }

    fn mult(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrix dimensions do not match: {:?} and {:?}", self.shape(), other.shape())
        }
        let mut result_matrix: Matrix = Matrix::ones(self.rows, other.cols);
        //swap this to good indexing at some point
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

    fn sub(&self, other: &Matrix) -> Matrix {
        //they need to be the same size, so this still works
        let mut result_matrix: Matrix = Matrix::ones(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                result_matrix[(i, j)] = self[(i, j)] - other[(i, j)];
            }
        }
        return result_matrix;
    }

    fn vec_mult(&self, other: &Vec<f32>) -> Matrix {
        let input_mat = Matrix::vec_to_mat(other);
        return input_mat.mult(self);
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

    fn emult(&self, other: &Matrix) -> Matrix {  // elementwise multiply
        let mut result_matrix: Matrix = Matrix::ones(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_matrix[(i,j)] = self[(i,j)] * other[(i,j)];
            }
        }
        return result_matrix;
    }

    fn _p(&self) -> Vec<Vec<f32>> {
        // pretty print ish
        let mut d: Vec<Vec<f32>> = (0..self.rows).map(|_| Vec::with_capacity(self.cols)).collect();
        let mut rind: usize = 0;
        for ref mut row in d.iter_mut() {
            for i in 0..self.cols {
                row.push(self.data[rind * self.cols + i]);
            }
            rind += 1;
        }
        return d;
    }

    fn tp(&self) -> Matrix {  //transpose
        let mut result_matrix: Matrix = Matrix::ones(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_matrix[(j, i)] = self[(i, j)];
            }
        }
        return result_matrix;
    }

    fn shape(&self) -> Vec<usize> {
        return vec![self.rows, self.cols];
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

impl Index<(usize, RangeFull)> for Matrix {
    type Output = [f32];

    fn index(&self, coords: (usize, RangeFull)) -> &[f32] {
        let (i, _) = coords;
        return &self.data[i * self.cols.. (i+1) * self.cols];
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
    regularization_param: f32,

}

impl NeuralNetwork {
    fn new(layer_sizes: Vec<usize>, learning_rate: f32) -> NeuralNetwork {
        /* While this isn't enforced, for any reasonable network, num_layers
         * should be >= 3, where layer 1 is the input size, and layer 3 is the
         * output size. This isn't enforced because I'm lazy.
         */

        // Each 'layer' is actually an MxN matrix, where M is the number of neurons
        // in layer L, and N is the number of neurons in layer L + 1.
        // That means that connections should actually be a tensor of order 3, of shape
        // (num_layers - 1) * layer_L.len * layer_L+1.len
    
        //TODO: add bias
        let connections: Vec<Matrix> = (0..(layer_sizes.len()-1)).map(|l| 
            Matrix::rand(layer_sizes[l], layer_sizes[l+1])).collect();

        NeuralNetwork { 
            num_layers: layer_sizes.len(),
            layer_sizes: layer_sizes, 
            connections: connections,
            learning_rate: learning_rate,
            regularization_param: 0.0001,
        }
    }

    fn train_once(&mut self, x: &Vec<f32>, y: &Vec<f32>) {
        let xs = x.len();
        let ys = y.len();
        if !(xs == self.layer_sizes[0] && ys == self.layer_sizes[self.num_layers - 1]) {
            panic!("Incorrect layer size")
        }
        let mut partial_outputs = Vec::with_capacity(self.connections.len() + 1);
        //partial_outputs[i] == input[i]

        partial_outputs.push(x.clone());
        // feed forward and save the intermediate layer values
        for layer in &self.connections {
            let idx = partial_outputs.len() - 1;
            let new_layer = layer.vec_mult(&partial_outputs[idx]).apply(NeuralNetwork::_sigmoid);
            partial_outputs.push(new_layer.to_vec());
        }
        self._backpropogate(y, partial_outputs);
    }

    fn _backpropogate(& mut self, y: &Vec<f32>, inputs: Vec<Vec<f32>>) {
        /* The delta rule states that ΔW_ji = a(t_j - y_j) * g'(h_j) * x_i
         * That is, the delta from neuron js ith weight is the above where (
         * according to wikipedia)
         * a = learning rate
         * t_j = target output
         * y_j = actual output
         * g'  = derivative of the activation function
         * h_j = weighted sum of inputs to this neuron (ie. pre-activation y_j)
         * x_i = the ith input
         *
         
         * Then all we need is T-Y when the layer is an inner layer. This is gradients of the next
         * layer multiplied by the weights of the current layer.
         *
         * partial_inputs[i] == input[i], meaning o[i].mult(connections[i]) is the output of layer
         * i.
         */
        let mut errors: Vec<Matrix> = Vec::with_capacity(self.num_layers);  // row vectors
        let error = Matrix::vec_to_mat(&inputs[inputs.len() - 1]).sub(&Matrix::vec_to_mat(y));
        errors.push(error.emult(&Matrix::vec_to_mat(&inputs[inputs.len() - 2]).apply(NeuralNetwork::_d_sigmoid)));
        for idx in (1..self.connections.len()).rev() {
            let error = self.connections[idx].mult(&errors[0].tp()).tp();
            errors.insert(0, error.emult(&Matrix::vec_to_mat(&inputs[idx]).apply(NeuralNetwork::_d_sigmoid)));
        }
        
        if errors.len() != self.connections.len() {
            panic!("Its broken");
        }

        let mut deltas: Vec<Matrix> = Vec::with_capacity(self.num_layers);
        for idx in 0..self.connections.len() {
            deltas.push(Matrix::vec_to_mat(&inputs[idx]).tp().mult(&errors[idx].apply(|x:f32| x * self.learning_rate)));
            //deltas.push(errors[idx].tp().mult(&Matrix::vec_to_mat(&i[idx])).apply(|x:f32| x * self.learning_rate).tp());
        }

        //update weights and regularize
        for idx in 0..self.connections.len() {
            self.connections[idx] = self.connections[idx].sub(&deltas[idx].apply(|x:f32| x - self.regularization_param));
        }
    }

    fn _solo_feed_forward(&self, x: &Vec<f32>) -> Matrix {
        //god this is pretty
        let input: Matrix = Matrix::vec_to_mat(x);
        return self.connections.iter().fold(input, |i: Matrix, l: &Matrix| i.mult(&l.apply(NeuralNetwork::_sigmoid)));
    }

    fn _sigmoid(t: f32) -> f32 {
        let e: f32 = 2.71828182846;
        return 1.0 / (1.0 + e.powf(-t));
    }

    fn _cost(&self, y: &Vec<f32>, pred: &Vec<f32>) -> f32 {
        //just rmse for now
        let mut accum = 0.0;
        let outs = y.len() as f32;
        for i in 0..y.len() {
            accum += (y[i] - pred[i]) * (y[i] - pred[i]) / outs;
        }

        return accum.sqrt();
    }

    fn _d_sigmoid(t: f32) -> f32 {
        return NeuralNetwork::_sigmoid(t) * (1.0 - NeuralNetwork::_sigmoid(t));
    }

    fn predict(&self, x: &Vec<f32>) -> Vec<f32>{
       return self._solo_feed_forward(x)._p()[0].clone();
    }

    fn train(&mut self, samples: &Vec<Vec<f32>>, outs: &Vec<Vec<f32>> ) {
        for i in 0..samples.len() {
            self.train_once(&samples[i], &outs[i]);
        }
    }
}
