extern crate rand;

fn main() {
    let nn = NeuralNetwork::new(3, vec![2,3,1]);
}


struct Matrix {
   rows: usize,
   cols: usize,
   data: Vec<Vec<f32>>
}


impl Matrix {
    fn new(rows: usize, cols: usize) -> Matrix {
        let mut d: Vec<Vec<f32>> = (0..rows).map(|i|
            Vec::with_capacity(cols)).collect();

        Matrix {
            rows: rows,
            cols: cols,
            data: d
        }
    }
    fn mult(&self, other: Matrix) -> Matrix {
        return Matrix::new(1,1);
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
            Matrix::new(layer_sizes[l], layer_sizes[l+1])).collect();

        for conn in connections.iter_mut() {
            for neuron in conn.data.iter_mut() {
                for weight in neuron.iter_mut() {
                    *weight = rand::random();
                }
            }
        }
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
